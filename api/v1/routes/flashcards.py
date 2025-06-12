from api.utils.file_processing import estimate_flashcard_count, extract_text_from_file, chunk_file_by_type
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from api.utils.authentication import get_current_user
from api.v1.models import deck_card as card_models
from api.v1.schemas import flashcards as schemas
from api.v1.models import deck as models
from api.v1.models.user import User
from sqlalchemy.orm import Session
from sqlalchemy import func
from api.db.database import get_db
from typing import List, Optional, Literal
from uuid import UUID
import os, httpx, re, asyncio, random
from datetime import datetime
import logging

flashcards = APIRouter(prefix="/flashcard", tags=["Flashcards"])
model_util_endpoint = os.getenv("MODEL_UTILITY")
model_chat_endpoint = os.getenv("MODEL_ENDPOINT")


@flashcards.post("/decks/", response_model=schemas.DeckOut)
def create_deck(
    deck: schemas.DeckCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    existing = (
        db.query(models.Deck)
        .filter_by(user_id=current_user.user_id, title=deck.title)
        .first()
    )
    if existing:
        raise HTTPException(
            status_code=400, detail="A deck with this title already exists."
        )

    db_deck = models.Deck(title=deck.title, user_id=current_user.user_id)
    db.add(db_deck)
    db.commit()
    db.refresh(db_deck)
    return db_deck


@flashcards.get("/decks/", response_model=List[schemas.DeckOut])
def get_user_decks(
    include_card_count: Optional[bool] = False,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not include_card_count:
        return db.query(models.Deck).filter_by(user_id=current_user.user_id).all()

    # With card count annotation
    decks = (
        db.query(
            models.Deck.deck_id,
            models.Deck.title,
            models.Deck.date_created,
            func.count(card_models.DeckCard.card_id).label("card_count"),
        )
        .outerjoin(
            card_models.DeckCard, models.Deck.deck_id == card_models.DeckCard.deck_id
        )
        .filter(models.Deck.user_id == current_user.user_id)
        .group_by(models.Deck.deck_id)
        .all()
    )

    # Optional: add to DeckOut schema if you want to show `card_count` in response
    return [
        {
            "deck_id": deck.deck_id,
            "title": deck.title,
            "date_created": deck.date_created,
            "card_count": deck.card_count,
        }
        for deck in decks
    ]


# ✅ Add flashcards to a specific deck
@flashcards.post("/decks/{deck_id}/cards/")
def add_cards_to_deck(
    deck_id: UUID,
    payload: schemas.AddCards,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    db_deck = (
        db.query(models.Deck)
        .filter_by(deck_id=deck_id, user_id=current_user.user_id)
        .first()
    )
    if not db_deck:
        raise HTTPException(status_code=404, detail="Deck not found.")

    # Create DeckCard instances
    cards = []
    for card in payload.cards:
        question = card.question.strip()
        answer = card.answer.strip()

        if not question or not answer:
            raise HTTPException(
                status_code=422,
                detail="Each card must include a non-empty question and answer.",
            )

        cards.append(
            card_models.DeckCard(
                deck_id=deck_id,
                user_id=current_user.user_id,
                question=question,
                answer=answer,
                card_with_answer=f"Q: {question}\nA: {answer}",
            )
        )

    db.add_all(cards)
    db.commit()

    return {
        "message": f"{len(cards)} cards added successfully.",
        "card_ids": [card.card_id for card in cards],
    }


async def generate_flashcards_from_notes(
    deck_id: UUID,
    notes: schemas.NoteChunks,
    num_flashcards: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    debug: Optional[str] = None,
):
    flashcard_url = f"{model_chat_endpoint}/flashcard"
    summarization_url = f"{model_util_endpoint}/flashcard-summarization"

    db_deck = (
        db.query(models.Deck)
        .filter_by(deck_id=deck_id, user_id=current_user.user_id)
        .first()
    )
    if not db_deck:
        raise HTTPException(status_code=404, detail="Deck not found.")

    try:
        logging.info(f"Sending {len(notes.chunks)} chunks for summarization to {summarization_url}")
        async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
            summary_resp = await client.post(
                summarization_url, json={"chunks": notes.chunks}
            )
        summary_resp.raise_for_status()
        key_points = summary_resp.json().get("points", [])
        logging.info(f"Received {len(key_points)} summary points from model utility.")
        for idx, point in enumerate(key_points):
            logging.info(f"Summary {idx+1}: {point}")
    except httpx.HTTPError as e:
        logging.error(f"Summarization failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")

    cleaned_key_points = [
        re.sub(r"^\d+\.\s*", "", re.sub(r"Here are.*?:", "", point)).strip()
        for point in key_points
        if point.strip()
    ]

    async def generate_card(point: str, index: int, semaphore: asyncio.Semaphore):
        for attempt in range(3):
            try:
                async with semaphore:
                    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
                        resp = await client.post(flashcard_url, json={"prompt": point})
                resp.raise_for_status()
                data = resp.json()
                q, a = data.get("question", "").strip(), data.get("answer", "").strip()
                if all([q, a]) and len(q.split()) >= 3 and len(a.split()) >= 3:
                    return {"question": q, "answer": a}
            except Exception:
                await asyncio.sleep(1 + attempt)
        return None

    semaphore = asyncio.BoundedSemaphore(min(10, num_flashcards))
    results = await asyncio.gather(
        *[generate_card(p, i, semaphore) for i, p in enumerate(cleaned_key_points)]
    )
    qa_pairs = [qa for qa in results if qa][:num_flashcards]

    if debug == "qa":
        return {
            "requested": num_flashcards,
            "generated": len(qa_pairs),
            "cards": qa_pairs,
            "summary_points": cleaned_key_points,
        }

    if not qa_pairs:
        raise HTTPException(status_code=400, detail="No valid flashcards generated.")

    flashcards_to_save = []
    for i, qa in enumerate(qa_pairs):
        flashcards_to_save.append(
            card_models.DeckCard(
                deck_id=deck_id,
                user_id=current_user.user_id,
                question=qa["question"],
                answer=qa["answer"],
                card_with_answer=f"Q: {qa['question']}\nA: {qa['answer']}",
                source_summary=(
                    cleaned_key_points[i] if i < len(cleaned_key_points) else None
                ),
                source_chunk=notes.chunks[i] if i < len(notes.chunks) else None,
                chunk_index=i,
            )
        )

    db.bulk_save_objects(flashcards_to_save)
    db.commit()

    return {
        "message": f"{len(qa_pairs)} of {num_flashcards} flashcards generated and saved.",
        "cards": qa_pairs,
        "status": "success",
    }


async def generate_flashcards_from_keypoints(
    deck_id: UUID,
    key_points: List[str],
    source_chunks: List[str],
    db: Session,
    current_user: User,
):
    flashcard_url = f"{model_chat_endpoint}/flashcard"

    # 🧼 Clean key points
    cleaned_key_points = [
        re.sub(r"^\d+\.\s*", "", point.strip()) for point in key_points if point.strip()
    ]

    async def generate_card(point: str, index: int, semaphore: asyncio.Semaphore):
        for attempt in range(3):
            try:
                async with semaphore:
                    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
                        resp = await client.post(flashcard_url, json={"prompt": point})
                resp.raise_for_status()
                response_data = resp.json()
                question = response_data.get("question", "").strip()
                answer = response_data.get("answer", "").strip()
                if question and answer and len(question) > 5:
                    return {"question": question, "answer": answer}
            except Exception:
                await asyncio.sleep(2**attempt)
        return None

    qa_pairs = []
    semaphore = asyncio.Semaphore(2)
    for i in range(0, len(cleaned_key_points), 3):
        batch = cleaned_key_points[i : i + 3]
        results = await asyncio.gather(
            *[generate_card(p, i + j, semaphore) for j, p in enumerate(batch)]
        )
        qa_pairs.extend([qa for qa in results if qa])
        await asyncio.sleep(1)

    if not qa_pairs:
        raise HTTPException(status_code=400, detail="No valid flashcards generated.")

    flashcards_to_save = []
    for i, qa in enumerate(qa_pairs):
        flashcards_to_save.append(
            card_models.DeckCard(
                deck_id=deck_id,
                user_id=current_user.user_id,
                question=qa["question"],
                answer=qa["answer"],
                card_with_answer=f"Q: {qa['question']}\nA: {qa['answer']}",
                source_summary=key_points[i] if i < len(key_points) else None,
                source_chunk=source_chunks[i] if i < len(source_chunks) else None,
                chunk_index=i,
            )
        )

    db.bulk_save_objects(flashcards_to_save)
    db.commit()

    return {
        "message": f"{len(qa_pairs)} flashcards generated directly from key points.",
        "cards": qa_pairs,
        "summary_points": cleaned_key_points,
        "status": "success",
    }


@flashcards.post("/decks/{deck_id}/generate-flashcards/")
async def upload_notes_and_generate_flashcards(
    deck_id: UUID,
    file: UploadFile = File(...),
    num_flashcards: Optional[int] = None,
    max_cards: Optional[int] = 25,
    debug: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    filename = file.filename
    ext = filename.split(".")[-1].lower()
    if ext not in {"txt", "pdf", "docx", "md", "pptx"}:
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    try:
        file_bytes = await file.read()
        full_text = extract_text_from_file(file_bytes, filename)
        full_text = re.sub(r"[^\x00-\x7F]+", " ", full_text)
        full_text = re.sub(r"\s{3,}", "\n\n", full_text).strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File parsing failed: {str(e)}")

    chunks = chunk_file_by_type(ext, file_bytes, full_text)
    # Log the chunks for debugging
    logging.info(f"Generated {len(chunks)} chunks from file '{filename}'.")
    for i, chunk in enumerate(chunks):
        logging.info(f"Chunk {i+1}: {chunk[:200]}{'...' if len(chunk) > 200 else ''}")

    if not chunks:
        raise HTTPException(status_code=400, detail="No valid text found in file.")

    if debug == "raw":
        return {
            "filename": filename,
            "extracted_text": full_text[:5000],
            "length": len(full_text),
        }
    if debug == "chunks":
        return {"filename": filename, "num_chunks": len(chunks), "chunks": chunks[:30]}

    if not num_flashcards:
        num_flashcards = estimate_flashcard_count(
            ext, file_bytes, min_per_unit=3, min_cards=5, max_cards=max_cards
        )

    return await generate_flashcards_from_notes(
        deck_id=deck_id,
        notes=schemas.NoteChunks(chunks=chunks),
        num_flashcards=num_flashcards,
        db=db,
        current_user=current_user,
        debug=debug,
    )


@flashcards.post("/decks/{deck_id}/regenerate-adaptive-drills/")
async def generate_adaptive_drills(
    deck_id: UUID,
    mode: Literal["wrong", "bookmark"] = "wrong",
    min_wrong_count: int = 2,
    max_drills: int = 10,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # 🔍 Step 1: Get the deck
    deck = (
        db.query(models.Deck)
        .filter_by(deck_id=deck_id, user_id=current_user.user_id)
        .first()
    )
    if not deck:
        raise HTTPException(status_code=404, detail="Deck not found.")

    # 📌 Step 2: Filter relevant cards
    cards_query = db.query(card_models.DeckCard).filter_by(
        deck_id=deck_id, user_id=current_user.user_id
    )

    if mode == "wrong":
        cards_query = cards_query.filter(
            card_models.DeckCard.wrong_count >= min_wrong_count
        )
    elif mode == "bookmark":
        cards_query = cards_query.filter(card_models.DeckCard.is_bookmarked == True)

    relevant_cards = cards_query.limit(max_drills).all()
    if not relevant_cards:
        raise HTTPException(
            status_code=404, detail="No cards match the adaptive filter."
        )

    # 🧠 Step 3: Extract summaries and source chunks
    key_points = [card.source_summary for card in relevant_cards if card.source_summary]
    source_chunks = [card.source_chunk for card in relevant_cards if card.source_chunk]

    if not key_points or not source_chunks:
        raise HTTPException(
            status_code=400, detail="Missing source summaries or chunks."
        )

    # ⚡ Step 4: Direct flashcard generation (no re-summarization)
    try:
        result = await generate_flashcards_from_keypoints(
            deck_id=deck_id,
            key_points=key_points,
            source_chunks=source_chunks,
            db=db,
            current_user=current_user,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Adaptive generation failed: {str(e)}"
        )

    return {
        "message": f"{len(result['cards'])} adaptive flashcards generated from {len(key_points)} entries.",
        "cards": result["cards"],
        "source_chunks_used": source_chunks,
        "summary_points": result["summary_points"],
        "status": "success",
    }


@flashcards.get("/decks/{deck_id}/practice")
def get_practice_card(
    deck_id: UUID,
    only_unstudied: bool = False,
    bookmarked_only: bool = False,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    query = db.query(card_models.DeckCard).filter_by(
        deck_id=deck_id, user_id=current_user.user_id
    )
    if only_unstudied:
        query = query.filter_by(is_studied=False)
    if bookmarked_only:
        query = query.filter_by(is_bookmarked=True)

    cards = query.all()
    if not cards:
        raise HTTPException(status_code=404, detail="No flashcards found for practice.")

    card = random.choice(cards)
    question = card.card_with_answer.split("\n")[0][3:].strip()  # "Q: ..."

    return {"card_id": str(card.card_id), "question": question}


@flashcards.get("/cards/{card_id}/reveal")
def reveal_card_answer(
    card_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    card = (
        db.query(card_models.DeckCard)
        .filter_by(card_id=card_id, user_id=current_user.user_id)
        .first()
    )
    if not card:
        raise HTTPException(status_code=404, detail="Card not found.")

    answer = card.answer
    answer = card.card_with_answer.split("\n")[1][3:].strip()  # "A: ..."
    return {"card_id": str(card.card_id), "answer": answer}


@flashcards.post("/cards/{card_id}/submit-response")
def submit_flashcard_response(
    card_id: UUID,
    review: schemas.FlashcardReviewInput,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    card = (
        db.query(card_models.DeckCard)
        .filter_by(card_id=card_id, user_id=current_user.user_id)
        .first()
    )
    if not card:
        raise HTTPException(status_code=404, detail="Flashcard not found.")

    # ⏱️ Update usage statistics
    card.times_reviewed += 1
    card.last_reviewed = datetime.utcnow()
    card.is_studied = True
    if review.is_correct:
        card.correct_count += 1
    else:
        card.wrong_count += 1

    db.commit()

    return {
        "message": "Response recorded.",
        "card_id": str(card.card_id),
        "is_correct": review.is_correct,
        "times_reviewed": card.times_reviewed,
        "correct_count": card.correct_count,
        "wrong_count": card.wrong_count,
        "last_reviewed": card.last_reviewed.isoformat(),
    }


# Analytics
@flashcards.post("/cards/{card_id}/mark-studied")
def mark_card_as_studied(
    card_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    card = (
        db.query(card_models.DeckCard)
        .filter_by(card_id=card_id, user_id=current_user.user_id)
        .first()
    )
    if not card:
        raise HTTPException(status_code=404, detail="Flashcard not found.")

    card.is_studied = True
    card.times_reviewed += 1
    card.last_reviewed = datetime.utcnow()
    db.commit()

    return {"message": "Flashcard marked as studied."}


@flashcards.post("/cards/{card_id}/bookmark")
def bookmark_card(
    card_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    card = (
        db.query(card_models.DeckCard)
        .filter_by(card_id=card_id, user_id=current_user.user_id)
        .first()
    )
    if not card:
        raise HTTPException(status_code=404, detail="Card not found.")

    card.is_bookmarked = True
    db.commit()
    return {"message": "Card bookmarked."}


@flashcards.post("/cards/{card_id}/unbookmark")
def unbookmark_card(
    card_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    card = (
        db.query(card_models.DeckCard)
        .filter_by(card_id=card_id, user_id=current_user.user_id)
        .first()
    )
    if not card:
        raise HTTPException(status_code=404, detail="Card not found.")

    card.is_bookmarked = False
    db.commit()
    return {"message": "Card unbookmarked."}


@flashcards.get("/cards/bookmarked", response_model=List[schemas.DeckCardOut])
def get_bookmarked_cards(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    cards = (
        db.query(card_models.DeckCard)
        .filter_by(user_id=current_user.user_id, is_bookmarked=True)
        .all()
    )
    return cards


@flashcards.get("/cards/unstudied", response_model=List[schemas.DeckCardOut])
def get_unstudied_cards(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    cards = (
        db.query(card_models.DeckCard)
        .filter_by(user_id=current_user.user_id, is_studied=False)
        .all()
    )
    return cards


@flashcards.get("/cards/hard", response_model=List[schemas.DeckCardOut])
def get_difficult_cards(
    min_reviews: int = 2,
    max_accuracy: float = 0.6,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    cards = (
        db.query(card_models.DeckCard)
        .filter(card_models.DeckCard.user_id == current_user.user_id)
        .filter(card_models.DeckCard.times_reviewed >= min_reviews)
        .all()
    )

    difficult = []
    for card in cards:
        total = card.correct_count + card.wrong_count
        accuracy = card.correct_count / total if total > 0 else 0
        if accuracy <= max_accuracy:
            difficult.append(card)

    return difficult


@flashcards.post("/cards/{card_id}/reset")
def reset_card_progress(
    card_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    card = (
        db.query(card_models.DeckCard)
        .filter_by(card_id=card_id, user_id=current_user.user_id)
        .first()
    )
    if not card:
        raise HTTPException(status_code=404, detail="Card not found.")

    card.is_studied = False
    card.times_reviewed = 0
    card.correct_count = 0
    card.wrong_count = 0
    card.last_reviewed = None

    db.commit()
    return {"message": "Card progress reset."}


# Quiz Endpoints
@flashcards.get("/decks/{deck_id}/quiz", response_model=schemas.QuizStartResponse)
def start_quiz(
    deck_id: UUID,
    limit: int = 5,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    cards = (
        db.query(card_models.DeckCard)
        .filter_by(deck_id=deck_id, user_id=current_user.user_id)
        .all()
    )
    if not cards:
        raise HTTPException(
            status_code=404, detail="No flashcards available in this deck."
        )

    selected = random.sample(cards, min(limit, len(cards)))
    quiz_cards = [
        schemas.QuizCard(
            card_id=card.card_id,
            question=card.question,
        )
        for card in selected
    ]
    return schemas.QuizStartResponse(deck_id=deck_id, cards=quiz_cards)


@flashcards.post("/quiz/submit", response_model=schemas.QuizResults)
def submit_quiz_self_graded(
    responses: List[schemas.QuizSubmission],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    correct = 0
    wrong = 0
    detailed = []

    for submission in responses:
        card = (
            db.query(card_models.DeckCard)
            .filter_by(card_id=submission.card_id, user_id=current_user.user_id)
            .first()
        )

        if not card:
            continue  # or raise HTTPException if strict

        # 🧠 Update card review stats
        card.times_reviewed += 1
        card.is_studied = True
        card.last_reviewed = datetime.utcnow()

        if submission.is_correct:
            card.correct_count += 1
            correct += 1
        else:
            card.wrong_count += 1
            wrong += 1

        # ✅ Add to results
        detailed.append(
            {
                "card_id": card.card_id,
                "your_answer": submission.user_answer.strip(),
                "correct_answer": card.answer,
                "correct": submission.is_correct,
            }
        )

    db.commit()

    return schemas.QuizResults(
        total_questions=len(responses),
        correct=correct,
        wrong=wrong,
        detailed_results=detailed,
    )


# Get all the cards in a deck
@flashcards.get("/decks/{deck_id}/get-cards", response_model=List[schemas.DeckCardOut])
def get_deck_cards(
    deck_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    cards = (
        db.query(card_models.DeckCard)
        .filter_by(deck_id=deck_id, user_id=current_user.user_id)
        .all()
    )
    if not cards:
        raise HTTPException(status_code=404, detail="No cards found in this deck.")

    return cards


# delete endpoint for decks and their cards
@flashcards.delete("/decks/{deck_id}")
def delete_deck(
    deck_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    deck = (
        db.query(models.Deck)
        .filter_by(deck_id=deck_id, user_id=current_user.user_id)
        .first()
    )
    if not deck:
        raise HTTPException(status_code=404, detail="Deck not found.")

    # Delete all cards in the deck
    db.query(card_models.DeckCard).filter_by(deck_id=deck_id).delete()

    # Delete the deck itself
    db.delete(deck)
    db.commit()

    return {"message": "Deck and all associated cards deleted successfully."}
