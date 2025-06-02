from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from api.utils.authentication import get_current_user
from api.utils.file_processing import estimate_flashcard_count, parse_flashcard_response, _clean_math_text, _filter_and_clean, clean_and_structure_text, extract_text_from_file, chunk_file_by_type
from api.v1.models import deck_card as card_models
from api.v1.schemas import flashcards as schemas
from api.v1.models import deck as models
from api.v1.models.user import User
from sqlalchemy.orm import Session
from sqlalchemy import func
from api.db.database import get_db
from pptx import Presentation
from PyPDF2 import PdfReader
from docx import Document
from typing import List, Optional
from uuid import UUID
import os, httpx, re, io, asyncio, random
from datetime import datetime

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


# ✅ Get all cards in a specific deck
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
):
    flashcard_url = f"{model_chat_endpoint}/flashcard"
    summarization_url = f"{model_util_endpoint}/flashcard-summarization"

    db_deck = db.query(models.Deck).filter_by(deck_id=deck_id, user_id=current_user.user_id).first()
    if not db_deck:
        raise HTTPException(status_code=404, detail="Deck not found.")

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
            summary_resp = await client.post(summarization_url, json={"chunks": notes.chunks})
        summary_resp.raise_for_status()
        key_points = summary_resp.json().get("points", [])
    except httpx.HTTPError as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")

    if not key_points:
        raise HTTPException(status_code=400, detail="No key points returned from summarization.")

    # 🧼 Clean key points
    cleaned_key_points = [
        re.sub(r"^\d+\.\s*", "", re.sub(r"Here are.*?:", "", point)).strip()
        for point in key_points
        if point.strip()
    ][:num_flashcards]

    if not cleaned_key_points:
        raise HTTPException(status_code=400, detail="No valid key points to process.")

    # 🔁 Generate flashcards
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
                await asyncio.sleep(2 ** attempt)
        return None

    qa_pairs = []
    batch_size = 3
    semaphore = asyncio.Semaphore(2)
    for i in range(0, len(cleaned_key_points), batch_size):
        batch = cleaned_key_points[i:i + batch_size]
        results = await asyncio.gather(*[generate_card(p, i + j, semaphore) for j, p in enumerate(batch)])
        qa_pairs.extend([qa for qa in results if qa])
        await asyncio.sleep(2)  # throttle between batches

    if not qa_pairs:
        raise HTTPException(status_code=400, detail="No valid flashcards generated.")
    if len(qa_pairs) < num_flashcards:
        print(f"⚠️ Only {len(qa_pairs)} out of {num_flashcards} flashcards generated.")

    flashcards_to_save = []
    for i, qa in enumerate(qa_pairs):
        flashcards_to_save.append(
            card_models.DeckCard(
                deck_id=deck_id,
                user_id=current_user.user_id,
                question=qa["question"],
                answer=qa["answer"],
                card_with_answer=f"Q: {qa['question']}\nA: {qa['answer']}",
                source_summary=cleaned_key_points[i] if i < len(cleaned_key_points) else None,
                source_chunk=notes.chunks[i] if i < len(notes.chunks) else None,
                chunk_index=i
            )
        )

    db.bulk_save_objects(flashcards_to_save)
    db.commit()

    return {
        "message": f"{len(qa_pairs)} of {num_flashcards} flashcards generated and saved.",
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
    current_user: models.User = Depends(get_current_user),
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

    # 🔢 Determine adaptive number of flashcards
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
    )


@flashcards.get("/decks/{deck_id}/practice")
def get_practice_card(
    deck_id: UUID,
    only_unstudied: bool = False,
    bookmarked_only: bool = False,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    query = db.query(card_models.DeckCard).filter_by(deck_id=deck_id, user_id=current_user.user_id)
    if only_unstudied:
        query = query.filter_by(is_studied=False)
    if bookmarked_only:
        query = query.filter_by(is_bookmarked=True)

    cards = query.all()
    if not cards:
        raise HTTPException(status_code=404, detail="No flashcards found for practice.")

    card = random.choice(cards)
    question = card.card_with_answer.split("\n")[0][3:].strip()  # "Q: ..."

    return {
        "card_id": str(card.card_id),
        "question": question
    }


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
            continue

        card.times_reviewed += 1
        card.is_studied = True
        card.last_reviewed = datetime.utcnow()

        if submission.is_correct:
            card.correct_count += 1
            correct += 1
        else:
            card.wrong_count += 1
            wrong += 1

        detailed.append(
            {
                "card_id": str(card.card_id),
                "question": card.question,
                "correct_answer": card.answer,
                "user_answer": submission.user_answer.strip(),
                "is_correct": submission.is_correct,
            }
        )

    db.commit()

    return schemas.QuizResults(
        total_questions=len(responses),
        correct=correct,
        wrong=wrong,
        detailed_results=detailed,
    )
