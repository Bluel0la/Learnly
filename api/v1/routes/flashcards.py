from api.utils.file_processing import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_MB, extract_text_from_file, clean_extracted_text
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from api.utils.authentication import get_current_user
from api.v1.models import deck_card as card_models
from api.v1.schemas import flashcards as schemas
from api.v1.models import deck as models
from api.v1.models.deck import Deck
from api.v1.models.deck_card import DeckCard
from api.v1.models.user import User
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import func
from api.db.database import get_db
from typing import List, Optional, Literal
from uuid import UUID
import os, httpx, re, asyncio, random, uuid, logging
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

@flashcards.post("/decks/{deck_id}/generate-flashcards/")
async def generate_flashcards_from_file(
    deck_id: UUID,
    file: UploadFile = File(...),
    num_flashcards: Optional[int] = None,
    max_cards: Optional[int] = 25,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # ✅ Validate deck ownership
    deck = db.query(Deck).filter(Deck.deck_id == deck_id, Deck.user_id == current_user.user_id).first()
    if not deck:
        raise HTTPException(status_code=404, detail="Deck not found or access denied.")

    # ✅ Validate file type
    filename = file.filename
    extension = os.path.splitext(filename)[1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    # ✅ Validate file size
    contents = await file.read()
    file_size_mb = len(contents) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB.")

    # ✅ Save file temporarily
    temp_path = f"/tmp/{filename}"
    with open(temp_path, "wb") as f:
        f.write(contents)

    # ✅ Extract and clean text
    try:
        raw_text = extract_text_from_file(temp_path)
        clean_text = clean_extracted_text(raw_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting file: {str(e)}")

    # ✅ Summarization step
    summarization_url = f"{model_util_endpoint}/flashcard-summarization"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                summarization_url,
                json={"text": clean_text, "max_tokens": 512},
                timeout=60.0,
            )
            response.raise_for_status()
            summary_result = response.json()
            summary_chunks = summary_result.get("summaries", [])
        except httpx.RequestError as e:
            raise HTTPException(status_code=500, detail=f"Summarization request error: {str(e)}")
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=f"Summarization failed: {e.response.text}")

    # ✅ Limit summary chunks
    if num_flashcards:
        summary_chunks = summary_chunks[:min(num_flashcards, len(summary_chunks), max_cards)]

    # ✅ Flashcard generation step
    flashcard_url = f"{model_chat_endpoint}/flashcard"
    async with httpx.AsyncClient() as client:
        try:
            flashcard_response = await client.post(
                flashcard_url,
                json={"summary_chunks": summary_chunks},
                timeout=60.0,
            )
            flashcard_response.raise_for_status()
            flashcards = flashcard_response.json().get("flashcards", [])
        except httpx.RequestError as e:
            raise HTTPException(status_code=500, detail=f"Flashcard request error: {str(e)}")
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=f"Flashcard generation failed: {e.response.text}")

    # ✅ Save flashcards to DB
    for idx, card in enumerate(flashcards):
        db_card = DeckCard(
            card_id=uuid.uuid4(),
            deck_id=deck_id,
            user_id=current_user.user_id,
            question=card.get("question"),
            answer=card.get("answer"),
            card_with_answer=f"Q: {card.get('question')}\nA: {card.get('answer')}",
            source_summary=card.get("summary"),
            source_chunk=card.get("summary"),
            chunk_index=idx
        )
        db.add(db_card)

    db.commit()

    return JSONResponse({
        "message": "Flashcards generated and saved successfully.",
        "filename": filename,
        "deck_id": str(deck_id),
        "num_flashcards": len(flashcards),
        "summaries": summary_chunks,
        "flashcards": flashcards,
    })


@flashcards.post("/decks/{deck_id}/regenerate-adaptive-drills/")
async def regenerate_adaptive_drills(
    deck_id: UUID,
    mode: Literal["wrong", "bookmark"] = "wrong",
    min_wrong_count: int = 2,
    max_drills: int = 10,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # ✅ Validate deck ownership
    deck = (
        db.query(Deck)
        .filter(Deck.deck_id == deck_id, Deck.user_id == current_user.user_id)
        .first()
    )
    if not deck:
        raise HTTPException(status_code=404, detail="Deck not found or access denied.")

    # ✅ Filter cards based on adaptive mode
    cards_query = db.query(DeckCard).filter(
        DeckCard.deck_id == deck_id, DeckCard.user_id == current_user.user_id
    )

    if mode == "wrong":
        cards_query = cards_query.filter(DeckCard.wrong_count >= min_wrong_count)
    elif mode == "bookmark":
        cards_query = cards_query.filter(DeckCard.is_bookmarked == True)

    relevant_cards = cards_query.limit(max_drills).all()
    if not relevant_cards:
        raise HTTPException(
            status_code=404, detail="No cards match the adaptive filter."
        )

    # ✅ Extract source summaries and chunks
    summary_chunks = [
        card.source_summary for card in relevant_cards if card.source_summary
    ]
    raw_chunks = [card.source_chunk for card in relevant_cards if card.source_chunk]

    if not summary_chunks or not raw_chunks:
        raise HTTPException(
            status_code=400, detail="Some cards are missing summary or chunk."
        )

    # ✅ Call flashcard generation model directly
    flashcard_url = f"{model_chat_endpoint}/flashcard"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                flashcard_url,
                json={"summary_chunks": summary_chunks},
                timeout=60.0,
            )
            response.raise_for_status()
            flashcards = response.json().get("flashcards", [])
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=500, detail=f"Flashcard request error: {str(e)}"
        )
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Flashcard generation failed: {e.response.text}",
        )

    # ✅ Save adaptive flashcards to DB
    saved_cards = []
    for idx, card in enumerate(flashcards):
        db_card = DeckCard(
            card_id=uuid.uuid4(),
            deck_id=deck_id,
            user_id=current_user.user_id,
            question=card.get("question"),
            answer=card.get("answer"),
            card_with_answer=f"Q: {card.get('question')}\nA: {card.get('answer')}",
            source_summary=card.get("summary"),
            source_chunk=raw_chunks[idx] if idx < len(raw_chunks) else None,
            chunk_index=idx,
        )
        db.add(db_card)
        saved_cards.append(db_card)

    db.commit()

    return {
        "message": f"{len(saved_cards)} adaptive flashcards regenerated and saved.",
        "cards": [
            {
                "question": card.question,
                "answer": card.answer,
                "chunk_index": card.chunk_index,
            }
            for card in saved_cards
        ],
        "summary_points": summary_chunks,
        "source_chunks_used": raw_chunks,
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
