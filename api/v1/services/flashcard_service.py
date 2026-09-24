"""
Flashcard business logic (extracted from routes/flashcards.py).

OpenAI 2-step flow (per user choice: summarize then gen):
  raw file text -> clean -> llm_service.summarize_text()
               -> llm_service.generate_flashcards() -> DeckCard rows
"""
import random
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

from sqlalchemy.orm import Session

from api.utils.file_processing import (
    ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE_MB,
    clean_extracted_text,
    extract_text_from_file,
)
from api.v1.models.deck import Deck
from api.v1.models.deck_card import DeckCard
from api.v1.services import llm_service
from api.core.exceptions import NotFoundException, BadRequestException


def _get_owned_deck(db: Session, user_id: UUID, deck_id: UUID) -> Deck:
    deck = db.query(Deck).filter_by(deck_id=deck_id, user_id=user_id).first()
    if not deck:
        raise NotFoundException("Deck not found or access denied.")
    return deck


def _get_owned_card(db: Session, user_id: UUID, card_id: UUID) -> DeckCard:
    card = db.query(DeckCard).filter_by(card_id=card_id, user_id=user_id).first()
    if not card:
        raise NotFoundException("Card not found.")
    return card


async def generate_flashcards_from_upload(
    db: Session,
    user_id: UUID,
    deck_id: UUID,
    filename: str,
    contents: bytes,
    num_flashcards: int | None = None,
    max_cards: int = 25,
) -> dict:
    _get_owned_deck(db, user_id, deck_id)

    extension = Path(filename or "").suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise BadRequestException(f"Unsupported file type: {extension or '(none)'}.")

    file_size_mb = len(contents) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        raise BadRequestException(
            f"File too large. Maximum size is {MAX_FILE_SIZE_MB}MB."
        )
    if not contents.strip():
        raise BadRequestException("Uploaded file is empty.")

    # Secure temp file (fixes old /tmp/{filename} traversal/collision).
    with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as tmp:
        tmp.write(contents)
        temp_path = tmp.name
    try:
        raw_text = extract_text_from_file(temp_path)
    finally:
        Path(temp_path).unlink(missing_ok=True)

    clean_text = clean_extracted_text(raw_text)
    if not clean_text.strip():
        raise BadRequestException("No readable text found in file.")

    # Step 1: summarize, Step 2: Q/A generation — both via OpenAI.
    summary_chunks = await llm_service.summarize_text(clean_text)
    if num_flashcards:
        summary_chunks = summary_chunks[: min(num_flashcards, max_cards)]
    else:
        summary_chunks = summary_chunks[:max_cards]

    if not summary_chunks:
        raise BadRequestException("Could not summarize the uploaded file.")

    cards = await llm_service.generate_flashcards(summary_chunks)
    if num_flashcards:
        cards = cards[: min(num_flashcards, max_cards)]

    saved = 0
    for idx, card in enumerate(cards):
        question = (card.get("question") or "").strip()
        answer = (card.get("answer") or "").strip()
        if not question or not answer:
            continue
        db.add(
            DeckCard(
                card_id=uuid4(),
                deck_id=deck_id,
                user_id=user_id,
                question=question,
                answer=answer,
                card_with_answer=f"Q: {question}\nA: {answer}",
                source_summary=(card.get("summary") or "").strip() or None,
                source_chunk=(card.get("summary") or "").strip() or None,
                chunk_index=idx,
            )
        )
        saved += 1
    db.commit()

    return {
        "message": "Flashcards generated and saved successfully.",
        "filename": filename,
        "deck_id": str(deck_id),
        "num_flashcards": saved,
        "summaries": summary_chunks,
        "flashcards": cards,
    }


async def regenerate_adaptive_drills(
    db: Session,
    user_id: UUID,
    deck_id: UUID,
    mode: str = "wrong",
    min_wrong_count: int = 2,
    max_drills: int = 10,
) -> dict:
    _get_owned_deck(db, user_id, deck_id)

    query = db.query(DeckCard).filter_by(deck_id=deck_id, user_id=user_id)
    if mode == "wrong":
        query = query.filter(DeckCard.wrong_count >= min_wrong_count)
    elif mode == "bookmark":
        query = query.filter(DeckCard.is_bookmarked.is_(True))
    else:
        raise BadRequestException("Mode must be 'wrong' or 'bookmark'.")

    relevant = query.limit(max_drills).all()
    if not relevant:
        raise NotFoundException("No cards match the adaptive filter.")

    summary_chunks = [c.source_summary for c in relevant if c.source_summary]
    raw_chunks = [c.source_chunk for c in relevant if c.source_chunk]
    if not summary_chunks:
        # Fall back to Q/A text when AI summaries were never stored.
        summary_chunks = [f"Q: {c.question}\nA: {c.answer}" for c in relevant]

    cards = await llm_service.generate_flashcards(summary_chunks[:max_drills])

    saved_cards: list[DeckCard] = []
    for idx, card in enumerate(cards):
        question = (card.get("question") or "").strip()
        answer = (card.get("answer") or "").strip()
        if not question or not answer:
            continue
        obj = DeckCard(
            card_id=uuid4(),
            deck_id=deck_id,
            user_id=user_id,
            question=question,
            answer=answer,
            card_with_answer=f"Q: {question}\nA: {answer}",
            source_summary=(card.get("summary") or "").strip() or None,
            source_chunk=raw_chunks[idx] if idx < len(raw_chunks) else None,
            chunk_index=idx,
        )
        db.add(obj)
        saved_cards.append(obj)
    db.commit()

    return {
        "message": f"{len(saved_cards)} adaptive flashcards regenerated and saved.",
        "cards": [
            {"question": c.question, "answer": c.answer, "chunk_index": c.chunk_index}
            for c in saved_cards
        ],
        "summary_points": summary_chunks,
        "source_chunks_used": raw_chunks,
        "status": "success",
    }


def get_practice_card(
    db: Session, user_id: UUID, deck_id: UUID,
    only_unstudied: bool = False, bookmarked_only: bool = False,
) -> dict:
    query = db.query(DeckCard).filter_by(deck_id=deck_id, user_id=user_id)
    if only_unstudied:
        query = query.filter_by(is_studied=False)
    if bookmarked_only:
        query = query.filter_by(is_bookmarked=True)
    cards = query.all()
    if not cards:
        raise NotFoundException("No flashcards found for practice.")
    card = random.choice(cards)
    return {"card_id": str(card.card_id), "question": card.question}


def record_review(db: Session, user_id: UUID, card_id: UUID, is_correct: bool) -> dict:
    card = _get_owned_card(db, user_id, card_id)
    card.times_reviewed = (card.times_reviewed or 0) + 1
    card.last_reviewed = datetime.now(timezone.utc)
    card.is_studied = True
    if is_correct:
        card.correct_count = (card.correct_count or 0) + 1
    else:
        card.wrong_count = (card.wrong_count or 0) + 1
    db.commit()
    return {
        "message": "Response recorded.",
        "card_id": str(card.card_id),
        "is_correct": is_correct,
        "times_reviewed": card.times_reviewed,
        "correct_count": card.correct_count,
        "wrong_count": card.wrong_count,
        "last_reviewed": card.last_reviewed.isoformat(),
    }


def grade_quiz_answer(user_answer: str, correct_answer: str) -> bool:
    return user_answer.strip().lower() == correct_answer.strip().lower()
