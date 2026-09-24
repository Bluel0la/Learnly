"""
Flashcard routes — thin wrappers over flashcard_service.

AI flow (OpenAI-only): upload -> summarize -> generate (both via llm_service).
Manual card CRUD + practice/review/quiz stay local (no AI).
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query, status
from api.utils.authentication import get_current_user
from api.v1.schemas import flashcards as schemas
from api.v1.models.deck import Deck
from api.v1.models.deck_card import DeckCard
from api.v1.models.user import User
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import func
from api.db.database import get_db
from typing import List, Optional, Literal
from uuid import UUID
from datetime import datetime, timezone

from api.v1.services import flashcard_service
from api.core.exceptions import NotFoundException, BadRequestException

flashcards = APIRouter(prefix="/flashcard", tags=["Flashcards"])


@flashcards.post("/decks/", response_model=schemas.DeckOut, status_code=status.HTTP_201_CREATED)
def create_deck(
    deck: schemas.DeckCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    title = deck.title.strip()
    if not title:
        raise HTTPException(status_code=422, detail="Deck title must not be empty.")
    existing = db.query(Deck).filter_by(user_id=current_user.user_id, title=title).first()
    if existing:
        raise HTTPException(status_code=400, detail="A deck with this title already exists.")

    db_deck = Deck(title=title, user_id=current_user.user_id)
    db.add(db_deck)
    db.commit()
    db.refresh(db_deck)
    return schemas.DeckOut(
        deck_id=db_deck.deck_id, title=db_deck.title,
        date_created=db_deck.created_at, card_count=0,
    )


@flashcards.get("/decks/", response_model=List[schemas.DeckOut])
def get_user_decks(
    include_card_count: Optional[bool] = False,
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not include_card_count:
        decks = (
            db.query(Deck)
            .filter_by(user_id=current_user.user_id)
            .order_by(Deck.created_at.desc())
            .offset(skip)
            .limit(limit)
            .all()
        )
        return [
            schemas.DeckOut(
                deck_id=d.deck_id, title=d.title,
                date_created=d.created_at, card_count=None,
            )
            for d in decks
        ]

    rows = (
        db.query(
            Deck.deck_id,
            Deck.title,
            Deck.created_at.label("date_created"),
            func.count(DeckCard.card_id).label("card_count"),
        )
        .outerjoin(DeckCard, Deck.deck_id == DeckCard.deck_id)
        .filter(Deck.user_id == current_user.user_id)
        .group_by(Deck.deck_id, Deck.title, Deck.created_at)
        .order_by(Deck.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    return [
        schemas.DeckOut(
            deck_id=r.deck_id, title=r.title,
            date_created=r.date_created, card_count=r.card_count,
        )
        for r in rows
    ]


@flashcards.post("/decks/{deck_id}/cards/", status_code=status.HTTP_201_CREATED)
def add_cards_to_deck(
    deck_id: UUID,
    payload: schemas.AddCards,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    db_deck = db.query(Deck).filter_by(deck_id=deck_id, user_id=current_user.user_id).first()
    if not db_deck:
        raise HTTPException(status_code=404, detail="Deck not found.")

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
            DeckCard(
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
    num_flashcards: Optional[int] = Query(default=None, ge=1, le=25),
    max_cards: int = Query(default=25, ge=1, le=25),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    contents = await file.read()
    try:
        result = await flashcard_service.generate_flashcards_from_upload(
            db, current_user.user_id, deck_id,
            filename=file.filename or "upload",
            contents=contents,
            num_flashcards=num_flashcards,
            max_cards=max_cards,
        )
    except (NotFoundException, BadRequestException) as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)
    return JSONResponse(result)


@flashcards.post("/decks/{deck_id}/regenerate-adaptive-drills/")
async def regenerate_adaptive_drills(
    deck_id: UUID,
    mode: Literal["wrong", "bookmark"] = "wrong",
    min_wrong_count: int = Query(default=2, ge=1),
    max_drills: int = Query(default=10, ge=1, le=25),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        return await flashcard_service.regenerate_adaptive_drills(
            db, current_user.user_id, deck_id,
            mode=mode, min_wrong_count=min_wrong_count, max_drills=max_drills,
        )
    except (NotFoundException, BadRequestException) as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)


@flashcards.get("/decks/{deck_id}/practice")
def get_practice_card(
    deck_id: UUID,
    only_unstudied: bool = False,
    bookmarked_only: bool = False,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        return flashcard_service.get_practice_card(
            db, current_user.user_id, deck_id, only_unstudied, bookmarked_only
        )
    except NotFoundException as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)


@flashcards.get("/cards/{card_id}/reveal")
def reveal_card_answer(
    card_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    card = (
        db.query(DeckCard)
        .filter_by(card_id=card_id, user_id=current_user.user_id)
        .first()
    )
    if not card:
        raise HTTPException(status_code=404, detail="Card not found.")
    # Use stored answer column (old code re-parsed card_with_answer and crashed
    # when the newline was missing).
    return {"card_id": str(card.card_id), "answer": card.answer}


@flashcards.post("/cards/{card_id}/submit-response")
def submit_flashcard_response(
    card_id: UUID,
    review: schemas.FlashcardReviewInput,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        return flashcard_service.record_review(db, current_user.user_id, card_id, review.is_correct)
    except NotFoundException as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)


@flashcards.post("/cards/{card_id}/mark-studied")
def mark_card_as_studied(
    card_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    card = db.query(DeckCard).filter_by(card_id=card_id, user_id=current_user.user_id).first()
    if not card:
        raise HTTPException(status_code=404, detail="Flashcard not found.")

    card.is_studied = True
    card.times_reviewed = (card.times_reviewed or 0) + 1
    card.last_reviewed = datetime.now(timezone.utc)
    db.commit()
    return {"message": "Flashcard marked as studied."}


@flashcards.post("/cards/{card_id}/bookmark")
def bookmark_card(
    card_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    card = db.query(DeckCard).filter_by(card_id=card_id, user_id=current_user.user_id).first()
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
    card = db.query(DeckCard).filter_by(card_id=card_id, user_id=current_user.user_id).first()
    if not card:
        raise HTTPException(status_code=404, detail="Card not found.")
    card.is_bookmarked = False
    db.commit()
    return {"message": "Card unbookmarked."}


# Static card-list routes BEFORE /cards/{card_id} catch-alls would matter;
# FastAPI matches by path shape so these are safe, but keep them grouped.
@flashcards.get("/cards/bookmarked", response_model=List[schemas.DeckCardOut])
def get_bookmarked_cards(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return (
        db.query(DeckCard)
        .filter_by(user_id=current_user.user_id, is_bookmarked=True)
        .order_by(DeckCard.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


@flashcards.get("/cards/unstudied", response_model=List[schemas.DeckCardOut])
def get_unstudied_cards(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return (
        db.query(DeckCard)
        .filter_by(user_id=current_user.user_id, is_studied=False)
        .order_by(DeckCard.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


@flashcards.get("/cards/hard", response_model=List[schemas.DeckCardOut])
def get_difficult_cards(
    min_reviews: int = Query(default=2, ge=1),
    max_accuracy: float = Query(default=0.6, ge=0.0, le=1.0),
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    cards = (
        db.query(DeckCard)
        .filter(
            DeckCard.user_id == current_user.user_id,
            DeckCard.times_reviewed >= min_reviews,
        )
        .all()
    )
    difficult = []
    for card in cards:
        total = (card.correct_count or 0) + (card.wrong_count or 0)
        accuracy = (card.correct_count or 0) / total if total > 0 else 0
        if accuracy <= max_accuracy:
            difficult.append(card)
    return difficult[skip : skip + limit]


@flashcards.post("/cards/{card_id}/reset")
def reset_card_progress(
    card_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    card = db.query(DeckCard).filter_by(card_id=card_id, user_id=current_user.user_id).first()
    if not card:
        raise HTTPException(status_code=404, detail="Card not found.")

    card.is_studied = False
    card.times_reviewed = 0
    card.correct_count = 0
    card.wrong_count = 0
    card.last_reviewed = None
    db.commit()
    return {"message": "Card progress reset."}


@flashcards.get("/decks/{deck_id}/quiz", response_model=schemas.QuizStartResponse)
def start_quiz(
    deck_id: UUID,
    limit: int = Query(default=5, ge=1, le=50),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    cards = (
        db.query(DeckCard)
        .filter_by(deck_id=deck_id, user_id=current_user.user_id)
        .order_by(func.random())
        .limit(limit)
        .all()
    )
    if not cards:
        raise HTTPException(status_code=404, detail="No flashcards available in this deck.")

    quiz_cards = [schemas.QuizCard(card_id=c.card_id, question=c.question) for c in cards]
    return schemas.QuizStartResponse(deck_id=deck_id, cards=quiz_cards)


@flashcards.post("/quiz/submit", response_model=schemas.QuizResults)
def submit_quiz_server_graded(
    responses: List[schemas.QuizSubmission],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Server-graded: compares user_answer to stored answer (no client trust)."""
    if not responses:
        raise HTTPException(status_code=422, detail="At least one response is required.")

    correct = 0
    wrong = 0
    detailed = []

    for submission in responses:
        card = (
            db.query(DeckCard)
            .filter_by(card_id=submission.card_id, user_id=current_user.user_id)
            .first()
        )
        if not card:
            continue

        is_correct = flashcard_service.grade_quiz_answer(submission.user_answer, card.answer or "")

        card.times_reviewed = (card.times_reviewed or 0) + 1
        card.is_studied = True
        card.last_reviewed = datetime.now(timezone.utc)
        if is_correct:
            card.correct_count = (card.correct_count or 0) + 1
            correct += 1
        else:
            card.wrong_count = (card.wrong_count or 0) + 1
            wrong += 1

        detailed.append(
            {
                "card_id": card.card_id,
                "your_answer": submission.user_answer.strip(),
                "correct_answer": card.answer,
                "correct": is_correct,
            }
        )

    db.commit()
    return schemas.QuizResults(
        total_questions=len(responses), correct=correct, wrong=wrong, detailed_results=detailed,
    )


@flashcards.get("/decks/{deck_id}/get-cards", response_model=List[schemas.DeckCardOut])
def get_deck_cards(
    deck_id: UUID,
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    cards = (
        db.query(DeckCard)
        .filter_by(deck_id=deck_id, user_id=current_user.user_id)
        .order_by(DeckCard.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    if not cards:
        raise HTTPException(status_code=404, detail="No cards found in this deck.")
    return cards


@flashcards.delete("/decks/{deck_id}")
def delete_deck(
    deck_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    deck = db.query(Deck).filter_by(deck_id=deck_id, user_id=current_user.user_id).first()
    if not deck:
        raise HTTPException(status_code=404, detail="Deck not found.")

    # Cards cascade via relationship; single delete suffices.
    db.delete(deck)
    db.commit()
    return {"message": "Deck and all associated cards deleted successfully."}
