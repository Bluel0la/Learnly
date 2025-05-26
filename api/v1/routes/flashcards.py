from fastapi import APIRouter, Depends, HTTPException, status
from api.utils.authentication import get_current_user
from api.v1.models import deck_card as card_models
from api.v1.schemas import flashcards as schemas
from api.v1.models import deck as models
from api.v1.models.user import User
from sqlalchemy.orm import Session
from api.db.database import get_db
from typing import List
from uuid import UUID
import os, httpx, asyncio


flashcards = APIRouter(prefix="/flashcard", tags=["Flashcards"])
model_endpoint = os.getenv("MODEL_ENDPOINT")


# ✅ Create a deck
@flashcards.post("/decks/", response_model=schemas.DeckOut)
def create_deck(
    deck: schemas.DeckCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    db_deck = models.Deck(title=deck.title, user_id=current_user.user_id)
    db.add(db_deck)
    db.commit()
    db.refresh(db_deck)
    return db_deck


# ✅ Get all decks for the current user
@flashcards.get("/decks/", response_model=List[schemas.DeckOut])
def get_user_decks(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return db.query(models.Deck).filter_by(user_id=current_user.user_id).all()


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

    cards = [
        card_models.DeckCard(
            deck_id=deck_id,
            user_id=current_user.user_id,
            card_with_answer=card.card_with_answer,
        )
        for card in payload.cards
    ]
    db.add_all(cards)
    db.commit()
    return {"message": f"{len(cards)} cards added successfully."}


# ✅ Get all cards in a specific deck
@flashcards.get("/decks/{deck_id}/cards/", response_model=List[schemas.DeckCardOut])
def get_cards_in_deck(
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
    return db.query(card_models.DeckCard).filter_by(deck_id=deck_id).all()


@flashcards.post("/decks/{deck_id}/generate-flashcards/")
async def generate_flashcards_from_notes(
    deck_id: UUID,
    notes: schemas.NoteChunks,  # accepts List[str]
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    model_util_endpoint = os.getenv("MODEL_UTILITY")
    model_chat_endpoint = os.getenv("MODEL_ENDPOINT")
    flashcard_url = f"{model_chat_endpoint}/flashcard"

    # 1. Get the deck and verify ownership
    db_deck = (
        db.query(models.Deck)
        .filter_by(deck_id=deck_id, user_id=current_user.user_id)
        .first()
    )
    if not db_deck:
        raise HTTPException(status_code=404, detail="Deck not found.")

    # 2. Summarize notes into key points
    summarization_url = f"{model_util_endpoint}/flashcard-summarization"
    try:
        async with httpx.AsyncClient() as client:
            summary_resp = await client.post(
                summarization_url, json={"chunks": notes.chunks}
            )
        summary_resp.raise_for_status()
        key_points = summary_resp.json().get("points", [])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")

    if not key_points:
        raise HTTPException(status_code=400, detail="No key points returned from summarization model.")

    # 3. Generate flashcards (Q&A) for each key point
    qa_pairs = []
    try:
        async with httpx.AsyncClient() as client:
            for point in key_points:
                flashcard_resp = await client.post(flashcard_url, json={"prompt": point})
                flashcard_resp.raise_for_status()
                card = flashcard_resp.json()
                qa_pairs.append(card)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Flashcard generation failed: {str(e)}")

    if not qa_pairs:
        raise HTTPException(status_code=400, detail="No Q&A pairs returned.")

    # 4. Save flashcards
    flashcards = [
        card_models.DeckCard(
            deck_id=deck_id,
            user_id=current_user.user_id,
            card_with_answer=f"Q: {qa['question']}\nA: {qa['answer']}",
        )
        for qa in qa_pairs
    ]
    db.add_all(flashcards)
    db.commit()

    return {"message": f"{len(flashcards)} flashcards generated and added to deck."}
