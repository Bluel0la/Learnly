from pydantic import BaseModel
from typing import List
from uuid import UUID
from datetime import datetime


class DeckCreate(BaseModel):
    title: str


class DeckOut(BaseModel):
    deck_id: UUID
    title: str
    date_created: datetime

    class Config:
        orm_mode = True


class CardCreate(BaseModel):
    card_with_answer: str


class AddCards(BaseModel):
    cards: List[CardCreate]


class DeckCardOut(BaseModel):
    card_id: UUID
    card_with_answer: str
    date_created: datetime

    class Config:
        orm_mode = True

class NoteChunks(BaseModel):
    chunks: List[str]
