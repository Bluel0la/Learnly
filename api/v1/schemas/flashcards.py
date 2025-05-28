from pydantic import BaseModel
from typing import List
from uuid import UUID
from datetime import datetime
from typing import Optional


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
    deck_id: UUID
    card_with_answer: str
    is_bookmarked: bool
    is_studied: bool
    times_reviewed: int
    correct_count: int
    wrong_count: int
    last_reviewed: Optional[datetime]

    class Config:
        orm_mode = True


class FlashcardReviewInput(BaseModel):
    is_correct: bool


class NoteChunks(BaseModel):
    chunks: List[str]


class DeckAnalytics(BaseModel):
    deck_id: UUID
    total_cards: int
    studied_cards: int
    total_reviews: int
    correct_answers: int
    wrong_answers: int
    accuracy_percent: float


class QuizCard(BaseModel):
    card_id: UUID
    question: str


class QuizStartResponse(BaseModel):
    deck_id: UUID
    cards: List[QuizCard]


class QuizSubmission(BaseModel):
    card_id: UUID
    user_answer: str


class QuizResults(BaseModel):
    total_questions: int
    correct: int
    wrong: int
    detailed_results: List[dict]
