from pydantic import BaseModel, Field
from typing import List
from uuid import UUID
from datetime import datetime
from typing import Optional


class DeckCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=150)


class DeckOut(BaseModel):
    deck_id: UUID
    title: str
    date_created: datetime
    card_count: Optional[int] = None

    class Config:
        from_attributes = True


class CardCreate(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    answer: str = Field(..., min_length=1, max_length=2000)


class AddCards(BaseModel):
    cards: List[CardCreate] = Field(..., min_length=1, max_length=100)


class DeckCardOut(BaseModel):
    card_id: UUID
    deck_id: UUID
    question: str
    answer: str
    is_bookmarked: bool
    is_studied: bool
    times_reviewed: int
    correct_count: int
    wrong_count: int
    last_reviewed: Optional[datetime]
    source_summary: Optional[str]
    source_chunk: Optional[str]
    chunk_index: Optional[int]

    class Config:
        from_attributes = True


class FlashcardReviewInput(BaseModel):
    is_correct: bool


class CardActionResponse(BaseModel):
    card_id: UUID
    message: str


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
    user_answer: str = Field(..., min_length=1, max_length=2000)

class QuizResultDetail(BaseModel):
    card_id: UUID
    your_answer: str
    correct_answer: str
    correct: bool

class QuizResults(BaseModel):
    total_questions: int
    correct: int
    wrong: int
    detailed_results: List[QuizResultDetail]
