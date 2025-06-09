from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from uuid import UUID
from datetime import datetime


# 🎯 1. Available Topic Info
class TopicInfo(BaseModel):
    topic_id: str
    name: str
    description: Optional[str] = None


# 🧠 2. Start Quiz Request
class StartQuizRequest(BaseModel):
    topic: str  # e.g., "addition", "decimal multiplication"
    num_questions: int = Field(default=5, ge=1, le=20)


# 🧠 3. Start Quiz Response
class StartQuizResponse(BaseModel):
    session_id: UUID
    message: str
    total_questions: int
    topic: str
    historical_accuracy: Optional[float] = None  # New field for historical accuracy


# 📩 4. Individual Question
class QuizQuestionOut(BaseModel):
    question_id: UUID
    question: str
    choices: List[str]
    topic: str
    difficulty: Literal["easy", "medium", "pro"]


# 📩 5. Batch of Questions
class QuestionBatchResponse(BaseModel):
    session_id: UUID
    current_batch: List[QuizQuestionOut]
    remaining: int


# ✅ 6. User Submission (one question)
class QuestionAnswerSubmission(BaseModel):
    question_id: UUID
    selected_answer: str


# ✅ 7. Submission Request (batch)
class SubmitAnswersRequest(BaseModel):
    responses: List[QuestionAnswerSubmission]


# ✅ 8. Graded Result (per question)
class GradedAnswerResult(BaseModel):
    question_id: UUID
    correct_answer: str
    selected_answer: str
    is_correct: bool
    explanation: Optional[str]


# ✅ 9. Submit Result Response
class SubmitResultResponse(BaseModel):
    correct: int
    wrong: int
    graded: List[GradedAnswerResult]
    total_attempted: int
    score_percent: float
    next_difficulty: Literal["easy", "medium", "pro"]  # 👈 new field


# 📈 10. Next Adaptive Batch
class AdaptiveQuestionBatch(QuestionBatchResponse):
    difficulty_level: Literal["easy", "medium", "pro"]
    previous_score_percent: float


# 🧾 11. End Session Request
class EndSessionRequest(BaseModel):
    session_id: UUID
    feedback: Optional[str] = None


# 🧾 12. End Session Response
class EndSessionSummary(BaseModel):
    session_id: UUID
    topic: str
    total_questions: int
    correct: int
    wrong: int
    accuracy: float
    ended_at: datetime


# 📚 13. Quiz History Entry
class QuizHistoryEntry(BaseModel):
    session_id: UUID
    topic: str
    date: datetime
    accuracy: float
    total_questions: int


# 📚 14. Quiz History Response
class QuizHistoryResponse(BaseModel):
    sessions: List[QuizHistoryEntry]


# 📋 15. Review Specific Session
class QuizSessionDetail(BaseModel):
    session_id: UUID
    topic: str
    total_questions: int
    score_percent: float
    results: List[GradedAnswerResult]


# 📊 16. Performance Summary
class TopicPerformance(BaseModel):
    topic: str
    total_answered: int
    correct: int
    wrong: int
    accuracy_percent: float
    average_difficulty: Optional[float] = None


class PerformanceSummary(BaseModel):
    user_id: UUID
    performance_by_topic: List[TopicPerformance]


class AdaptiveBatchRequest(BaseModel):
    difficulty: Literal["easy", "medium", "pro"]
    num_questions: int = Field(default=5, ge=1, le=20)
