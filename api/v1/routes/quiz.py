"""
Quiz route handlers — thin wrappers that delegate to quiz_service.

NOTE: static routes (/topics, /performance, /history, /simulated-exam)
are declared BEFORE /{session_id} routes so FastAPI doesn't match
"performance" as a session_id (previous 422 bug).
"""
from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session
from uuid import UUID

from api.db.database import get_db
from api.v1.models.user import User
from api.utils.authentication import get_current_user
from api.v1.schemas import quiz as schemas
from api.v1.services import quiz_service
from api.utils.math_topics import TOPIC_GENERATORS

quiz = APIRouter(prefix="/quiz/math", tags=["Quiz - Math"])


# --- 1. Get Available Math Topics ---
@quiz.get("/topics", response_model=list[schemas.TopicInfo])
def get_available_topics():
    return [
        schemas.TopicInfo(topic_id=key, name=key.replace("_", " ").title())
        for key in TOPIC_GENERATORS
    ]


# --- 2. Performance Summary (before /{session_id} routes!) ---
@quiz.get("/performance", response_model=schemas.PerformanceSummary)
def get_user_performance_summary(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = quiz_service.get_performance_summary(db, current_user.user_id)
    return schemas.PerformanceSummary(
        user_id=result["user_id"],
        performance_by_topic=[
            schemas.TopicPerformance(**t) for t in result["performance_by_topic"]
        ],
    )


# --- 3. Quiz History (paginated, before /{session_id} routes!) ---
@quiz.get("/history", response_model=schemas.QuizHistoryResponse)
def get_quiz_history(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = quiz_service.get_quiz_history(db, current_user.user_id, skip=skip, limit=limit)
    return schemas.QuizHistoryResponse(
        sessions=[schemas.QuizHistoryEntry(**s) for s in result["sessions"]]
    )


# --- 4. Simulated Exam (before /{session_id} routes!) ---
@quiz.post("/simulated-exam", response_model=schemas.SimulatedExamResponse, status_code=status.HTTP_201_CREATED)
def start_simulated_exam(
    payload: schemas.SimulatedExamRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = quiz_service.start_simulated_exam(
        db, current_user.user_id, payload.topics, payload.num_questions,
    )
    return schemas.SimulatedExamResponse(
        session_id=result["session_id"],
        questions=[schemas.SimulatedExamQuestion(**q) for q in result["questions"]],
        total=result["total"],
    )


# --- 5. Start Quiz Session ---
@quiz.post("/start", response_model=schemas.StartQuizResponse, status_code=status.HTTP_201_CREATED)
def start_quiz(
    payload: schemas.StartQuizRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = quiz_service.start_quiz_session(
        db, user_id=current_user.user_id,
        topic=payload.topic, num_questions=payload.num_questions,
    )
    return schemas.StartQuizResponse(**result)


# --- 6. Get Question Batch ---
@quiz.get("/questions/{session_id}", response_model=schemas.QuestionBatchResponse)
def get_question_batch(
    session_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = quiz_service.get_questions_for_session(db, current_user.user_id, session_id)
    return schemas.QuestionBatchResponse(
        session_id=result["session_id"],
        current_batch=[schemas.QuizQuestionOut(**q) for q in result["current_batch"]],
        remaining=result["remaining"],
    )


# --- 7. Submit Answers ---
@quiz.post("/{session_id}/submit", response_model=schemas.SubmitResultResponse)
def submit_answers(
    session_id: UUID,
    payload: schemas.SubmitAnswersRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = quiz_service.grade_answers(
        db, current_user.user_id, session_id, payload.responses,
    )
    return schemas.SubmitResultResponse(
        correct=result["correct"],
        wrong=result["wrong"],
        graded=[schemas.GradedAnswerResult(**g) for g in result["graded"]],
        total_attempted=result["total_attempted"],
        score_percent=result["score_percent"],
        next_difficulty=result["next_difficulty"],
    )


# --- 8. Next Adaptive Batch ---
@quiz.post(
    "/{session_id}/next-adaptive-batch", response_model=schemas.AdaptiveQuestionBatch
)
def get_next_adaptive_batch(
    session_id: UUID,
    payload: schemas.AdaptiveBatchRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = quiz_service.generate_adaptive_batch(
        db, current_user.user_id, session_id, payload.num_questions,
    )
    return schemas.AdaptiveQuestionBatch(
        session_id=result["session_id"],
        current_batch=[schemas.QuizQuestionOut(**q) for q in result["current_batch"]],
        remaining=result["remaining"],
        difficulty_level=result["difficulty_level"],
        previous_score_percent=result["previous_score_percent"],
    )


# --- 9. Review Session ---
@quiz.get("/{session_id}/review", response_model=schemas.QuizSessionDetail)
def review_quiz_session(
    session_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = quiz_service.review_session(db, current_user.user_id, session_id)
    return schemas.QuizSessionDetail(
        session_id=result["session_id"],
        topic=result["topic"],
        total_questions=result["total_questions"],
        score_percent=result["score_percent"],
        results=[schemas.GradedAnswerResult(**r) for r in result["results"]],
    )


# --- 10. End Session (implements missing EndSession schemas) ---
@quiz.post("/{session_id}/end", response_model=schemas.EndSessionSummary)
def end_quiz_session(
    session_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = quiz_service.end_session(db, current_user.user_id, session_id)
    return schemas.EndSessionSummary(**result)
