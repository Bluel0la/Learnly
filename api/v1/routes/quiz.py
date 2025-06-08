from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from uuid import uuid4, UUID
from datetime import datetime
import random

from api.db.database import get_db
from api.v1.models.user import User
from api.utils.authentication import get_current_user
from api.v1.schemas import quiz as schemas
from api.utils.math_topics import TOPIC_GENERATORS


quiz = APIRouter(prefix="/quiz/math", tags=["Quiz - Math"])

# --- In-memory session store (for now) ---
quiz_sessions: dict[UUID, dict] = {}

# --- 1. Get Available Math Topics ---
@quiz.get("/topics", response_model=list[schemas.TopicInfo])
def get_available_topics():
    return [
        schemas.TopicInfo(topic_id=key, name=key.replace("_", " ").title())
        for key in TOPIC_GENERATORS
    ]


# --- 2. Start Quiz Session ---
@quiz.post("/start", response_model=schemas.StartQuizResponse)
def start_quiz(
    payload: schemas.StartQuizRequest,
    current_user: User = Depends(get_current_user),
):
    if payload.topic not in TOPIC_GENERATORS:
        raise HTTPException(status_code=404, detail="Invalid topic selected.")

    session_id = uuid4()
    generator_fn = TOPIC_GENERATORS[payload.topic]

    questions = []
    for _ in range(payload.num_questions):
        difficulty = random.choice(["easy", "medium", "pro"])
        q = generator_fn(difficulty=difficulty)
        questions.append(
            {
                "question_id": uuid4(),
                "question": q["question"],
                "choices": q["choices"],
                "correct_answer": q["correct_answer"],
                "topic": q["topic"],
                "difficulty": difficulty,
            }
        )

    quiz_sessions[session_id] = {
        "user_id": current_user.user_id,
        "topic": payload.topic,
        "questions": questions,
        "created_at": datetime.utcnow(),
        "score": [],
    }

    return schemas.StartQuizResponse(
        session_id=session_id,
        topic=payload.topic,
        total_questions=len(questions),
        message=f"Quiz session started for topic: {payload.topic}",
    )


# --- 3. Get Question Batch ---
@quiz.get(
    "/questions/{session_id}", response_model=schemas.QuestionBatchResponse
)
def get_question_batch(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
):
    session = quiz_sessions.get(session_id)
    if not session or session["user_id"] != current_user.user_id:
        raise HTTPException(status_code=403, detail="Invalid or expired session.")

    batch = [
        schemas.QuizQuestionOut(
            question_id=q["question_id"],
            question=q["question"],
            choices=q["choices"],
            topic=q["topic"],
            difficulty=q["difficulty"],
        )
        for q in session["questions"]
    ]

    return schemas.QuestionBatchResponse(
        session_id=session_id, current_batch=batch, remaining=0
    )


# --- 4. Submit Answers ---
@quiz.post("/{session_id}/submit", response_model=schemas.SubmitResultResponse)
def submit_answers(
    session_id: UUID,
    payload: schemas.SubmitAnswersRequest,
    current_user: User = Depends(get_current_user),
):
    session = quiz_sessions.get(session_id)
    if not session or session["user_id"] != current_user.user_id:
        raise HTTPException(status_code=403, detail="Invalid or expired session.")

    question_map = {str(q["question_id"]): q for q in session["questions"]}
    graded_results = []
    correct = 0
    wrong = 0

    for ans in payload.responses:
        q = question_map.get(str(ans.question_id))
        if not q:
            continue

        is_correct = ans.selected_answer.strip() == q["correct_answer"].strip()
        session["score"].append(
            {"question_id": str(ans.question_id), "is_correct": is_correct}
        )

        if is_correct:
            correct += 1
        else:
            wrong += 1

        graded_results.append(
            schemas.GradedAnswerResult(
                question_id=ans.question_id,
                correct_answer=q["correct_answer"],
                selected_answer=ans.selected_answer,
                is_correct=is_correct,
                explanation=None,  # can be extended to use model explanation later
            )
        )

    return schemas.SubmitResultResponse(
        correct=correct,
        wrong=wrong,
        graded=graded_results,
        total_attempted=len(payload.responses),
        score_percent=(
            round((correct / len(payload.responses)) * 100, 2)
            if payload.responses
            else 0.0
        ),
    )
