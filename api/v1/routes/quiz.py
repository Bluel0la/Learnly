from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from uuid import uuid4, UUID
from sqlalchemy import func
import random, json

from api.db.database import get_db
from api.v1.models.user import User
from api.utils.authentication import get_current_user
from api.v1.schemas import quiz as schemas
from api.utils.math_topics import TOPIC_GENERATORS
from api.v1.models.quizzer import Quizzer
from api.v1.models.quiz_question import QuizzerQuestion

quiz = APIRouter(prefix="/quiz/math", tags=["Quiz - Math"])

# --- In-memory session store (for now) ---
quiz_sessions: dict[UUID, dict] = {}


def get_topic_accuracy(db: Session, user_id: UUID, topic: str) -> float:
    total = (
        db.query(func.count())
        .select_from(QuizzerQuestion)
        .filter_by(user_id=user_id, topic=topic)
        .scalar()
    )
    correct = (
        db.query(func.count())
        .select_from(QuizzerQuestion)
        .filter_by(user_id=user_id, topic=topic, is_correct=1)
        .scalar()
    )

    if total == 0:
        return 50.0  # default mid-point
    return round((correct / total) * 100, 2)


# Determine difficulty from accuracy score
def determine_difficulty_from_accuracy(accuracy: float) -> str:
    if accuracy >= 80:
        return "pro"
    elif accuracy >= 50:
        return "medium"
    else:
        return "easy"


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
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if payload.topic not in TOPIC_GENERATORS:
        raise HTTPException(status_code=404, detail="Invalid topic selected.")

    generator_fn = TOPIC_GENERATORS[payload.topic]
    quiz_id = uuid4()

    # ✨ Get historical accuracy
    historical_accuracy = get_topic_accuracy(
        db, user_id=current_user.user_id, topic=payload.topic
    )
    base_difficulty = determine_difficulty_from_accuracy(historical_accuracy)

    # 🧠 Optionally add a bit of randomness for robustness
    def randomized_difficulty(base: str) -> str:
        if base == "medium":
            return random.choices(["easy", "medium", "pro"], weights=[0.2, 0.6, 0.2])[0]
        elif base == "pro":
            return random.choices(["medium", "pro"], weights=[0.3, 0.7])[0]
        else:
            return random.choices(["easy", "medium"], weights=[0.7, 0.3])[0]

    # Create the quiz session entry
    quiz_obj = Quizzer(
        quiz_id=quiz_id,
        user_id=current_user.user_id,
        topic=payload.topic,
        total_questions=payload.num_questions,
        difficulty=base_difficulty,
    )
    db.add(quiz_obj)
    db.commit()

    # Create question objects
    question_objs = []
    for _ in range(payload.num_questions):
        difficulty = randomized_difficulty(base_difficulty)
        q = generator_fn(difficulty=difficulty)
        question_objs.append(
            QuizzerQuestion(
                quiz_id=quiz_id,
                user_id=current_user.user_id,
                topic=q["topic"],
                difficulty=difficulty,
                question_text=q["question"],
                correct_answer=q["correct_answer"],
                choices=json.dumps(q["choices"]),
                explanation=q.get("explanation"),
            )
        )

    db.bulk_save_objects(question_objs)
    db.commit()

    return schemas.StartQuizResponse(
        session_id=quiz_id,
        topic=payload.topic,
        total_questions=len(question_objs),
        message=f"Quiz session started for topic: {payload.topic}",
        historical_accuracy=historical_accuracy,  # only if you added this to schema
    )


# --- 3. Get Question Batch ---
@quiz.get("/questions/{session_id}", response_model=schemas.QuestionBatchResponse)
def get_question_batch(
    session_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    quiz = (
        db.query(Quizzer)
        .filter_by(quiz_id=session_id, user_id=current_user.user_id)
        .first()
    )
    if not quiz:
        raise HTTPException(status_code=404, detail="Quiz not found or unauthorized.")

    questions = (
        db.query(QuizzerQuestion)
        .filter_by(quiz_id=session_id, user_id=current_user.user_id)
        .all()
    )

    batch = [
        schemas.QuizQuestionOut(
            question_id=q.question_id,
            question=q.question_text,
            choices=json.loads(q.choices),
            topic=q.topic,
            difficulty=q.difficulty,
        )
        for q in questions
    ]

    return schemas.QuestionBatchResponse(
        session_id=session_id, current_batch=batch, remaining=0  # placeholder for now
    )


# --- 4. Submit Answers ---
@quiz.post("/{session_id}/submit", response_model=schemas.SubmitResultResponse)
def submit_answers(
    session_id: UUID,
    payload: schemas.SubmitAnswersRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    include_explanations: bool = False,
):
    quiz = (
        db.query(Quizzer)
        .filter_by(quiz_id=session_id, user_id=current_user.user_id)
        .first()
    )
    if not quiz:
        raise HTTPException(status_code=404, detail="Quiz not found or unauthorized.")

    question_map = {
        str(q.question_id): q
        for q in db.query(QuizzerQuestion)
        .filter_by(quiz_id=session_id, user_id=current_user.user_id)
        .all()
    }

    graded_results = []
    correct = 0
    wrong = 0

    for ans in payload.responses:
        q = question_map.get(str(ans.question_id))
        if not q:
            continue

        is_correct = ans.selected_answer.strip() == q.correct_answer.strip()
        q.user_answer = ans.selected_answer
        q.is_correct = 1 if is_correct else 0

        if is_correct:
            correct += 1
        else:
            wrong += 1

        graded_results.append(
            schemas.GradedAnswerResult(
                question_id=q.question_id,
                correct_answer=q.correct_answer,
                selected_answer=ans.selected_answer,
                is_correct=is_correct,
                explanation=(
                    q.explanation if include_explanations or not is_correct else None
                ),
            )
        )

    # Update quiz stats
    quiz.correct_answers = correct
    quiz.status = "completed"

    db.commit()

    # 🔁 Adaptive difficulty logic
    total = len(graded_results)
    score_percent = round((correct / total) * 100, 2) if total else 0.0

    if score_percent >= 85:
        next_difficulty = "pro"
    elif score_percent >= 60:
        next_difficulty = "medium"
    else:
        next_difficulty = "easy"

    return schemas.SubmitResultResponse(
        correct=correct,
        wrong=wrong,
        graded=graded_results,
        total_attempted=total,
        score_percent=score_percent,
        next_difficulty=next_difficulty,
    )


@quiz.post("/{session_id}/next-batch", response_model=schemas.AdaptiveQuestionBatch)
def get_next_adaptive_batch(
    session_id: UUID,
    payload: schemas.AdaptiveBatchRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    quiz = (
        db.query(Quizzer)
        .filter_by(quiz_id=session_id, user_id=current_user.user_id)
        .first()
    )
    if not quiz:
        raise HTTPException(status_code=404, detail="Quiz not found.")

    generator_fn = TOPIC_GENERATORS.get(quiz.topic)
    if not generator_fn:
        raise HTTPException(status_code=400, detail="Unsupported topic.")

    questions = []
    for _ in range(payload.num_questions):
        q = generator_fn(difficulty=payload.difficulty)
        q_id = uuid4()

        # Save each generated question to DB
        db_question = QuizzerQuestion(
            question_id=q_id,
            quiz_id=quiz.quiz_id,
            user_id=current_user.user_id,
            topic=quiz.topic,
            difficulty=payload.difficulty,
            question_text=q["question"],
            correct_answer=q["correct_answer"],
            choices=json.dumps(q["choices"]),
            explanation=q.get("explanation"),
        )
        db.add(db_question)

        questions.append(
            schemas.QuizQuestionOut(
                question_id=q_id,
                question=q["question"],
                choices=q["choices"],
                topic=quiz.topic,
                difficulty=payload.difficulty,
            )
        )

    quiz.total_questions += payload.num_questions
    db.commit()

    return schemas.AdaptiveQuestionBatch(
        session_id=session_id,
        current_batch=questions,
        remaining=0,
        difficulty_level=payload.difficulty,
        previous_score_percent=(
            round((quiz.correct_answers / quiz.total_questions) * 100, 2)
            if quiz.total_questions
            else 0.0
        ),
    )
