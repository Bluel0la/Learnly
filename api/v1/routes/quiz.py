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


def get_user_difficulty_for_topic(db: Session, user_id: UUID, topic: str) -> str:
    questions = (
        db.query(QuizzerQuestion)
        .filter_by(user_id=user_id, topic=topic)
        .filter(QuizzerQuestion.is_correct.isnot(None))
        .all()
    )
    if not questions:
        return "easy"

    total = len(questions)
    correct = sum(1 for q in questions if q.is_correct == 1)
    accuracy = correct / total

    if accuracy >= 0.85:
        return "pro"
    elif accuracy >= 0.6:
        return "medium"
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


@quiz.post(
    "/{session_id}/next-adaptive-batch", response_model=schemas.AdaptiveQuestionBatch
)
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

    # 🧠 Analyze last N graded questions for adaptivity
    graded_qs = (
        db.query(QuizzerQuestion)
        .filter(
            QuizzerQuestion.quiz_id == session_id,
            QuizzerQuestion.user_id == current_user.user_id,
            QuizzerQuestion.is_correct.isnot(None),
        )
        .order_by(
            QuizzerQuestion.question_id.desc()
        )  # Assuming question_id is time-incremented
        .limit(5)
        .all()
    )

    recent_correct = sum(1 for q in graded_qs if q.is_correct == 1)
    recent_total = len(graded_qs)
    score_percent = (
        round((recent_correct / recent_total) * 100, 2) if recent_total else 0.0
    )

    # 🧠 Decide next difficulty adaptively
    if score_percent >= 85:
        next_difficulty = "pro"
    elif score_percent >= 60:
        next_difficulty = "medium"
    else:
        next_difficulty = "easy"

    # 🧮 Generate new batch
    questions = []
    for _ in range(payload.num_questions):
        q = generator_fn(difficulty=next_difficulty)
        q_id = uuid4()

        db_question = QuizzerQuestion(
            question_id=q_id,
            quiz_id=quiz.quiz_id,
            user_id=current_user.user_id,
            topic=quiz.topic,
            difficulty=next_difficulty,
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
                difficulty=next_difficulty,
            )
        )

    quiz.total_questions += payload.num_questions
    db.commit()

    return schemas.AdaptiveQuestionBatch(
        session_id=session_id,
        current_batch=questions,
        remaining=0,
        difficulty_level=next_difficulty,
        previous_score_percent=score_percent,
    )


@quiz.get("/{session_id}/review", response_model=schemas.QuizSessionDetail)
def review_quiz_session(
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
        raise HTTPException(status_code=404, detail="Quiz not found.")

    questions = (
        db.query(QuizzerQuestion)
        .filter_by(quiz_id=session_id, user_id=current_user.user_id)
        .all()
    )

    total_questions = len(questions)
    correct = sum(1 for q in questions if q.is_correct == 1)
    score_percent = (
        round((correct / total_questions) * 100, 2) if total_questions else 0.0
    )

    results = [
        schemas.GradedAnswerResult(
            question_id=q.question_id,
            correct_answer=q.correct_answer,
            selected_answer=q.user_answer or "",
            is_correct=bool(q.is_correct),
            explanation=q.explanation,
        )
        for q in questions
    ]

    return schemas.QuizSessionDetail(
        session_id=session_id,
        topic=quiz.topic,
        total_questions=total_questions,
        score_percent=score_percent,
        results=results,
    )


@quiz.get("/performance", response_model=schemas.PerformanceSummary)
def get_user_performance_summary(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    all_questions = (
        db.query(QuizzerQuestion).filter_by(user_id=current_user.user_id).all()
    )

    if not all_questions:
        return schemas.PerformanceSummary(
            user_id=current_user.user_id, performance_by_topic=[]
        )

    topic_stats = {}

    difficulty_map = {"easy": 1, "medium": 2, "pro": 3}

    for q in all_questions:
        topic = q.topic
        stats = topic_stats.setdefault(
            topic,
            {
                "correct": 0,
                "wrong": 0,
                "total": 0,
                "difficulty_sum": 0,
                "difficulty_count": 0,
            },
        )

        if q.is_correct == 1:
            stats["correct"] += 1
        elif q.is_correct == 0:
            stats["wrong"] += 1

        stats["total"] += 1

        if q.difficulty in difficulty_map:
            stats["difficulty_sum"] += difficulty_map[q.difficulty]
            stats["difficulty_count"] += 1

    performance_by_topic = []
    for topic, data in topic_stats.items():
        accuracy = (
            round((data["correct"] / data["total"]) * 100, 2) if data["total"] else 0.0
        )
        avg_diff = (
            round(data["difficulty_sum"] / data["difficulty_count"], 2)
            if data["difficulty_count"]
            else None
        )
        performance_by_topic.append(
            schemas.TopicPerformance(
                topic=topic,
                total_answered=data["total"],
                correct=data["correct"],
                wrong=data["wrong"],
                accuracy_percent=accuracy,
                average_difficulty=avg_diff,
            )
        )

    return schemas.PerformanceSummary(
        user_id=current_user.user_id, performance_by_topic=performance_by_topic
    )


@quiz.get("/history", response_model=schemas.QuizHistoryResponse)
def get_quiz_history(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    quiz_records = (
        db.query(Quizzer)
        .filter_by(user_id=current_user.user_id)
        .order_by(Quizzer.date_created.desc())
        .all()
    )

    history_entries = []
    for q in quiz_records:
        accuracy = (
            round((q.correct_answers / q.total_questions) * 100, 2)
            if q.total_questions
            else 0.0
        )
        history_entries.append(
            schemas.QuizHistoryEntry(
                session_id=q.quiz_id,
                topic=q.topic,
                date=q.date_created,
                accuracy=accuracy,
                total_questions=q.total_questions,
            )
        )

    return schemas.QuizHistoryResponse(sessions=history_entries)


@quiz.post("/simulated-exam", response_model=schemas.SimulatedExamResponse)
def start_simulated_exam(
    payload: schemas.SimulatedExamRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not payload.topics:
        raise HTTPException(status_code=400, detail="At least one topic is required.")

    if payload.num_questions < 2 * len(payload.topics):
        raise HTTPException(
            status_code=400,
            detail="Number of questions must be at least twice the number of selected topics.",
        )

    session_id = uuid4()
    questions = []

    # Distribute questions evenly among topics
    per_topic = payload.num_questions // len(payload.topics)
    extra = payload.num_questions % len(payload.topics)

    for i, topic in enumerate(payload.topics):
        if topic not in TOPIC_GENERATORS:
            raise HTTPException(status_code=400, detail=f"Invalid topic: {topic}")

        generator_fn = TOPIC_GENERATORS[topic]
        topic_q_count = per_topic + (1 if i < extra else 0)

        user_difficulty = get_user_difficulty_for_topic(db, current_user.user_id, topic)

        for _ in range(topic_q_count):
            q_data = generator_fn(difficulty=user_difficulty)
            q_id = uuid4()

            db_question = QuizzerQuestion(
                question_id=q_id,
                quiz_id=session_id,
                user_id=current_user.user_id,
                topic=topic,
                difficulty=user_difficulty,
                question_text=q_data["question"],
                correct_answer=q_data["correct_answer"],
                choices=json.dumps(q_data["choices"]),
                explanation=q_data.get("explanation"),
            )
            db.add(db_question)

            questions.append(
                schemas.SimulatedExamQuestion(
                    question_id=q_id,
                    topic=topic,
                    question=q_data["question"],
                    difficulty=user_difficulty,
                    choices=q_data["choices"],
                )
            )

    db.add(
        Quizzer(
            quiz_id=session_id,
            user_id=current_user.user_id,
            topic="multi-topic",
            total_questions=len(questions),
            status="in_progress",
        )
    )

    db.commit()

    return schemas.SimulatedExamResponse(
        session_id=session_id, questions=questions, total=len(questions)
    )
