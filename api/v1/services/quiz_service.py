"""
Business logic for the Math Quiz module.

All database queries, question generation, grading, and adaptive difficulty
logic lives here. Route handlers in routes/quiz.py delegate to these functions.
"""
import json
import random
from uuid import UUID, uuid4
from typing import Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from api.v1.models.quizzer import Quizzer
from api.v1.models.quiz_question import QuizzerQuestion
from api.utils.math_topics import TOPIC_GENERATORS
from api.core.exceptions import NotFoundException, BadRequestException
from api.core.config import settings


# ---------------------------------------------------------------------------
# Difficulty helpers
# ---------------------------------------------------------------------------

def get_topic_accuracy(db: Session, user_id: UUID, topic: str) -> float:
    """Return the user's historical accuracy (0-100) for a given topic."""
    total = (
        db.query(func.count())
        .select_from(QuizzerQuestion)
        .filter_by(user_id=user_id, topic=topic)
        .scalar()
    )
    correct = (
        db.query(func.count())
        .select_from(QuizzerQuestion)
        .filter_by(user_id=user_id, topic=topic, is_correct=True)
        .scalar()
    )
    if total == 0:
        return 50.0
    return round((correct / total) * 100, 2)


def determine_difficulty_from_accuracy(accuracy: float) -> str:
    if accuracy >= settings.QUIZ_PRO_THRESHOLD:
        return "pro"
    elif accuracy >= settings.QUIZ_MEDIUM_THRESHOLD:
        return "medium"
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
    correct = sum(1 for q in questions if q.is_correct is True)
    accuracy = (correct / total) * 100

    if accuracy >= settings.QUIZ_PRO_THRESHOLD:
        return "pro"
    elif accuracy >= settings.QUIZ_MEDIUM_THRESHOLD:
        return "medium"
    return "easy"


def _randomized_difficulty(base: str) -> str:
    if base == "medium":
        return random.choices(["easy", "medium", "pro"], weights=settings.WEIGHTS_FROM_MEDIUM)[0]
    elif base == "pro":
        return random.choices(["medium", "pro"], weights=settings.WEIGHTS_FROM_PRO)[0]
    return random.choices(["easy", "medium"], weights=settings.WEIGHTS_FROM_EASY)[0]


# ---------------------------------------------------------------------------
# Quiz lifecycle
# ---------------------------------------------------------------------------

def start_quiz_session(
    db: Session, user_id: UUID, topic: str, num_questions: int
) -> dict:
    """Create a new quiz session and generate questions. Returns a summary dict."""
    if topic not in TOPIC_GENERATORS:
        raise NotFoundException("Invalid topic selected.")

    generator_fn = TOPIC_GENERATORS[topic]
    quiz_id = uuid4()

    historical_accuracy = get_topic_accuracy(db, user_id=user_id, topic=topic)
    base_difficulty = determine_difficulty_from_accuracy(historical_accuracy)

    quiz_obj = Quizzer(
        quiz_id=quiz_id,
        user_id=user_id,
        topic=topic,
        total_questions=num_questions,
        difficulty=base_difficulty,
    )
    db.add(quiz_obj)
    db.commit()

    question_objs = []
    for _ in range(num_questions):
        difficulty = _randomized_difficulty(base_difficulty)
        q = generator_fn(difficulty=difficulty)
        question_objs.append(
            QuizzerQuestion(
                quiz_id=quiz_id,
                user_id=user_id,
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

    return {
        "session_id": quiz_id,
        "topic": topic,
        "total_questions": len(question_objs),
        "message": f"Quiz session started for topic: {topic}",
        "historical_accuracy": historical_accuracy,
    }


def get_questions_for_session(
    db: Session, user_id: UUID, session_id: UUID
) -> dict:
    """Return all questions for a quiz session."""
    quiz = (
        db.query(Quizzer)
        .filter_by(quiz_id=session_id, user_id=user_id)
        .first()
    )
    if not quiz:
        raise NotFoundException("Quiz not found or unauthorized.")

    questions = (
        db.query(QuizzerQuestion)
        .filter_by(quiz_id=session_id, user_id=user_id)
        .all()
    )

    batch = [
        {
            "question_id": q.question_id,
            "question": q.question_text,
            "choices": json.loads(q.choices),
            "topic": q.topic,
            "difficulty": q.difficulty,
        }
        for q in questions
    ]

    return {"session_id": session_id, "current_batch": batch, "remaining": 0}


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

def grade_answers(
    db: Session, user_id: UUID, session_id: UUID, responses: list
) -> dict:
    """Grade submitted answers, update DB, and return results."""
    quiz = (
        db.query(Quizzer)
        .filter_by(quiz_id=session_id, user_id=user_id)
        .first()
    )
    if not quiz:
        raise NotFoundException("Quiz not found or unauthorized.")

    question_map = {
        str(q.question_id): q
        for q in db.query(QuizzerQuestion)
        .filter_by(quiz_id=session_id, user_id=user_id)
        .all()
    }

    graded_results = []
    correct = 0
    wrong = 0

    for ans in responses:
        q = question_map.get(str(ans.question_id))
        if not q:
            continue

        is_correct = ans.selected_answer.strip() == q.correct_answer.strip()
        q.user_answer = ans.selected_answer
        q.is_correct = is_correct

        if is_correct:
            correct += 1
        else:
            wrong += 1

        graded_results.append(
            {
                "question_id": q.question_id,
                "correct_answer": q.correct_answer,
                "selected_answer": ans.selected_answer,
                "is_correct": is_correct,
                "explanation": q.explanation if not is_correct else None,
            }
        )

    quiz.correct_answers = correct
    quiz.status = "completed"
    db.commit()

    total = len(graded_results)
    score_percent = round((correct / total) * 100, 2) if total else 0.0

    if score_percent >= settings.QUIZ_PRO_THRESHOLD:
        next_difficulty = "pro"
    elif score_percent >= settings.QUIZ_MEDIUM_THRESHOLD:
        next_difficulty = "medium"
    else:
        next_difficulty = "easy"

    return {
        "correct": correct,
        "wrong": wrong,
        "graded": graded_results,
        "total_attempted": total,
        "score_percent": score_percent,
        "next_difficulty": next_difficulty,
    }


# ---------------------------------------------------------------------------
# Adaptive batch
# ---------------------------------------------------------------------------

def generate_adaptive_batch(
    db: Session, user_id: UUID, session_id: UUID, num_questions: int
) -> dict:
    quiz = (
        db.query(Quizzer)
        .filter_by(quiz_id=session_id, user_id=user_id)
        .first()
    )
    if not quiz:
        raise NotFoundException("Quiz not found.")

    generator_fn = TOPIC_GENERATORS.get(quiz.topic)
    if not generator_fn:
        raise BadRequestException("Unsupported topic.")

    graded_qs = (
        db.query(QuizzerQuestion)
        .filter(
            QuizzerQuestion.quiz_id == session_id,
            QuizzerQuestion.user_id == user_id,
            QuizzerQuestion.is_correct.isnot(None),
        )
        .order_by(QuizzerQuestion.question_id.desc())
        .limit(5)
        .all()
    )

    recent_correct = sum(1 for q in graded_qs if q.is_correct is True)
    recent_total = len(graded_qs)
    score_percent = (
        round((recent_correct / recent_total) * 100, 2) if recent_total else 0.0
    )

    if score_percent >= settings.QUIZ_PRO_THRESHOLD:
        next_difficulty = "pro"
    elif score_percent >= settings.QUIZ_MEDIUM_THRESHOLD:
        next_difficulty = "medium"
    else:
        next_difficulty = "easy"

    questions = []
    for _ in range(num_questions):
        q = generator_fn(difficulty=next_difficulty)
        q_id = uuid4()

        db_question = QuizzerQuestion(
            question_id=q_id,
            quiz_id=quiz.quiz_id,
            user_id=user_id,
            topic=quiz.topic,
            difficulty=next_difficulty,
            question_text=q["question"],
            correct_answer=q["correct_answer"],
            choices=json.dumps(q["choices"]),
            explanation=q.get("explanation"),
        )
        db.add(db_question)

        questions.append(
            {
                "question_id": q_id,
                "question": q["question"],
                "choices": q["choices"],
                "topic": quiz.topic,
                "difficulty": next_difficulty,
            }
        )

    quiz.total_questions += num_questions
    db.commit()

    return {
        "session_id": session_id,
        "current_batch": questions,
        "remaining": 0,
        "difficulty_level": next_difficulty,
        "previous_score_percent": score_percent,
    }


# ---------------------------------------------------------------------------
# Review & history
# ---------------------------------------------------------------------------

def review_session(db: Session, user_id: UUID, session_id: UUID) -> dict:
    quiz = (
        db.query(Quizzer)
        .filter_by(quiz_id=session_id, user_id=user_id)
        .first()
    )
    if not quiz:
        raise NotFoundException("Quiz not found.")

    questions = (
        db.query(QuizzerQuestion)
        .filter_by(quiz_id=session_id, user_id=user_id)
        .all()
    )

    total_questions = len(questions)
    correct = sum(1 for q in questions if q.is_correct is True)
    score_percent = (
        round((correct / total_questions) * 100, 2) if total_questions else 0.0
    )

    results = [
        {
            "question_id": q.question_id,
            "correct_answer": q.correct_answer,
            "selected_answer": q.user_answer or "",
            "is_correct": bool(q.is_correct),
            "explanation": q.explanation,
        }
        for q in questions
    ]

    return {
        "session_id": session_id,
        "topic": quiz.topic,
        "total_questions": total_questions,
        "score_percent": score_percent,
        "results": results,
    }


def get_performance_summary(db: Session, user_id: UUID) -> dict:
    all_questions = (
        db.query(QuizzerQuestion).filter_by(user_id=user_id).all()
    )

    if not all_questions:
        return {"user_id": user_id, "performance_by_topic": []}

    topic_stats: dict = {}
    difficulty_map = {"easy": 1, "medium": 2, "pro": 3}

    for q in all_questions:
        topic = q.topic
        stats = topic_stats.setdefault(
            topic,
            {"correct": 0, "wrong": 0, "total": 0, "difficulty_sum": 0, "difficulty_count": 0},
        )

        if q.is_correct is True:
            stats["correct"] += 1
        elif q.is_correct is False:
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
            {
                "topic": topic,
                "total_answered": data["total"],
                "correct": data["correct"],
                "wrong": data["wrong"],
                "accuracy_percent": accuracy,
                "average_difficulty": avg_diff,
            }
        )

    return {"user_id": user_id, "performance_by_topic": performance_by_topic}


def get_quiz_history(
    db: Session, user_id: UUID, skip: int = 0, limit: int = 20
) -> dict:
    quiz_records = (
        db.query(Quizzer)
        .filter_by(user_id=user_id)
        .order_by(Quizzer.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )

    sessions = []
    for q in quiz_records:
        accuracy = (
            round((q.correct_answers / q.total_questions) * 100, 2)
            if q.total_questions
            else 0.0
        )
        sessions.append(
            {
                "session_id": q.quiz_id,
                "topic": q.topic,
                "date": q.created_at,
                "accuracy": accuracy,
                "total_questions": q.total_questions,
            }
        )

    return {"sessions": sessions}


# ---------------------------------------------------------------------------
# Simulated exam
# ---------------------------------------------------------------------------

def start_simulated_exam(
    db: Session, user_id: UUID, topics: list[str], num_questions: int
) -> dict:
    if not topics:
        raise BadRequestException("At least one topic is required.")

    if num_questions < 2 * len(topics):
        raise BadRequestException(
            "Number of questions must be at least twice the number of selected topics."
        )

    session_id = uuid4()
    questions = []

    per_topic = num_questions // len(topics)
    extra = num_questions % len(topics)

    for i, topic in enumerate(topics):
        if topic not in TOPIC_GENERATORS:
            raise BadRequestException(f"Invalid topic: {topic}")

        generator_fn = TOPIC_GENERATORS[topic]
        topic_q_count = per_topic + (1 if i < extra else 0)
        user_difficulty = get_user_difficulty_for_topic(db, user_id, topic)

        for _ in range(topic_q_count):
            q_data = generator_fn(difficulty=user_difficulty)
            q_id = uuid4()

            db_question = QuizzerQuestion(
                question_id=q_id,
                quiz_id=session_id,
                user_id=user_id,
                topic=topic,
                difficulty=user_difficulty,
                question_text=q_data["question"],
                correct_answer=q_data["correct_answer"],
                choices=json.dumps(q_data["choices"]),
                explanation=q_data.get("explanation"),
            )
            db.add(db_question)

            questions.append(
                {
                    "question_id": q_id,
                    "topic": topic,
                    "question": q_data["question"],
                    "difficulty": user_difficulty,
                    "choices": q_data["choices"],
                }
            )

    db.add(
        Quizzer(
            quiz_id=session_id,
            user_id=user_id,
            topic="multi-topic",
            total_questions=len(questions),
            status="in_progress",
        )
    )

    db.commit()

    return {"session_id": session_id, "questions": questions, "total": len(questions)}
