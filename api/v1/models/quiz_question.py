from sqlalchemy import Column, Integer, String, ForeignKey, TIMESTAMP, UUID, Text
from uuid import uuid4
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from api.db.database import Base


class QuizzerQuestion(Base):
    __tablename__ = "quizzer_question"

    question_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    quiz_id = Column(UUID(as_uuid=True), ForeignKey("quizzer.quiz_id", ondelete="CASCADE"))
    user_id = Column(UUID(as_uuid=True), ForeignKey("user.user_id", ondelete="CASCADE"))

    topic = Column(String(50), nullable=False)
    difficulty = Column(String(20), nullable=False)
    question_text = Column(Text, nullable=False)
    correct_answer = Column(String(100), nullable=False)
    choices = Column(Text, nullable=False)  # JSON-encoded list of str
    explanation = Column(Text, nullable=True)

    user_answer = Column(String(100), nullable=True)
    is_correct = Column(
        Integer, nullable=True
    )  # 1 for correct, 0 for incorrect, NULL for unanswered

    quiz = relationship("Quizzer", back_populates="questions")
    user = relationship("User")
