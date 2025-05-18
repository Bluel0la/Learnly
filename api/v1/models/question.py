from sqlalchemy import Column, ForeignKey, Text, UUID
from uuid import uuid4
from sqlalchemy.orm import relationship
from api.db.database import Base  # Import Base from your database setup


class QuizQuestion(Base):
    __tablename__ = "quiz_question"

    question_answer_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    quiz_id = Column(
        UUID(as_uuid=True), ForeignKey("quiz.quiz_id", ondelete="CASCADE"), index=True
    )
    user_id = Column(
        UUID(as_uuid=True), ForeignKey("user.user_id", ondelete="CASCADE"), index=True
    )
    question_answer = Column(Text, nullable=False)

    quiz = relationship("Quiz", back_populates="questions")
    answers = relationship(
        "Answer", back_populates="question", cascade="all, delete-orphan"
    )

    user = relationship("User")
