from uuid import uuid4

from sqlalchemy import Column, Boolean, String, ForeignKey, UUID, Text
from sqlalchemy.orm import relationship

from api.db.database import Base
from api.v1.models.mixins import TimestampMixin


class QuizzerQuestion(TimestampMixin, Base):
    __tablename__ = "quizzer_question"

    question_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    quiz_id = Column(
        UUID(as_uuid=True),
        ForeignKey("quizzer.quiz_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    topic = Column(String(150), nullable=False)
    difficulty = Column(String(20), nullable=False)
    question_text = Column(Text, nullable=False)
    correct_answer = Column(String(100), nullable=False)
    choices = Column(Text, nullable=False)  # JSON-encoded list of str
    explanation = Column(Text, nullable=True)

    user_answer = Column(String(100), nullable=True)
    is_correct = Column(Boolean, nullable=True)  # True/False/None (unanswered)

    quizzer = relationship("Quizzer", back_populates="questions")
    user = relationship("User")
