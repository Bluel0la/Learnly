from uuid import uuid4

from sqlalchemy import Column, Integer, String, ForeignKey, UUID
from sqlalchemy.orm import relationship

from api.db.database import Base
from api.v1.models.mixins import TimestampMixin


class Quiz(TimestampMixin, Base):
    __tablename__ = "quiz"

    quiz_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(
        UUID(as_uuid=True), ForeignKey("user.user_id", ondelete="CASCADE"), index=True
    )
    best_score = Column(Integer)

    user = relationship("User", back_populates="quizzes")
    questions = relationship(
        "QuizQuestion", back_populates="quiz", cascade="all, delete"
    )
