from sqlalchemy import Column, Integer, String, ForeignKey, TIMESTAMP, UUID
from uuid import uuid4
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from api.db.database import Base


class Quiz(Base):
    __tablename__ = "quiz"

    quiz_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(
        UUID(as_uuid=True), ForeignKey("user.user_id", ondelete="CASCADE"), index=True
    )
    date_created = Column(TIMESTAMP, server_default=func.now())
    best_score = Column(Integer)

    user = relationship("User", back_populates="quizzes")
    questions = relationship(
        "QuizQuestion", back_populates="quiz", cascade="all, delete"
    )
