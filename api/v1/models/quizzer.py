from sqlalchemy import Column, Integer, String, ForeignKey, TIMESTAMP, UUID
from uuid import uuid4
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from api.db.database import Base


class Quizzer(Base):
    __tablename__ = "quizzer"

    quiz_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("user.user_id", ondelete="CASCADE"))
    date_created = Column(TIMESTAMP, server_default=func.now())
    topic = Column(String(50), nullable=False)  # e.g., "addition"
    total_questions = Column(Integer, default=0)
    correct_answers = Column(Integer, default=0)
    difficulty = Column(
        String(20), nullable=True
    )  # optional: could track initial level
    status = Column(String(20), default="in_progress")  # could be "completed", etc.

    user = relationship("User", back_populates="quizzes")
    questions = relationship(
        "QuizQuestion", back_populates="quizzer", cascade="all, delete"
    )
