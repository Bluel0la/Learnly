import enum
from uuid import uuid4

from sqlalchemy import Column, Integer, String, ForeignKey, UUID, Enum as SAEnum
from sqlalchemy.orm import relationship

from api.db.database import Base
from api.v1.models.mixins import TimestampMixin


class DifficultyEnum(str, enum.Enum):
    easy = "easy"
    medium = "medium"
    pro = "pro"


class QuizStatusEnum(str, enum.Enum):
    in_progress = "in_progress"
    completed = "completed"


class Quizzer(TimestampMixin, Base):
    __tablename__ = "quizzer"

    quiz_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    topic = Column(String(150), nullable=False)
    total_questions = Column(Integer, default=0, nullable=False)
    correct_answers = Column(Integer, default=0, nullable=False)
    difficulty = Column(SAEnum(DifficultyEnum), nullable=False, default=DifficultyEnum.easy)
    status = Column(
        SAEnum(QuizStatusEnum), nullable=False, default=QuizStatusEnum.in_progress
    )

    user = relationship("User", back_populates="quizzers")
    questions = relationship(
        "QuizzerQuestion", back_populates="quizzer", cascade="all, delete-orphan"
    )
