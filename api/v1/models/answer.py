from sqlalchemy import Column, Integer, Text, ForeignKey, UUID
from sqlalchemy.orm import relationship
from api.db.database import Base  # Import Base from your database setup

class Answer(Base):
    __tablename__ = "answers"

    question_id = Column(
        UUID(as_uuid=True),
        ForeignKey("quiz_question.question_answer_id", ondelete="CASCADE"),
        primary_key=True,
    )
    answer = Column(Text)

    question = relationship("QuizQuestion", back_populates="answers")
