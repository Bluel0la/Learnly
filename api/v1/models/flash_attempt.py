import uuid
from sqlalchemy import Column, ForeignKey, TIMESTAMP, Boolean, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from api.db.database import Base  # Import Base from your database setup
from sqlalchemy.sql import func


class FlashcardAttempt(Base):
    __tablename__ = "flashcard_attempt"

    attempt_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    card_id = Column(
        UUID(as_uuid=True),
        ForeignKey("deck_card.card_id", ondelete="CASCADE"),
        index=True,
    )
    user_id = Column(
        UUID(as_uuid=True), ForeignKey("user.user_id", ondelete="CASCADE"), index=True
    )
    attempt_time = Column(TIMESTAMP, server_default=func.now())
    correct = Column(Boolean, default=False)
    time_taken_seconds = Column(Integer)  # Optional: store how long it took
    attempt_number = Column(Integer)  # Useful for tracking retries
    hint_used = Column(Boolean, default=False)

    card = relationship("DeckCard")
    user = relationship("User")
