import uuid
from sqlalchemy import Column, ForeignKey, TIMESTAMP, Boolean, Integer, UUID
from sqlalchemy.orm import relationship
from api.db.database import Base  # Import Base from your database setup
from sqlalchemy.sql import func
from api.v1.models.mixins import TimestampMixin


class FlashcardAttempt(TimestampMixin, Base):
    __tablename__ = "flashcard_attempt"

    attempt_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    card_id = Column(
        UUID(as_uuid=True),
        ForeignKey("deck_card.card_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    attempt_time = Column(TIMESTAMP, server_default=func.now(), nullable=False)
    correct = Column(Boolean, default=False, nullable=False)
    time_taken_seconds = Column(Integer, nullable=True)  # Optional: store how long it took
    attempt_number = Column(Integer, nullable=False, default=1)  # Useful for tracking retries
    hint_used = Column(Boolean, default=False, nullable=False)

    card = relationship("DeckCard")
    user = relationship("User")
