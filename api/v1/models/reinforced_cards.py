import uuid
from sqlalchemy import Column, ForeignKey, UUID
from sqlalchemy.orm import relationship
from api.db.database import Base  # Import Base from your database setup
from sqlalchemy import String
from api.v1.models.mixins import TimestampMixin


class ReinforcedCard(TimestampMixin, Base):
    __tablename__ = "reinforced_card"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    card_id = Column(
        UUID(as_uuid=True),
        ForeignKey("deck_card.card_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    original_card_id = Column(
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
    reason = Column(String(255), nullable=True)  # E.g., "based on failed attempt", "timeout", etc.

    card = relationship("DeckCard", foreign_keys=[card_id])
    original_card = relationship("DeckCard", foreign_keys=[original_card_id])
    user = relationship("User")

