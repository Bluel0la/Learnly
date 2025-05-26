from sqlalchemy import Column, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from api.db.database import Base  # Import Base from your database setup
from sqlalchemy import String

class ReinforcedCard(Base):
    __tablename__ = "reinforced_card"

    card_id = Column(
        UUID(as_uuid=True),
        ForeignKey("deck_card.card_id", ondelete="CASCADE"),
        primary_key=True,
    )
    original_card_id = Column(
        UUID(as_uuid=True), ForeignKey("deck_card.card_id", ondelete="CASCADE")
    )
    reason = Column(String(255))  # E.g., "based on failed attempt", "timeout", etc.

