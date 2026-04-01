import uuid

from sqlalchemy import Column, String, ForeignKey, UUID
from sqlalchemy.orm import relationship

from api.db.database import Base
from api.v1.models.mixins import TimestampMixin


class Deck(TimestampMixin, Base):
    __tablename__ = "deck"

    deck_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(
        UUID(as_uuid=True), ForeignKey("user.user_id", ondelete="CASCADE"), index=True
    )
    title = Column(String(255))

    user = relationship("User", back_populates="decks")
    cards = relationship("DeckCard", back_populates="deck", cascade="all, delete")
