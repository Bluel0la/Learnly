import uuid
from sqlalchemy import Column, String, ForeignKey, TIMESTAMP, UUID
from sqlalchemy.orm import relationship
from api.db.database import Base  # Import Base from your database setup
from sqlalchemy.sql import func

# The Deck model represents a collection of cards associated with a user.
class Deck(Base):
    __tablename__ = "deck"

    deck_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(
        UUID(as_uuid=True), ForeignKey("user.user_id", ondelete="CASCADE"), index=True
    )
    title = Column(String(255))
    date_created = Column(TIMESTAMP, server_default=func.now())

    user = relationship("User", back_populates="decks")
    cards = relationship("DeckCard", back_populates="deck", cascade="all, delete")
