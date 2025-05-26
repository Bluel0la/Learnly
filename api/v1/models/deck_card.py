import uuid
from sqlalchemy import Column, Text, ForeignKey, TIMESTAMP, UUID, Integer
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from api.db.database import Base  # Import Base from your database setup

class DeckCard(Base):
    __tablename__ = "deck_card"

    card_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    deck_id = Column(
        UUID(as_uuid=True), ForeignKey("deck.deck_id", ondelete="CASCADE"), index=True
    )
    user_id = Column(
        UUID(as_uuid=True), ForeignKey("user.user_id", ondelete="CASCADE"), index=True
    )
    card_with_answer = Column(Text, nullable=False)
    date_created = Column(TIMESTAMP, server_default=func.now())
    source_summary = Column(Text, nullable=True)
    source_chunk = Column(Text, nullable=True)
    chunk_index = Column(Integer, nullable=True)  # Optional: helpful for traceability


    deck = relationship("Deck", back_populates="cards")
    user = relationship("User")
