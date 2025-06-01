import uuid
from sqlalchemy import Column, Text, ForeignKey, TIMESTAMP, UUID, Integer, Boolean, DateTime
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
    question = Column(Text, nullable=True)
    answer = Column(Text, nullable=True)

    date_created = Column(TIMESTAMP, server_default=func.now())
    source_summary = Column(Text, nullable=True)
    source_chunk = Column(Text, nullable=True)
    chunk_index = Column(Integer, nullable=True)  # Optional: helpful for traceability

    # Tracking + usage fields
    is_bookmarked = Column(Boolean, default=False)
    is_studied = Column(Boolean, default=False)
    times_reviewed = Column(Integer, default=0)
    correct_count = Column(Integer, default=0)
    wrong_count = Column(Integer, default=0)
    last_reviewed = Column(DateTime, nullable=True)

    deck = relationship("Deck", back_populates="cards")
    user = relationship("User")
