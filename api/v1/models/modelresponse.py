from sqlalchemy import (
    Column,Text, ForeignKey, TIMESTAMP, UUID
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from api.db.database import Base  # Import Base from your database setup
import uuid


class ModelResponse(Base):
    __tablename__ = "model_response"

    response_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query_id = Column(UUID(as_uuid=True), ForeignKey("user_prompt.query_id", ondelete="CASCADE"), index=True)
    chat_id = Column(UUID(as_uuid=True), ForeignKey("chat.chat_id", ondelete="CASCADE"), index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("user.user_id", ondelete="CASCADE"), index=True)
    model_response = Column(Text, nullable=False)
    date_sent = Column(TIMESTAMP, server_default=func.now())

    chat = relationship("Chat", back_populates="responses")
    prompt = relationship("UserPrompt", back_populates="response")
    user = relationship("User")

