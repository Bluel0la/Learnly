from sqlalchemy import Column, Integer, String, UUID, TIMESTAMP, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from api.db.database import Base
from uuid import uuid4


class Chat(Base):
    __tablename__ = "chat"

    chat_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4, index=True)
    user_id = Column(
        UUID(as_uuid=True), ForeignKey("user.user_id", ondelete="CASCADE"), index=True
    )
    chat_title = Column(String(255))
    created_at = Column(TIMESTAMP, server_default=func.now())

    user = relationship("User", back_populates="chats")
    prompts = relationship("UserPrompt", back_populates="chat", cascade="all, delete")
    responses = relationship(
        "ModelResponse", back_populates="chat", cascade="all, delete"
    )
