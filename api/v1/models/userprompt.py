from sqlalchemy import Column, Integer, ForeignKey, Text, TIMESTAMP, UUID, String
from uuid import uuid4
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from api.db.database import Base


class UserPrompt(Base):
    __tablename__ = "user_prompt"

    query_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    chat_id = Column(
        UUID(as_uuid=True), ForeignKey("chat.chat_id", ondelete="CASCADE"), index=True
    )
    user_id = Column(
        UUID(as_uuid=True), ForeignKey("user.user_id", ondelete="CASCADE"), index=True
    )
    query = Column(Text, nullable=False)
    task_type = Column(String, nullable=True)
    date_sent = Column(TIMESTAMP, server_default=func.now())

    chat = relationship("Chat", back_populates="prompts")
    user = relationship("User")
    response = relationship(
        "ModelResponse", back_populates="prompt", uselist=False, cascade="all, delete"
    )
