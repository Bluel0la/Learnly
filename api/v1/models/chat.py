from uuid import uuid4

from sqlalchemy import Column, String, UUID, ForeignKey
from sqlalchemy.orm import relationship

from api.db.database import Base
from api.v1.models.mixins import TimestampMixin


class Chat(TimestampMixin, Base):
    __tablename__ = "chat"

    chat_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4, index=True)
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chat_title = Column(String(255), nullable=False, default="Untitled Chat")

    user = relationship("User", back_populates="chats")
    prompts = relationship("UserPrompt", back_populates="chat", cascade="all, delete")
    responses = relationship(
        "ModelResponse", back_populates="chat", cascade="all, delete"
    )
