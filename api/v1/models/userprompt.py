from uuid import uuid4

from sqlalchemy import Column, ForeignKey, Text, UUID, String
from sqlalchemy.orm import relationship

from api.db.database import Base
from api.v1.models.mixins import TimestampMixin


class UserPrompt(TimestampMixin, Base):
    __tablename__ = "user_prompt"

    query_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    chat_id = Column(
        UUID(as_uuid=True),
        ForeignKey("chat.chat_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    query = Column(Text, nullable=False)
    task_type = Column(String(50), nullable=True, default="general_chat")

    chat = relationship("Chat", back_populates="prompts")
    user = relationship("User")
    response = relationship(
        "ModelResponse", back_populates="prompt", uselist=False, cascade="all, delete"
    )
