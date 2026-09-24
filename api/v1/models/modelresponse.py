import uuid

from sqlalchemy import Column, Text, ForeignKey, UUID
from sqlalchemy.orm import relationship

from api.db.database import Base
from api.v1.models.mixins import TimestampMixin


class ModelResponse(TimestampMixin, Base):
    __tablename__ = "model_response"

    response_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query_id = Column(
        UUID(as_uuid=True),
        ForeignKey("user_prompt.query_id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
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
    model_response = Column(Text, nullable=False)

    chat = relationship("Chat", back_populates="responses")
    prompt = relationship("UserPrompt", back_populates="response")
    user = relationship("User")
