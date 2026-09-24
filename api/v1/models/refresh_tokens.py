import uuid
from sqlalchemy import Column, String, DateTime, ForeignKey
from sqlalchemy import UUID
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from api.db.database import Base  # Adjust based on your structure
from datetime import timedelta
from api.v1.models.mixins import TimestampMixin


class RefreshToken(TimestampMixin, Base):
    __tablename__ = "refresh_tokens"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    token = Column(String, nullable=False, unique=True, index=True)
    expires_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc) + timedelta(days=7),
    )

    user = relationship("User", back_populates="refresh_tokens")
