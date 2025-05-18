from sqlalchemy import Column, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime, timedelta
from api.db.database import Base


class RevokedToken(Base):
    __tablename__ = "revoked_tokens"

    token = Column(String, primary_key=True, index=True)
    user_id = Column(ForeignKey("user.user_id", ondelete="CASCADE"), nullable=False)  # Add this line

    expires_at = Column(
        DateTime, nullable=False, default=lambda: datetime.utcnow() + timedelta(days=1)
    )
    user = relationship("User", back_populates="revoked_tokens")

