import enum
import uuid

from sqlalchemy import Column, String, Integer, Enum as SAEnum, UUID
from sqlalchemy.orm import relationship

from api.db.database import Base
from api.v1.models.mixins import TimestampMixin, SoftDeleteMixin


class GenderEnum(str, enum.Enum):
    male = "male"
    female = "female"
    other = "other"
    prefer_not_to_say = "prefer_not_to_say"


class User(TimestampMixin, SoftDeleteMixin, Base):
    __tablename__ = "user"

    user_id = Column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True
    )
    firstname = Column(String(100))
    lastname = Column(String(100))
    educational_level = Column(String(100), nullable=True)
    email = Column(String(255), unique=True, nullable=False)
    password = Column(String, nullable=False)
    gender = Column(SAEnum(GenderEnum), nullable=True)
    age = Column(Integer)

    chats = relationship("Chat", back_populates="user", cascade="all, delete")
    decks = relationship("Deck", back_populates="user", cascade="all, delete")
    quizzes = relationship("Quiz", back_populates="user", cascade="all, delete")
    refresh_tokens = relationship(
        "RefreshToken", back_populates="user", cascade="all, delete"
    )
    revoked_tokens = relationship(
        "RevokedToken", back_populates="user", cascade="all, delete"
    )
    quizzers = relationship("Quizzer", back_populates="user", cascade="all, delete")
