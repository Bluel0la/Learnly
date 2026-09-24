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
    __tablename__ = "users"

    user_id = Column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True
    )
    firstname = Column(String(100), nullable=False)
    lastname = Column(String(100), nullable=False)
    educational_level = Column(String(100), nullable=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column("password", String, nullable=False)
    gender = Column(SAEnum(GenderEnum), nullable=True)
    age = Column(Integer, nullable=True)

    chats = relationship("Chat", back_populates="user", cascade="all, delete-orphan")
    decks = relationship("Deck", back_populates="user", cascade="all, delete-orphan")
    refresh_tokens = relationship(
        "RefreshToken", back_populates="user", cascade="all, delete-orphan"
    )
    revoked_tokens = relationship(
        "RevokedToken", back_populates="user", cascade="all, delete-orphan"
    )
    quizzers = relationship("Quizzer", back_populates="user", cascade="all, delete-orphan")

    # Back-compat: old code uses user.password
    @property
    def password(self) -> str:
        return self.hashed_password

    @password.setter
    def password(self, value: str) -> None:
        self.hashed_password = value
