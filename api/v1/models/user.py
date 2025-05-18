from sqlalchemy import Column, String, Boolean, Enum, UUID, Integer
from sqlalchemy.orm import relationship
from api.db.database import Base
import uuid


class User(Base):
    __tablename__ = "user"

    user_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    firstname = Column(String(100))
    lastname = Column(String(100))
    educational_level = Column(String(100))
    email = Column(String(255), unique=True, nullable=False)
    password = Column(String, nullable=False)
    gender = Column(String(10))
    age = Column(Integer)

    chats = relationship("Chat", back_populates="user", cascade="all, delete")
    decks = relationship("Deck", back_populates="user", cascade="all, delete")
    quizzes = relationship("Quiz", back_populates="user", cascade="all, delete")
    refresh_tokens = relationship(
        "RefreshToken", back_populates="user", cascade="all, delete"
    )  # Add this line
    revoked_tokens = relationship(
        "RevokedToken", back_populates="user", cascade="all, delete"
    )  # Add this line