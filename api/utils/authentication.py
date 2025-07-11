from passlib.context import CryptContext
from datetime import datetime, timedelta
from jose import jwt, JWTError
from dotenv import load_dotenv
import os
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from api.db.database import get_db
from api.v1.models.user import User
from api.v1.models.revoked_tokens import RevokedToken
from api.v1.models.refresh_tokens import RefreshToken


load_dotenv(".env")

ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES"))
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Hash password
def hash_password(password: str) -> str:
    return pwd_context.hash(password)

# Verify password
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


# Check if the token has been blacklisted
def is_token_revoked(db: Session, token: str) -> bool:
    return db.query(RevokedToken).filter_by(token=token).first() is not None


# Function to revoke a token
def revoke_token(db: Session, token: str, user_id: str):
    expires_at = datetime.utcnow() + timedelta(
        days=7
    )  # or extract from token if preferred
    revoked = RevokedToken(token=token, user_id=user_id, expires_at=expires_at)
    db.add(revoked)
    db.commit()


# Generate JWT token
def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str, db: Session):
    if is_token_revoked(db, token):
        raise HTTPException(status_code=401, detail="Token has been revoked.")

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None


def get_current_user(
    token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)
) -> User:
    if (
        db.query(RevokedToken).filter(RevokedToken.token == token).first()
        or db.query(RefreshToken).filter(RefreshToken.token == token).first()
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Token is revoked"
        )

    payload = decode_access_token(token, db)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
        )

    email = payload.get("sub")
    if not email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token data"
        )

    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    return user  #  Now returns a User object instead of a dict


def is_admin(
    current_user: User = Depends(get_current_user), db: Session = Depends(get_db)
) -> bool:
    """
    Check if the current user has an admin role.
    """
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Access denied: Admins only")

    return True
