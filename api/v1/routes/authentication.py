from fastapi import APIRouter, HTTPException, Depends, status
from sqlalchemy.orm import Session
from api.db.database import get_db  
from api.v1.models.user import User 
from api.utils.authentication import get_current_user
from api.v1.models.revoked_tokens import RevokedToken
from api.v1.schemas.UserRegister import UserCreate, UserSignin
from api.utils.authentication import hash_password, verify_password, create_access_token, decode_access_token
from jose import jwt
from fastapi.security import OAuth2PasswordBearer
from dotenv import load_dotenv
import os
blacklisted_tokens = set()


load_dotenv(".env")
ALGORITHM = os.getenv("ALGORITHM")
SECRET_KEY = os.getenv("SECRET")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

auth = APIRouter(prefix="/auth", tags=["Authentication"])

# User Registration
@auth.post("/signup", status_code=status.HTTP_201_CREATED)
def register_user(user_data: UserCreate, db: Session = Depends(get_db)):
    # Check if email already exists
    existing_user = db.query(User).filter(User.email == user_data.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    # Hash password
    hashed_pwd = hash_password(user_data.password)

    # Create new user
    new_user = User(
        firstname=user_data.firstname,
        lastname=user_data.lastname,
        email=user_data.email,
        password=hashed_pwd,
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return {"message": "User registered successfully", "user_id": new_user.user_id}

# User Login
@auth.post("/login")
def login(user_data: UserSignin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == user_data.email).first()
    if not user or not verify_password(user_data.password, user.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Generate JWT
    access_token = create_access_token(data={"sub": user.email})

    return {"access_token": access_token, "token_type": "bearer"}


@auth.get("/me")
def get_current_user_details(
    token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)
):
    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token")

    email = payload.get("sub")
    user = db.query(User).filter(User.email == email).first()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return {
        "user_id": user.user_id,
        "first_name": user.firstname,
        "last_name": user.lastname,
        "gender": user.gender,
        "age": user.age,
        "email": user.email,
        "educational_level": user.educational_level,
    }


@auth.post("/logout")
def logout(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    """
    Logout the user by revoking their access token.
    """
    # Add the token to the RevokedToken table
    revoked_token = RevokedToken(token=token)
    db.add(revoked_token)
    db.commit()

    return {"detail": "Successfully logged out"}
