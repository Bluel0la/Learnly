from fastapi import APIRouter, HTTPException, Depends, status, BackgroundTasks
from sqlalchemy.orm import Session
from api.db.database import get_db  
from api.v1.models.user import User 
from api.utils.authentication import get_current_user
from api.v1.models.revoked_tokens import RevokedToken
from api.v1.schemas.UserRegister import UserCreate, UserSignin, UserUpdate
from api.utils.authentication import hash_password, verify_password, create_access_token, decode_access_token, revoke_token
from jose import jwt, JWTError
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
    payload = decode_access_token(token, db)
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
def logout(
    background_tasks: BackgroundTasks,
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
):
    try:
        payload = jwt.decode(
            token, SECRET_KEY, algorithms=[ALGORITHM]
        )
        email = payload.get("sub")
        if email is None:
            raise ValueError("Token payload missing subject (email).")
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
        )

    # 🔍 Query user by email
    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    background_tasks.add_task(revoke_token, db, token, user.id)
    return {"detail": "Successfully logged out"}


# Update User Information
@auth.put("/update")
def update_details(
    user_data: UserUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):

    user = db.query(User).filter(User.user_id == current_user.user_id).first()

    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

        # Update user fields if provided
    if user_data.firstname is not None:
        user.firstname = user_data.firstname
    if user_data.lastname is not None:
        user.lastname = user_data.lastname
    if user_data.educational_level is not None:
        user.educational_level = user_data.educational_level
    if user_data.age is not None:
        user.age = user_data.age

    # Commit the changes to the database
    db.commit()
    db.refresh(user)

    return{
        "message": "User detail successfully updated",
        "user_id": current_user.user_id
    }

# Delete a user's account
@auth.delete("/delete")
def delete_account(
    db: Session = Depends(get_db), current_user: User = Depends(get_current_user)
):
    user = db.query(User).filter(User.user_id == current_user.user_id).first()

    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    db.delete(user)
    db.commit()

    return {"message": "User account successfully deleted"}
