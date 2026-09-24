from pydantic import BaseModel, EmailStr, Field
from typing import Optional


class UserCreate(BaseModel):
    firstname: str = Field(..., min_length=1, max_length=50)
    lastname: str = Field(..., min_length=1, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=6, max_length=128)

    class Config:
        str_strip_whitespace = True

class UserSignin(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6, max_length=128)

    class Config:
        str_strip_whitespace = True

class UserUpdate(BaseModel):
    educational_level: Optional[str] = Field(default=None, min_length=1, max_length=100)
    firstname: Optional[str] = Field(default=None, min_length=1, max_length=50)
    lastname: Optional[str] = Field(default=None, min_length=1, max_length=50)
    age: Optional[int] = Field(default=None, ge=5, le=120)


class PasswordChange(BaseModel):
    current_password: str = Field(..., min_length=6, max_length=128)
    new_password: str = Field(..., min_length=6, max_length=128)