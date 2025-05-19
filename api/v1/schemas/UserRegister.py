from pydantic import BaseModel, EmailStr, Field
from typing import Optional


class UserCreate(BaseModel):
    firstname: str = Field(..., min_length=1, max_length=50)
    lastname: str = Field(..., min_length=1, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=6)

    class Config:
        str_strip_whitespace = True

class UserSignin(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6)

    class Config:
        str_strip_whitespace = True

class UserUpdate(BaseModel):
    educational_level: Optional[str] = None
    firstname: Optional[str] = None
    lastname: Optional[str] = None
    age: Optional[int] = None