from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from uuid import uuid4, UUID
from datetime import datetime
import random

from api.db.database import get_db
from api.v1.models.user import User
from api.utils.authentication import get_current_user
from api.v1.schemas import quiz as schemas
# from api.utils.maths_generators


quiz_router = APIRouter(prefix="/quiz/math", tags=["Quiz - Math"])
