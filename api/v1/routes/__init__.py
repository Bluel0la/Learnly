from fastapi import APIRouter
from api.v1.routes.authentication import auth
from api.v1.routes.chat import chat
from api.v1.routes.flashcards import flashcards
api_version_one = APIRouter(prefix="/api/v1")

api_version_one.include_router(auth)
api_version_one.include_router(chat)
api_version_one.include_router(flashcards)