"""
Chat route handlers — thin wrappers that delegate to chat_service.
"""
import json
from fastapi import APIRouter, Depends, UploadFile, File, Query, status
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.orm import Session
from uuid import UUID

from api.db.database import get_db
from api.v1.models.user import User
from api.utils.authentication import get_current_user
from api.utils.ocr_clean import normalize_text, clean_ocr_text
from api.utils.rates import is_rate_limited
from api.v1.schemas.chat import ModelRequest, ModelResponse as ModelResponseSchema, ChatCreate
from api.v1.services import chat_service
from api.core.exceptions import RateLimitedException, BadRequestException

chat = APIRouter(prefix="/chat", tags=["Chat"])

MAX_IMAGE_BYTES = 10 * 1024 * 1024


@chat.post("/start-session", status_code=status.HTTP_201_CREATED)
def create_chat(
    chat_data: ChatCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return chat_service.create_chat_session(
        db, user_id=current_user.user_id, chat_title=chat_data.chat_title
    )


@chat.delete("/delete-chat/{chat_id}")
def delete_chat(
    chat_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return chat_service.delete_chat_session(db, current_user.user_id, chat_id)


@chat.post("/send-message", response_model=ModelResponseSchema)
async def query_model(
    user_input: ModelRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if is_rate_limited(f"chat:{current_user.user_id}"):
        raise RateLimitedException()
    return await chat_service.send_message(
        db,
        user_id=current_user.user_id,
        chat_id=user_input.chat_id,
        prompt=user_input.prompt,
    )


@chat.post("/send-message-stream")
async def query_model_stream(
    user_input: ModelRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """SSE stream: `data: {"token": ...}` events, then `data: {"done": {...}}`
    (or `data: {"error": ...}`). Persists only on full completion."""
    if is_rate_limited(f"chat:{current_user.user_id}"):
        raise RateLimitedException()

    async def event_gen():
        async for event in chat_service.send_message_stream(
            db,
            user_id=current_user.user_id,
            chat_id=user_input.chat_id,
            prompt=user_input.prompt,
        ):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@chat.get("/session/{chat_id}")
def get_chat_history(
    chat_id: UUID,
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return chat_service.get_chat_history(
        db, current_user.user_id, chat_id, skip=skip, limit=limit
    )


@chat.get("/sessions")
def get_all_chat_sessions(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return chat_service.get_all_sessions(
        db, current_user.user_id, skip=skip, limit=limit
    )


@chat.post("/extract-text/")
async def extract_text(
    file: UploadFile = File(...), current_user: User = Depends(get_current_user)
):
    """Image -> OpenAI vision transcription (OCR.space removed)."""
    ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise BadRequestException(
            f"Unsupported file type: {file.content_type}. Please upload a JPEG, PNG, or WEBP image."
        )

    if is_rate_limited(f"ocr:{current_user.user_id}"):
        raise RateLimitedException()

    image_bytes = await file.read()
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise BadRequestException("Image too large. Maximum size is 10MB.")
    if not image_bytes:
        raise BadRequestException("Uploaded image is empty.")

    compressed_image = chat_service.compress_image(image_bytes)
    parsed_text = await chat_service.extract_text_from_image(compressed_image)
    normalized = normalize_text(parsed_text)
    corrected = clean_ocr_text(normalized)

    return JSONResponse(content={"text": corrected})
