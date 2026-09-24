"""
Business logic for the Chat module (OpenAI-only).

Replaces the old MODEL_ENDPOINT / OCR.space flow with llm_service:
- send_message(): 1 DB fetch + 1 OpenAI chat_complete() + single transaction
- extract_text_from_image(): OpenAI vision (no OCR.space key)
"""
import io
from uuid import UUID, uuid4

from PIL import Image
from sqlalchemy.orm import Session, joinedload

from api.v1.models.chat import Chat
from api.v1.models.userprompt import UserPrompt
from api.v1.models.modelresponse import ModelResponse
from api.v1.services import llm_service
from api.core.exceptions import NotFoundException, BadRequestException


# ---------------------------------------------------------------------------
# Chat session management
# ---------------------------------------------------------------------------

def create_chat_session(db: Session, user_id: UUID, chat_title: str) -> dict:
    new_chat = Chat(
        chat_id=uuid4(),
        user_id=user_id,
        chat_title=(chat_title or "Untitled Chat").strip()[:100],
    )
    db.add(new_chat)
    db.commit()
    db.refresh(new_chat)

    return {
        "message": "Chat session created successfully",
        "chat_id": str(new_chat.chat_id),
        "chat_title": new_chat.chat_title,
        "created_at": new_chat.created_at,
    }


def delete_chat_session(db: Session, user_id: UUID, chat_id: UUID) -> dict:
    chat_session = (
        db.query(Chat).filter_by(chat_id=chat_id, user_id=user_id).first()
    )
    if not chat_session:
        raise NotFoundException("Chat not found")

    # Relationships cascade, single delete is enough.
    db.delete(chat_session)
    db.commit()

    return {"message": "Chat session deleted successfully"}


def get_chat_history(
    db: Session, user_id: UUID, chat_id: UUID, skip: int = 0, limit: int = 50
) -> list:
    chat_session = (
        db.query(Chat).filter_by(chat_id=chat_id, user_id=user_id).first()
    )
    if not chat_session:
        raise NotFoundException("Chat not found")

    history = (
        db.query(UserPrompt)
        .options(joinedload(UserPrompt.response))
        .filter_by(chat_id=chat_id, user_id=user_id)
        .order_by(UserPrompt.created_at)
        .offset(skip)
        .limit(limit)
        .all()
    )

    return [
        {
            "query": item.query,
            "response": item.response.model_response if item.response else None,
            "timestamp": item.created_at,
        }
        for item in history
    ]


def get_all_sessions(
    db: Session, user_id: UUID, skip: int = 0, limit: int = 20
) -> list:
    chat_sessions = (
        db.query(Chat)
        .filter(Chat.user_id == user_id)
        .order_by(Chat.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )

    return [
        {
            "chat_id": str(s.chat_id),
            "chat_title": s.chat_title,
            "created_at": s.created_at,
        }
        for s in chat_sessions
    ]


# ---------------------------------------------------------------------------
# LLM interaction (OpenAI, async)
# ---------------------------------------------------------------------------

async def send_message(
    db: Session, user_id: UUID, chat_id: UUID, prompt: str
) -> dict:
    """Build history locally, call OpenAI once, persist in one transaction."""
    prompt = (prompt or "").strip()
    if not prompt:
        raise BadRequestException("Prompt must not be empty.")
    if len(prompt) > 4000:
        raise BadRequestException("Prompt is too long (max 4000 characters).")

    chat_session = (
        db.query(Chat).filter_by(chat_id=chat_id, user_id=user_id).first()
    )
    if not chat_session:
        raise NotFoundException("Chat session not found.")

    past_turns = (
        db.query(UserPrompt)
        .options(joinedload(UserPrompt.response))
        .filter_by(chat_id=chat_id, user_id=user_id)
        .order_by(UserPrompt.created_at.desc())
        .limit(10)
        .all()
    )
    past_turns_reversed = list(reversed(past_turns))

    last_task_type = past_turns_reversed[-1].task_type if past_turns_reversed else None

    # Local history (no embedding-service call).
    from api.utils.context import past_turns_to_dicts

    history_dicts = past_turns_to_dicts(past_turns_reversed)
    messages = llm_service.build_history_messages(history_dicts, prompt)

    model_text = await llm_service.chat_complete(messages)

    # Cheap separate classification only when this isn't a follow-up;
    # otherwise inherit last turn's label.
    lowered = prompt.strip().lower()
    is_followup = lowered.startswith(("why", "how", "what does")) or lowered in {
        "explain", "please explain", "can you explain that?", "what?",
    }
    if is_followup and last_task_type:
        task_type = last_task_type
    else:
        task_type = await llm_service.classify_task(prompt)

    # Single transaction for prompt + response.
    try:
        prompt_obj = UserPrompt(
            query_id=uuid4(),
            user_id=user_id,
            chat_id=chat_id,
            query=prompt,
            task_type=task_type,
        )
        db.add(prompt_obj)
        db.flush()

        response_obj = ModelResponse(
            response_id=uuid4(),
            query_id=prompt_obj.query_id,
            user_id=user_id,
            chat_id=chat_id,
            model_response=model_text,
        )
        db.add(response_obj)
        db.commit()
        db.refresh(prompt_obj)
        db.refresh(response_obj)
    except Exception:
        db.rollback()
        raise

    return {
        "chat_id": str(chat_id),
        "query_id": str(prompt_obj.query_id),
        "response": model_text,
        "task_type": task_type,
    }


# ---------------------------------------------------------------------------
# Image processing (OpenAI vision replaces OCR.space)
# ---------------------------------------------------------------------------

def compress_image(image_bytes: bytes, max_size_kb=1024, max_dim=1000) -> bytes:
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")

    if max(img.size) > max_dim:
        img.thumbnail((max_dim, max_dim))

    min_q, max_q = 10, 95
    best_compressed = None
    compressed = image_bytes

    while min_q <= max_q:
        mid_q = (min_q + max_q) // 2
        output = io.BytesIO()
        img.save(output, format="JPEG", optimize=True, quality=mid_q)
        compressed = output.getvalue()
        if len(compressed) <= max_size_kb * 1024:
            best_compressed = compressed
            min_q = mid_q + 1
        else:
            max_q = mid_q - 1

    return best_compressed if best_compressed else compressed


async def extract_text_from_image(image_data: bytes) -> str:
    """Vision transcription via OpenAI (no OCR.space, no retries needed here)."""
    if not image_data:
        raise BadRequestException("Empty image data.")
    return await llm_service.vision_extract_text(image_data)
