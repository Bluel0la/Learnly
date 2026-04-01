"""
Business logic for the Chat module.

Handles LLM classification, conversation history building, prompt/response
persistence, and image OCR. All external API calls use async httpx.
"""
import io
import os
from uuid import UUID, uuid4

import httpx
from dotenv import load_dotenv
from PIL import Image
from sqlalchemy.orm import Session, joinedload

from api.v1.models.chat import Chat
from api.v1.models.userprompt import UserPrompt
from api.v1.models.modelresponse import ModelResponse
from api.core.exceptions import (
    NotFoundException,
    ExternalServiceException,
    BadRequestException,
)

load_dotenv(".env")

MODEL_ENDPOINT = os.getenv("MODEL_ENDPOINT")
OCR_API_KEY = os.getenv("OCR_API")


# ---------------------------------------------------------------------------
# Follow-up detection
# ---------------------------------------------------------------------------

def is_followup_question(query: str) -> bool:
    """Heuristic to detect if a prompt is a clarification/follow-up."""
    query = query.strip().lower()
    return (
        query.startswith("why")
        or query.startswith("how")
        or query.startswith("what does")
        or query in {"explain", "please explain", "can you explain that?", "what?"}
    )


# ---------------------------------------------------------------------------
# Chat session management
# ---------------------------------------------------------------------------

def create_chat_session(db: Session, user_id: UUID, chat_title: str) -> dict:
    new_chat = Chat(
        chat_id=uuid4(),
        user_id=user_id,
        chat_title=chat_title,
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

    db.query(UserPrompt).filter_by(chat_id=chat_id, user_id=user_id).delete()
    db.query(ModelResponse).filter_by(chat_id=chat_id, user_id=user_id).delete()
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
# LLM interaction (async)
# ---------------------------------------------------------------------------

async def _call_llm(prompt: str) -> dict:
    """Make an async POST to the LLM endpoint. Returns the JSON response."""
    url = f"{MODEL_ENDPOINT}/chat"
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(url, json={"prompt": prompt})
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as e:
            raise ExternalServiceException(f"LLM request failed: {str(e)}")
        except httpx.HTTPStatusError:
            raise ExternalServiceException("Failed to get response from model server.")


async def send_message(
    db: Session, user_id: UUID, chat_id: UUID, prompt: str
) -> dict:
    """Classify the prompt, build context, call LLM, and persist the exchange."""
    # Validate chat session
    chat_session = (
        db.query(Chat).filter_by(chat_id=chat_id, user_id=user_id).first()
    )
    if not chat_session:
        raise NotFoundException("Chat session not found.")

    # Fetch recent turns
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

    # Determine task type
    if is_followup_question(prompt) and last_task_type:
        task_type = last_task_type
    else:
        first_data = await _call_llm(prompt)
        task_type = first_data.get("task_type")
        if not first_data.get("response"):
            raise ExternalServiceException("Model response is empty or malformed.")

    # Fetch relevant turns (async version)
    from api.utils.context import fetch_relevant_turns

    relevant_turns = await fetch_relevant_turns(prompt, past_turns_reversed)

    # Build conversation history
    conversation_history = []
    for turn in relevant_turns:
        if turn.get("user"):
            conversation_history.append(f"User: {turn['user']}")
        if turn.get("ai"):
            conversation_history.append(f"AI: {turn['ai']}")

    conversation_history.append(f"User: {prompt}")
    conversation_history.append("AI:")
    full_prompt = "\n".join(conversation_history)

    # Final LLM call
    final_data = await _call_llm(full_prompt)
    model_text = final_data.get("response")
    task_type = final_data.get("task_type", task_type)

    # Persist prompt and response
    prompt_obj = UserPrompt(
        query_id=uuid4(),
        user_id=user_id,
        chat_id=chat_id,
        query=prompt,
        task_type=task_type,
    )
    db.add(prompt_obj)
    db.commit()
    db.refresh(prompt_obj)

    response_obj = ModelResponse(
        response_id=uuid4(),
        query_id=prompt_obj.query_id,
        user_id=user_id,
        chat_id=chat_id,
        model_response=model_text,
    )
    db.add(response_obj)
    db.commit()
    db.refresh(response_obj)

    return {
        "chat_id": str(chat_id),
        "query_id": str(prompt_obj.query_id),
        "response": model_text,
        "task_type": task_type,
    }


# ---------------------------------------------------------------------------
# Image processing & OCR
# ---------------------------------------------------------------------------

def compress_image(image_bytes: bytes, max_size_kb=1024, max_dim=1000) -> bytes:
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")

    if max(img.size) > max_dim:
        img.thumbnail((max_dim, max_dim))

    min_q, max_q = 10, 95
    best_compressed = None

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


async def extract_text_from_image(image_data: bytes, retries: int = 3) -> str:
    """Send image to OCR.space API and return cleaned text."""
    delay = 2
    async with httpx.AsyncClient(timeout=60.0) as client:
        for attempt in range(retries):
            try:
                response = await client.post(
                    url="https://api.ocr.space/parse/image",
                    files={"filename": ("compressed.jpg", image_data)},
                    data={
                        "apikey": OCR_API_KEY,
                        "language": "eng",
                        "OCREngine": "2",
                    },
                )
                response.raise_for_status()
                result = response.json()

                if result.get("IsErroredOnProcessing"):
                    error_msg = result.get("ErrorMessage", "Unknown error")
                    raise BadRequestException(error_msg)

                return result["ParsedResults"][0].get("ParsedText", "")

            except httpx.RequestError:
                if attempt < retries - 1:
                    import asyncio

                    await asyncio.sleep(delay)
                    delay *= 2
                else:
                    raise ExternalServiceException("OCR service request failed.")
