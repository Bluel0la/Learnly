from api.v1.schemas.chat import ModelRequest, ModelResponse as ModelResponseSchema, ChatCreate
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from api.utils.authentication import get_current_user
from api.utils.ocr_clean import normalize_text, clean_ocr_text
from api.utils.rates import is_rate_limited
from api.v1.models.modelresponse import ModelResponse
from api.v1.models.userprompt import UserPrompt
from sqlalchemy.orm import Session, joinedload
from fastapi.responses import JSONResponse
from api.utils.context import fetch_relevant_turns
from api.v1.models.user import User
from api.v1.models.chat import Chat
from api.db.database import get_db
from dotenv import load_dotenv
from PIL import Image
from uuid import uuid4
from uuid import UUID
import io, os, requests, httpx, asyncio

load_dotenv(".env")

model_endpoint = os.getenv("MODEL_ENDPOINT")
OCR_API_KEY = os.getenv("OCR_API")


chat = APIRouter(prefix="/chat", tags=["Chat"])


@chat.post("/start-session")
def create_chat(
    chat_data: ChatCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    new_chat = Chat(
        chat_id=uuid4(),
        user_id=current_user.user_id,
        chat_title=chat_data.chat_title,
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

@chat.delete("/delete-chat/{chat_id}")
def delete_chat(
    chat_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # Ensure chat belongs to current user
    chat_session = (
        db.query(Chat).filter_by(chat_id=chat_id, user_id=current_user.user_id).first()
    )

    if not chat_session:
        raise HTTPException(status_code=404, detail="Chat not found")

    # Delete associated prompts and responses
    db.query(UserPrompt).filter_by(chat_id=chat_id, user_id=current_user.user_id).delete()
    db.query(ModelResponse).filter_by(chat_id=chat_id, user_id=current_user.user_id).delete()

    # Delete the chat session itself
    db.delete(chat_session)
    db.commit()

    return {"message": "Chat session deleted successfully"}


def is_followup_question(query: str) -> bool:
    """Heuristic to detect if a prompt is a clarification/follow-up."""
    query = query.strip().lower()
    return (
        query.startswith("why")
        or query.startswith("how")
        or query.startswith("what does")
        or query in {"explain", "please explain", "can you explain that?", "what?"}
    )


@chat.post("/send-message", response_model=ModelResponseSchema)
def query_model(
    user_input: ModelRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # Step 1: Validate chat session
    chat_session = (
        db.query(Chat)
        .filter_by(chat_id=user_input.chat_id, user_id=current_user.user_id)
        .first()
    )
    if not chat_session:
        raise HTTPException(status_code=404, detail="Chat session not found.")

    # Step 2: Fetch past 10 turns
    past_turns = (
        db.query(UserPrompt)
        .options(joinedload(UserPrompt.response))
        .filter_by(chat_id=user_input.chat_id, user_id=current_user.user_id)
        .order_by(UserPrompt.date_sent.desc())
        .limit(10)
        .all()
    )
    past_turns_reversed = list(reversed(past_turns))
    last_task_type = past_turns_reversed[-1].task_type if past_turns_reversed else None

    # Step 3: Determine task type (follow-up override allowed)
    model_endpoint_url = f"{model_endpoint}/chat"
    try:
        if is_followup_question(user_input.prompt) and last_task_type:
            task_type = last_task_type
        else:
            # Fresh classification
            first_response = requests.post(
                model_endpoint_url, json={"prompt": user_input.prompt}
            )
            if first_response.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="Failed to get response from model server.",
                )
            model_response_data = first_response.json()
            task_type = model_response_data.get("task_type")

            if not model_response_data.get("response"):
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Model response is empty or malformed.",
                )

    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Initial model request failed: {str(e)}",
        )

    # Step 4: Fetch top-K relevant turns semantically from remote embedding service
    relevant_turns = fetch_relevant_turns(user_input.prompt, past_turns_reversed)

    # Step 5: Build structured conversation history
    conversation_history = []
    for turn in relevant_turns:
        if turn.get("user"):
            conversation_history.append(f"User: {turn['user']}")
        if turn.get("ai"):
            conversation_history.append(f"AI: {turn['ai']}")

    conversation_history.append(f"User: {user_input.prompt}")
    conversation_history.append("AI:")
    full_prompt = "\n".join(conversation_history)

    # Step 6: Final model call with full prompt
    try:
        final_response = requests.post(model_endpoint_url, json={"prompt": full_prompt})
        if final_response.status_code != 200:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Final model generation failed.",
            )

        final_data = final_response.json()
        model_text = final_data.get("response")
        task_type = final_data.get("task_type", task_type)

    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Final model request failed: {str(e)}",
        )

    # Step 7: Store user prompt and model response
    prompt = UserPrompt(
        query_id=uuid4(),
        user_id=current_user.user_id,
        chat_id=user_input.chat_id,
        query=user_input.prompt,
        task_type=task_type,
    )
    db.add(prompt)
    db.commit()
    db.refresh(prompt)

    response = ModelResponse(
        response_id=uuid4(),
        query_id=prompt.query_id,
        user_id=current_user.user_id,
        chat_id=user_input.chat_id,
        model_response=model_text,
    )
    db.add(response)
    db.commit()
    db.refresh(response)

    return {
        "chat_id": str(user_input.chat_id),
        "query_id": str(prompt.query_id),
        "response": model_text,
        "task_type": task_type,
    }


@chat.get("/session/{chat_id}")
def get_chat_history(
    chat_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # Ensure chat belongs to current user
    chat_session = (
        db.query(Chat).filter_by(chat_id=chat_id, user_id=current_user.user_id).first()
    )

    if not chat_session:
        raise HTTPException(status_code=404, detail="Chat not found")

    # Get all user prompts + joined responses ordered by date
    history = (
        db.query(UserPrompt)
        .options(joinedload(UserPrompt.response))
        .filter_by(chat_id=chat_id, user_id=current_user.user_id)
        .order_by(UserPrompt.date_sent)
        .all()
    )

    result = []
    for item in history:
        result.append(
            {
                "query": item.query,
                "response": item.response.model_response if item.response else None,
                "timestamp": item.date_sent,
            }
        )

    return result


@chat.get("/sessions")
def get_all_chat_sessions(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    chat_sessions = (
        db.query(Chat)
        .filter(Chat.user_id == current_user.user_id)
        .order_by(Chat.created_at.desc())
        .all()
    )

    return [
        {
            "chat_id": str(session.chat_id),
            "chat_title": session.chat_title,
            "created_at": session.created_at,
        }
        for session in chat_sessions
    ]


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


async def post_to_ocr_space_async(
    image_data: bytes, retries: int = 3, delay: int = 2,
) -> dict:
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
                return response.json()
            except httpx.RequestError:
                if attempt < retries - 1:
                    await asyncio.sleep(delay)
                    delay *= 2
                else:
                    raise

async def post_to_optiic_ocr_async(
    image_data: bytes, retries: int = 3, delay: int = 2,
) -> dict:
    url = "https://api.optiic.dev/process"

    async with httpx.AsyncClient(timeout=60.0) as client:
        for attempt in range(retries):
            try:
                response = await client.post(
                    url,
                    files={"image": ("image.jpg", image_data)},
                    data={"apiKey": OCR_API_KEY},
                )
                response.raise_for_status()
                return response.json()

            except httpx.RequestError as e:
                if attempt < retries - 1:
                    await asyncio.sleep(delay)
                    delay *= 2
                else:
                    raise RuntimeError(f"Optiic OCR request failed: {e}")


@chat.post("/extract-text/")
async def extract_text(
    file: UploadFile = File(...), current_user: User = Depends(get_current_user)
):

    #  Content-type validation
    ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        return JSONResponse(
            content={
                "error": f"Unsupported file type: {file.content_type}. Please upload a JPEG, PNG, or WEBP image."
            },
            status_code=415,
        )

    #  Rate limiting per user
    if is_rate_limited(current_user.user_id):
        return JSONResponse(
            content={
                "error": "Rate limit exceeded. Please wait a moment before trying again."
            },
            status_code=429,
        )

    try:
        image_bytes = await file.read()
        compressed_image = compress_image(image_bytes)
        result = await post_to_ocr_space_async(compressed_image)

        if result.get("IsErroredOnProcessing"):
            error_msg = result.get("ErrorMessage", "Unknown error")
            return JSONResponse(content={"error": error_msg}, status_code=400)

        parsed_text = result["ParsedResults"][0].get("ParsedText", "")
        normalized = normalize_text(parsed_text)
        corrected = clean_ocr_text(normalized)

        return JSONResponse(content={"text": corrected})

    except ValueError as ve:
        return JSONResponse(content={"error": str(ve)}, status_code=422)
    except Exception as e:
        return JSONResponse(
            content={"error": f"Unexpected server error: {e}"}, status_code=500
        )
