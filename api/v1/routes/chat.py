from api.v1.schemas.chat import ModelRequest, ModelResponse as ModelResponseSchema, ChatCreate
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from api.utils.authentication import get_current_user
from api.v1.models.modelresponse import ModelResponse
from api.v1.models.userprompt import UserPrompt
from sqlalchemy.orm import Session, joinedload
from fastapi.responses import JSONResponse
from api.v1.models.user import User
from api.v1.models.chat import Chat
from api.db.database import get_db
from dotenv import load_dotenv
from langdetect import detect
from uuid import uuid4
from uuid import UUID
import numpy as np
import requests
import re
import os

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


@chat.post("/send-message", response_model=ModelResponseSchema)
def query_model(
    user_input: ModelRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # Ensure chat session belongs to the user
    chat_session = (
        db.query(Chat)
        .filter_by(chat_id=user_input.chat_id, user_id=current_user.user_id)
        .first()
    )
    if not chat_session:
        raise HTTPException(status_code=404, detail="Chat session not found.")

    try:
        model_api_response = requests.post(
            model_endpoint, json={"prompt": user_input.prompt}
        )

        if model_api_response.status_code != 200:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Failed to get response from model server.",
            )

        model_response_data = model_api_response.json()
        model_text = model_response_data.get("response")
        if not model_text:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Model response is empty or malformed.",
            )

        # Save prompt
        prompt = UserPrompt(
            query_id=uuid4(),
            user_id=current_user.user_id,
            chat_id=user_input.chat_id,
            query=user_input.prompt,
        )
        db.add(prompt)
        db.commit()
        db.refresh(prompt)

        # Save response
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
        }

    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model service unreachable: {str(e)}",
        )


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


@chat.post("/extract-text/")
async def extract_text(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        response = requests.post(
            url="https://api.ocr.space/parse/image",
            files={"filename": (file.filename, contents)},
            data={
                "apikey": OCR_API_KEY,
                "language": "eng",  # Auto: 'eng', or others like 'spa', 'fra', 'deu'
                "OCREngine": "2",  # 1 = default, 2 = improved engine
            },
        )
        result = response.json()

        if result.get("IsErroredOnProcessing"):
            return JSONResponse(
                content={"error": result.get("ErrorMessage")}, status_code=400
            )

        parsed_text = result["ParsedResults"][0]["ParsedText"]
        return JSONResponse(content={"text": parsed_text})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
