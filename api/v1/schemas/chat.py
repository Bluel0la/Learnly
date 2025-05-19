from pydantic import BaseModel
from typing import Optional
from uuid import UUID

class ModelRequest(BaseModel):
    prompt: str
    chat_id: UUID  # 🆕 Required to link prompt to chat session


class ModelResponse(BaseModel):
    chat_id: UUID
    query_id: UUID
    response: str


class ChatCreate(BaseModel):
    chat_title: Optional[str] = "Untitled Chat"
