from pydantic import BaseModel, Field
from typing import Optional
from uuid import UUID


class ModelRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    chat_id: UUID  # Required to link prompt to chat session


class ModelResponse(BaseModel):
    chat_id: UUID
    query_id: UUID
    response: str
    task_type: Optional[str] = None


class ChatCreate(BaseModel):
    chat_title: Optional[str] = Field(default="Untitled Chat", min_length=1, max_length=100)
