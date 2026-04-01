"""
Fetch semantically relevant past conversation turns from a remote embedding service.

Converted from synchronous `requests` to async `httpx` to avoid blocking
FastAPI's event loop during chat message processing.
"""
import os
from typing import List, Dict

import httpx
from dotenv import load_dotenv

from api.v1.models.userprompt import UserPrompt
from api.core.exceptions import ExternalServiceException

load_dotenv(".env")

model_util_endpoint = os.getenv("MODEL_UTILITY")


async def fetch_relevant_turns(
    current_prompt: str, past_turns: List[UserPrompt]
) -> List[Dict[str, str]]:
    remote_endpoint = f"{model_util_endpoint}/relevant_turn"

    serialized_turns = []
    for turn in past_turns:
        if turn.response:
            serialized_turns.append(
                {"text": f"User: {turn.query}\nAI: {turn.response.model_response}"}
            )

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                remote_endpoint,
                json={"query": current_prompt, "turns": serialized_turns, "top_k": 3},
            )
            response.raise_for_status()
            result = response.json()
            return result.get("relevant_turns", [])

        except httpx.RequestError as e:
            raise ExternalServiceException(f"Embedding service error: {str(e)}")
        except httpx.HTTPStatusError as e:
            raise ExternalServiceException(
                f"Embedding service returned status {e.response.status_code}"
            )
