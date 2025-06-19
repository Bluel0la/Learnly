import requests, os
from fastapi import HTTPException, status
from api.v1.models.userprompt import UserPrompt
from typing import List, Dict
from dotenv import load_dotenv
load_dotenv(".env")

model_util_endpoint = os.getenv("MODEL_UTILITY")


def fetch_relevant_turns(
    current_prompt: str, past_turns: List[UserPrompt]
) -> List[Dict[str, str]]:
    remote_endpoint = f"{model_util_endpoint}/relevant_turn"

    serialized_turns = []
    for turn in past_turns:
        if turn.response:
            serialized_turns.append(
                {"text": f"User: {turn.query}\nAI: {turn.response.model_response}"}
            )

    try:
        response = requests.post(
            remote_endpoint,
            json={"query": current_prompt, "turns": serialized_turns, "top_k": 3},
        )

        if response.status_code != 200:
            raise ValueError(
                f"Remote embedding similarity failed with status: {response.status_code}"
            )

        result = response.json()
        return result.get("relevant_turns", [])

    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Embedding service error: {str(e)}",
        )
