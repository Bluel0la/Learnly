from collections import defaultdict
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Depends
from fastapi_utils.tasks import repeat_every
from typing import List

# --- Setup ---

app = FastAPI()

# user_id -> list of timestamps
user_request_log = defaultdict(list)
REQUEST_LIMIT = 5
WINDOW_SECONDS = 3600  # 1 hour

# --- Rate Limiting Utilities ---


def prune_old_requests(user_id: int):
    """Remove old timestamps beyond the sliding window."""
    now = datetime.utcnow()
    user_log = user_request_log[user_id]
    user_log[:] = [
        ts for ts in user_log if now - ts < timedelta(seconds=WINDOW_SECONDS)
    ]
    if not user_log:
        del user_request_log[user_id]


def is_rate_limited(user_id: int) -> bool:
    now = datetime.utcnow()
    prune_old_requests(user_id)

    user_log = user_request_log[user_id]
    if len(user_log) >= REQUEST_LIMIT:
        return True

    user_log.append(now)
    return False


def reset_all_request_logs():
    """Periodically clean up empty or outdated user logs."""
    now = datetime.utcnow()
    for uid in list(user_request_log.keys()):
        user_log = user_request_log[uid]
        user_log[:] = [
            ts for ts in user_log if now - ts < timedelta(seconds=WINDOW_SECONDS)
        ]
        if not user_log:
            del user_request_log[uid]


# --- Background Task (Runs every 10 minutes) ---


@app.on_event("startup")
@repeat_every(seconds=600)  # every 10 minutes
def periodic_cleanup():
    reset_all_request_logs()


# --- Dependency Simulation (Replace with actual auth) ---


def get_current_user_id() -> int:
    # Placeholder for authenticated user ID
    return 1


# --- Protected Endpoint Example ---


@app.get("/protected-resource/")
def access_protected(current_user_id: int = Depends(get_current_user_id)):
    if is_rate_limited(current_user_id):
        raise HTTPException(
            status_code=429, detail="Rate limit exceeded. Try again later."
        )

    return {"message": " Access granted. You are within the rate limit."}

# utils/chat_prompt_builder.py


def build_chat_prompt(
    past_turns: list, new_prompt: str, system_message: str = None
) -> str:
    """
    Format the chat history + user input into a prompt for the model.

    Args:
        past_turns: List of UserPrompt objects (with .response attribute loaded).
        new_prompt: The current user query.
        system_message: Optional system prefix to guide tone.

    Returns:
        Full formatted prompt string.
    """
    prompt_lines = []

    if system_message:
        prompt_lines.append(f"System: {system_message.strip()}")

    for turn in past_turns:
        prompt_lines.append(f"User: {turn.query.strip()}")
        if turn.response:
            prompt_lines.append(f"AI: {turn.response.model_response.strip()}")

    prompt_lines.append(f"User: {new_prompt.strip()}")
    prompt_lines.append("AI:")

    return "\n".join(prompt_lines)
