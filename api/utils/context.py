from typing import List, Dict

from api.v1.models.userprompt import UserPrompt


def past_turns_to_dicts(past_turns: List[UserPrompt]) -> List[Dict[str, str]]:
    turns: List[Dict[str, str]] = []
    for turn in past_turns:
        entry: Dict[str, str] = {}
        if turn.query:
            entry["user"] = turn.query
        if turn.response and turn.response.model_response:
            entry["ai"] = turn.response.model_response
        if entry:
            turns.append(entry)
    return turns


async def fetch_relevant_turns(
    current_prompt: str, past_turns: List[UserPrompt], max_turns: int = 6
) -> List[Dict[str, str]]:
    """Kept async for call-site compat; returns the last N turns locally."""
    _ = current_prompt
    dicts = past_turns_to_dicts(past_turns)
    return dicts[-max_turns:] if max_turns else dicts
