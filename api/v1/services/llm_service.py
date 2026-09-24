"""
OpenAI-only LLM service.

Replaces:
- POST {MODEL_ENDPOINT}/chat            -> chat_complete()
- POST {MODEL_UTILITY}/relevant_turn     -> build_history_messages() (local, no HTTP)
- POST {MODEL_UTILITY}/flashcard-summarization -> summarize_text()
- POST {MODEL_ENDPOINT}/flashcard        -> generate_flashcards()
- POST https://api.ocr.space/parse/image -> vision_extract_text()

All functions raise ExternalServiceException(503) on failure so routes
don't need to know about the OpenAI SDK.
"""
import base64
import json
import logging

from openai import AsyncOpenAI, APIError, RateLimitError

from api.core.config import settings
from api.core.exceptions import ExternalServiceException

logger = logging.getLogger(__name__)

_client: AsyncOpenAI | None = None


def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        if not settings.OPENAI_API_KEY:
            raise ExternalServiceException("OPENAI_API_KEY is not configured.")
        _client = AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY,
            timeout=settings.OPENAI_TIMEOUT_SECONDS,
        )
    return _client


def build_history_messages(
    past_turns: list[dict], current_prompt: str, max_turns: int = 6
) -> list[dict]:
    """Local replacement for the old /relevant_turn embedding service.

    Takes the most recent turns (already ordered oldest->newest) instead
    of a semantic top_k lookup. Simple, deterministic, no network call.
    Expects past_turns as [{"user": str, "ai": str}].
    """
    recent = past_turns[-max_turns:] if max_turns else past_turns
    messages: list[dict] = [
        {
            "role": "system",
            "content": (
                "You are Learnly, a patient tutor. Explain clearly, "
                "use simple formatting, and keep answers focused on the "
                "student's question. Reply with plain text unless asked otherwise."
            ),
        }
    ]
    for turn in recent:
        if turn.get("user"):
            messages.append({"role": "user", "content": turn["user"]})
        if turn.get("ai"):
            messages.append({"role": "assistant", "content": turn["ai"]})
    messages.append({"role": "user", "content": current_prompt})
    return messages


async def chat_complete(messages: list[dict], max_tokens: int | None = None) -> str:
    client = get_client()
    try:
        resp = await client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=messages,
            max_tokens=max_tokens or settings.OPENAI_MAX_TOKENS_CHAT,
            temperature=0.4,
        )
        text = (resp.choices[0].message.content or "").strip()
        if not text:
            raise ExternalServiceException("Model response is empty or malformed.")
        return text
    except RateLimitError as e:
        logger.warning("OpenAI rate limit: %s", e)
        raise ExternalServiceException("AI service is rate limited. Try again shortly.")
    except APIError as e:
        logger.exception("OpenAI API error")
        raise ExternalServiceException(f"AI request failed: {e}")
    except ExternalServiceException:
        raise
    except Exception as e:
        logger.exception("Unexpected OpenAI error")
        raise ExternalServiceException(f"AI request failed: {e}")


async def classify_task(prompt: str) -> str:
    """Single cheap classification call (replaces old double /chat call pattern)."""
    messages = [
        {
            "role": "system",
            "content": (
                "Classify the student's message into one short task_type slug "
                "(e.g. math_help, explanation, homework, general_chat). "
                "Reply with ONLY the slug."
            ),
        },
        {"role": "user", "content": prompt[:2000]},
    ]
    try:
        label = await chat_complete(messages, max_tokens=20)
        return label.split()[0].lower()[:50]
    except ExternalServiceException:
        return "general_chat"


async def summarize_text(text: str, max_tokens: int | None = None) -> list[str]:
    """Step 1 of flashcard gen: split long notes into summary points."""
    text = text.strip()
    if not text:
        return []
    # Chunk locally so one huge file doesn't blow the context window.
    chunk_size = 6000
    chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)][
        :10
    ]
    summaries: list[str] = []
    for chunk in chunks:
        messages = [
            {
                "role": "system",
                "content": (
                    "Summarize the study notes below into 3-6 concise bullet "
                    "points, one per line, no extra commentary."
                ),
            },
            {"role": "user", "content": chunk},
        ]
        summary = await chat_complete(
            messages, max_tokens=max_tokens or settings.OPENAI_MAX_TOKENS_SUMMARY
        )
        summaries.extend([s.strip("- ").strip() for s in summary.splitlines() if s.strip()])
    return summaries


async def generate_flashcards(summary_chunks: list[str]) -> list[dict]:
    """Step 2 of flashcard gen: summaries -> [{question, answer, summary}]."""
    if not summary_chunks:
        return []
    joined = "\n".join(f"- {s}" for s in summary_chunks[:25])
    messages = [
        {
            "role": "system",
            "content": (
                "Generate flashcards from the summaries. Reply with ONLY valid "
                'JSON: {"flashcards": [{"question": str, "answer": str, '
                '"summary": str}]}. Keep questions atomic, answers under 50 words.'
            ),
        },
        {"role": "user", "content": joined},
    ]
    client = get_client()
    try:
        resp = await client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.5,
            max_tokens=2000,
        )
        raw = resp.choices[0].message.content or "{}"
        data = json.loads(raw)
        cards = data.get("flashcards", [])
        return [
            {
                "question": str(c.get("question", "")).strip(),
                "answer": str(c.get("answer", "")).strip(),
                "summary": str(c.get("summary", "")).strip(),
            }
            for c in cards
            if c.get("question") and c.get("answer")
        ]
    except RateLimitError:
        raise ExternalServiceException("AI service is rate limited. Try again shortly.")
    except Exception as e:
        logger.exception("Flashcard generation failed")
        raise ExternalServiceException(f"Flashcard generation failed: {e}")


async def vision_extract_text(image_bytes: bytes) -> str:
    """Replaces OCR.space: GPT-4o-mini vision reads math/handwriting directly."""
    client = get_client()
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    try:
        resp = await client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Transcribe all text, math, and structure exactly. Reply with only the transcription.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                    ],
                }
            ],
            max_tokens=1500,
            temperature=0,
        )
        return (resp.choices[0].message.content or "").strip()
    except RateLimitError:
        raise ExternalServiceException("AI service is rate limited. Try again shortly.")
    except Exception as e:
        logger.exception("Vision OCR failed")
        raise ExternalServiceException(f"Image transcription failed: {e}")
