"""
LLM service (OpenAI-compatible API — currently NVIDIA NIM, meta/muse-glimmer-30b).

Replaces:
- POST {MODEL_ENDPOINT}/chat            -> chat_complete()
- POST {MODEL_UTILITY}/relevant_turn     -> build_history_messages() (local, no HTTP)
- POST {MODEL_UTILITY}/flashcard-summarization -> summarize_text()
- POST {MODEL_ENDPOINT}/flashcard        -> generate_flashcards()
- POST https://api.ocr.space/parse/image -> vision_extract_text()

Notes on the current provider:
- Sampling follows the vendor recipe (temperature=1, top_p=0.95).
- `reasoning_effort` is sent opportunistically and stripped on rejection.
- `response_format=json_object` may not be honored — flashcard parsing falls
  back to extracting JSON from raw text.
- Non-stream calls get exactly one automatic retry on provider timeouts.

All functions raise ExternalServiceException(503) on failure so routes
don't need to know about the LLM SDK.
"""
import base64
import json
import logging
import re

from openai import AsyncOpenAI, APIError, APITimeoutError, RateLimitError

from api.core.config import settings
from api.core.exceptions import ExternalServiceException

logger = logging.getLogger(__name__)

_client: AsyncOpenAI | None = None


def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        if not settings.LLM_API_KEY:
            raise ExternalServiceException("LLM_API_KEY is not configured.")
        _client = AsyncOpenAI(
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_BASE_URL,
            timeout=settings.LLM_TIMEOUT_SECONDS,
        )
    return _client


def _sampling_kwargs() -> dict:
    return {"temperature": settings.LLM_TEMPERATURE, "top_p": settings.LLM_TOP_P}


async def _create_with_timeout_retry(create_kwargs: dict, *, stream: bool = False):
    """Single create() with exactly one retry on provider timeout.

    Streams fail fast (no retry — a stalled stream retry doubles worst case).
    """
    client = get_client()
    try:
        return await client.chat.completions.create(stream=stream, **create_kwargs)
    except APITimeoutError:
        if stream or not settings.LLM_RETRY_ON_TIMEOUT:
            raise
        logger.warning("LLM request timed out, retrying once")
        return await client.chat.completions.create(stream=stream, **create_kwargs)


def _strip_optional_kwargs(create_kwargs: dict) -> dict:
    """Minimal fallback payload if the provider rejects sampling/reasoning knobs."""
    return {
        k: v
        for k, v in create_kwargs.items()
        if k not in ("temperature", "top_p", "extra_body")
    }


def _rejects_optional_kwargs(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in ("reasoning_effort", "temperature", "top_p", "response_format"))


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


async def chat_complete(
    messages: list[dict],
    max_tokens: int | None = None,
    effort: str = "low",
) -> str:
    """Single chat completion. `effort` controls the reasoning budget
    ("low" for snappy chat, "medium" for harder generation tasks)."""
    create_kwargs = {
        "model": settings.LLM_MODEL_CHAT,
        "messages": messages,
        "max_tokens": max_tokens or settings.LLM_MAX_TOKENS_CHAT,
        **_sampling_kwargs(),
        "extra_body": {"reasoning_effort": effort},
    }
    try:
        try:
            resp = await _create_with_timeout_retry(create_kwargs)
        except APIError as e:
            # Provider rejected the optional knobs — retry minimal.
            if not _rejects_optional_kwargs(e):
                raise
            logger.info("optional kwargs unsupported, retrying minimal: %s", e)
            resp = await _create_with_timeout_retry(_strip_optional_kwargs(create_kwargs))
        text = (resp.choices[0].message.content or "").strip()
        if not text:
            raise ExternalServiceException("Model response is empty or malformed.")
        return text
    except RateLimitError as e:
        logger.warning("LLM rate limit: %s", e)
        raise ExternalServiceException("AI service is rate limited. Try again shortly.")
    except APIError as e:
        logger.exception("LLM API error")
        raise ExternalServiceException(f"AI request failed: {e}")
    except ExternalServiceException:
        raise
    except Exception as e:
        logger.exception("Unexpected LLM error")
        raise ExternalServiceException(f"AI request failed: {e}")


_GREETING_RE = re.compile(
    r"^(hi+|hello+|hey+|yo|sup|good\s?(morning|afternoon|evening|day)|"
    r"howdy|hiya|greetings|how\s?are\s?you|thanks?|thank\s?you|bye+|"
    r"goodbye|see\s?you|ok+|okay|sure|yes|no|yeah|nope)\W*$",
    re.IGNORECASE,
)


def is_trivial_message(prompt: str) -> bool:
    """Greetings/thanks/acks that need no classification call."""
    return len(prompt.strip()) <= 40 and bool(_GREETING_RE.match(prompt.strip()))


async def stream_chat_complete(messages: list[dict], max_tokens: int | None = None):
    """Yield content tokens as they arrive (SSE-friendly async generator).

    Yields raw text deltas. Raises ExternalServiceException on provider errors.
    Uses the fast chat model at low reasoning effort. Fails fast (no retry).
    """
    create_kwargs = {
        "model": settings.LLM_MODEL_CHAT,
        "messages": messages,
        "max_tokens": max_tokens or settings.LLM_MAX_TOKENS_CHAT,
        **_sampling_kwargs(),
        "extra_body": {"reasoning_effort": "low"},
    }
    try:
        try:
            stream = await _create_with_timeout_retry(create_kwargs, stream=True)
        except APIError as e:
            if not _rejects_optional_kwargs(e):
                raise
            logger.info("optional kwargs unsupported, retrying stream minimal: %s", e)
            stream = await _create_with_timeout_retry(
                _strip_optional_kwargs(create_kwargs), stream=True
            )
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
    except RateLimitError as e:
        logger.warning("LLM rate limit: %s", e)
        raise ExternalServiceException("AI service is rate limited. Try again shortly.")
    except APIError as e:
        logger.exception("LLM API error")
        raise ExternalServiceException(f"AI request failed: {e}")
    except ExternalServiceException:
        raise
    except Exception as e:
        logger.exception("Unexpected LLM error")
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
        # Generous budget: reasoning models think before answering.
        label = await chat_complete(messages, max_tokens=300, effort="low")
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
            messages, max_tokens=max_tokens or settings.LLM_MAX_TOKENS_SUMMARY, effort="low"
        )
        summaries.extend([s.strip("- ").strip() for s in summary.splitlines() if s.strip()])
    return summaries


def _extract_cards(raw: str) -> list[dict]:
    """Parse [{question, answer, summary}] from a response that may wrap JSON in prose."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return []
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = data.get("flashcards", [])
    else:
        return []
    cards = []
    for c in items:
        if not isinstance(c, dict):
            continue
        question = str(c.get("question", "")).strip()
        answer = str(c.get("answer", "")).strip()
        if not question or not answer:
            continue
        cards.append(
            {
                "question": question,
                "answer": answer,
                "summary": str(c.get("summary", "")).strip(),
            }
        )
    return cards


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
    create_kwargs = {
        "model": settings.LLM_MODEL_GEN,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "max_tokens": 4096,
        **_sampling_kwargs(),
        "extra_body": {"reasoning_effort": "medium"},
    }
    try:
        try:
            resp = await _create_with_timeout_retry(create_kwargs)
        except APIError as e:
            # Provider doesn't honor json_object (or the knobs) — retry minimal.
            if not _rejects_optional_kwargs(e):
                raise
            logger.info("json_object unsupported, retrying minimal plain text: %s", e)
            fallback = _strip_optional_kwargs(create_kwargs)
            fallback.pop("response_format", None)
            resp = await _create_with_timeout_retry(fallback)
        raw = resp.choices[0].message.content or "{}"
        cards = _extract_cards(raw)
        if not cards:
            logger.warning("Flashcard parse yielded no cards; raw=%.200s", raw)
        return cards
    except RateLimitError:
        raise ExternalServiceException("AI service is rate limited. Try again shortly.")
    except ExternalServiceException:
        raise
    except Exception as e:
        logger.exception("Flashcard generation failed")
        raise ExternalServiceException(f"Flashcard generation failed: {e}")


async def vision_extract_text(image_bytes: bytes) -> str:
    """Vision transcription: the model reads math/handwriting directly."""
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    create_kwargs = {
        "model": settings.LLM_MODEL_CHAT,
        "messages": [
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
        "max_tokens": 2048,
        **_sampling_kwargs(),
        "extra_body": {"reasoning_effort": "low"},
    }
    try:
        try:
            resp = await _create_with_timeout_retry(create_kwargs)
        except APIError as e:
            if not _rejects_optional_kwargs(e):
                raise
            logger.info("optional kwargs unsupported, retrying vision minimal: %s", e)
            resp = await _create_with_timeout_retry(_strip_optional_kwargs(create_kwargs))
        text = (resp.choices[0].message.content or "").strip()
        if not text:
            raise ExternalServiceException("Image transcription came back empty.")
        return text
    except RateLimitError:
        raise ExternalServiceException("AI service is rate limited. Try again shortly.")
    except ExternalServiceException:
        raise
    except Exception as e:
        logger.exception("Vision OCR failed")
        raise ExternalServiceException(f"Image transcription failed: {e}")
