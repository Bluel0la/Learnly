"""
Rate limiting utility.

Provides a simple in-memory sliding-window rate limiter.

NOTE: This uses an in-memory dict, which means it resets on server restart
and is NOT shared across multiple Uvicorn workers. For production at scale,
replace `user_request_log` with a Redis-backed counter
(e.g. via `aioredis` or `redis-py`).
"""
from collections import defaultdict
from datetime import datetime, timedelta
from api.core.config import settings

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REQUEST_LIMIT = settings.RATE_LIMIT_REQUESTS
WINDOW_SECONDS = settings.RATE_LIMIT_WINDOW_SECONDS

# user_id -> list of request timestamps
user_request_log: dict[str, list[datetime]] = defaultdict(list)


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _prune_old_requests(user_id) -> None:
    """Remove timestamps outside the current sliding window."""
    now = datetime.utcnow()
    log = user_request_log[user_id]
    log[:] = [ts for ts in log if now - ts < timedelta(seconds=WINDOW_SECONDS)]
    if not log:
        del user_request_log[user_id]


def is_rate_limited(user_id) -> bool:
    """Return True if the user has exceeded the request limit within the window."""
    _prune_old_requests(user_id)
    log = user_request_log[user_id]
    if len(log) >= REQUEST_LIMIT:
        return True
    log.append(datetime.utcnow())
    return False


def reset_all_request_logs() -> None:
    """Periodic cleanup — call from a background task if desired."""
    now = datetime.utcnow()
    for uid in list(user_request_log.keys()):
        log = user_request_log[uid]
        log[:] = [ts for ts in log if now - ts < timedelta(seconds=WINDOW_SECONDS)]
        if not log:
            del user_request_log[uid]
