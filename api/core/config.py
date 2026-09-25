from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM — OpenAI-compatible API (currently NVIDIA NIM)
    LLM_API_KEY: str = ""
    # Interactive paths (chat, classify, vision): point at a FAST model.
    LLM_MODEL_CHAT: str = "meta/muse-glimmer-30b"
    # Heavy generation (summaries, flashcards).
    LLM_MODEL_GEN: str = "meta/muse-glimmer-30b"
    LLM_BASE_URL: str = "https://integrate.api.nvidia.com/v1"
    LLM_TIMEOUT_SECONDS: float = 300.0
    LLM_MAX_TOKENS_SUMMARY: int = 2048
    LLM_MAX_TOKENS_CHAT: int = 4096
    # Sampling per vendor recipe for the chat model.
    LLM_TEMPERATURE: float = 1.0
    LLM_TOP_P: float = 0.95
    # Single automatic retry on provider timeouts (non-stream calls only).
    LLM_RETRY_ON_TIMEOUT: bool = True

    # Security & Auth
    SECRET_KEY: str = ""
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440

    # Database
    DB_URL: str = ""

    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = 5
    RATE_LIMIT_WINDOW_SECONDS: int = 3600

    # Quiz Difficulty Thresholds (0-100 scale)
    QUIZ_PRO_THRESHOLD: float = 85.0
    QUIZ_MEDIUM_THRESHOLD: float = 60.0

    # Quiz Randomization Weights
    WEIGHTS_FROM_MEDIUM: list[float] = [0.2, 0.6, 0.2]
    WEIGHTS_FROM_PRO: list[float] = [0.3, 0.7]  # [medium, pro]
    WEIGHTS_FROM_EASY: list[float] = [0.7, 0.3]  # [easy, medium]


settings = Settings()
