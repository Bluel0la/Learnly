from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM — OpenAI only
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_TIMEOUT_SECONDS: float = 60.0
    OPENAI_MAX_TOKENS_SUMMARY: int = 512
    OPENAI_MAX_TOKENS_CHAT: int = 1024

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
