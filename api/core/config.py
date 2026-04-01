"""
Centralized Configuration Settings.

This module loads environment variables and defines application-wide 
constants, preventing magic numbers from being scattered across the codebase.
"""
import os
import ast
from dotenv import load_dotenv

load_dotenv(".env")

class Settings:
    # LLM & External Services
    MODEL_ENDPOINT: str = os.getenv("MODEL_ENDPOINT", "")
    MODEL_UTILITY: str = os.getenv("MODEL_UTILITY", "")
    OCR_API_KEY: str = os.getenv("OCR_API", "")

    # Security & Auth
    SECRET_KEY: str = os.getenv("SECRET_KEY", "")
    ALGORITHM: str = os.getenv("ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))

    # Rate Limiting
    # (Fallbacks provided if not found in .env)
    RATE_LIMIT_REQUESTS: int = int(os.getenv("RATE_LIMIT_REQUESTS", "5"))
    RATE_LIMIT_WINDOW_SECONDS: int = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "3600"))

    # Quiz Difficulty Thresholds (0-100 scale)
    QUIZ_PRO_THRESHOLD: float = float(os.getenv("QUIZ_PRO_THRESHOLD", "85.0"))
    QUIZ_MEDIUM_THRESHOLD: float = float(os.getenv("QUIZ_MEDIUM_THRESHOLD", "60.0"))

    # Quiz Randomization Weights
    # Defining weights for [easy, medium, pro] depending on current base level
    # Stored in env as list strings: "[0.2, 0.6, 0.2]"
    WEIGHTS_FROM_MEDIUM: list[float] = ast.literal_eval(
        os.getenv("WEIGHTS_FROM_MEDIUM", "[0.2, 0.6, 0.2]")
    )
    WEIGHTS_FROM_PRO: list[float] = ast.literal_eval(
        os.getenv("WEIGHTS_FROM_PRO", "[0.3, 0.7]")  # [medium, pro]
    )
    WEIGHTS_FROM_EASY: list[float] = ast.literal_eval(
        os.getenv("WEIGHTS_FROM_EASY", "[0.7, 0.3]")  # [easy, medium]
    )


settings = Settings()
