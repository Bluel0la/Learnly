"""
Centralized custom exceptions and global FastAPI exception handlers.

Usage:
    from api.core.exceptions import NotFoundException, ExternalServiceException
    raise NotFoundException("Quiz not found.")
    raise ExternalServiceException("LLM server unreachable.")

Register handlers in main.py:
    from api.core.exceptions import register_exception_handlers
    register_exception_handlers(app)
"""
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class AppException(Exception):
    """Base exception for the application."""

    def __init__(self, detail: str, status_code: int = 500):
        self.detail = detail
        self.status_code = status_code


class NotFoundException(AppException):
    def __init__(self, detail: str = "Resource not found."):
        super().__init__(detail=detail, status_code=404)


class ForbiddenException(AppException):
    def __init__(self, detail: str = "You do not have permission to perform this action."):
        super().__init__(detail=detail, status_code=403)


class BadRequestException(AppException):
    def __init__(self, detail: str = "Bad request."):
        super().__init__(detail=detail, status_code=400)


class UnauthorizedException(AppException):
    def __init__(self, detail: str = "Authentication required."):
        super().__init__(detail=detail, status_code=401)


class ExternalServiceException(AppException):
    """Raised when an external API (LLM, OCR, embedding, etc.) fails."""

    def __init__(self, detail: str = "External service unavailable."):
        super().__init__(detail=detail, status_code=503)


class RateLimitedException(AppException):
    def __init__(self, detail: str = "Rate limit exceeded. Please wait before trying again."):
        super().__init__(detail=detail, status_code=429)


def register_exception_handlers(app: FastAPI):
    """Register global handlers so all AppException subclasses return uniform JSON."""

    @app.exception_handler(AppException)
    async def app_exception_handler(request: Request, exc: AppException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"success": False, "detail": exc.detail},
        )
