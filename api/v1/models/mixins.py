"""
Reusable SQLAlchemy mixins for consistent model behavior.
"""
from sqlalchemy import Column, TIMESTAMP, Boolean
from sqlalchemy.sql import func


class TimestampMixin:
    """Adds standardized created_at and updated_at columns to any model."""

    created_at = Column(TIMESTAMP, server_default=func.now(), nullable=False)
    updated_at = Column(
        TIMESTAMP, server_default=func.now(), onupdate=func.now(), nullable=False
    )


class SoftDeleteMixin:
    """Adds a soft-delete flag so rows are never physically removed."""

    is_deleted = Column(Boolean, default=False, nullable=False)
