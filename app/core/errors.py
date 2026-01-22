"""Custom exception hierarchy for the Coinbase-enabled stack."""

from __future__ import annotations

from typing import Optional


class AppError(Exception):
    """Base error for the application."""


class ConfigurationError(AppError):
    """Raised when configuration is invalid or incomplete."""


class DatabaseError(AppError):
    """Raised for database connectivity or integrity violations."""

    def __init__(self, message: str, *, detail: Optional[str] = None) -> None:
        super().__init__(message)
        self.detail = detail


class CoinbaseAPIError(AppError):
    """Raised when Coinbase returns an error response."""

    def __init__(self, message: str, status_code: int, *, payload: Optional[dict] = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload or {}


class IdempotencyError(AppError):
    """Raised when idempotency keys are misused."""


class LedgerError(AppError):
    """Raised for double-entry ledger issues."""


__all__ = [
    "AppError",
    "ConfigurationError",
    "DatabaseError",
    "CoinbaseAPIError",
    "IdempotencyError",
    "LedgerError",
]
