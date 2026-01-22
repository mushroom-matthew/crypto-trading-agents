"""Application configuration loaded from environment via pydantic-settings."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field, HttpUrl, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for Coinbase-enabled trading."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow",
    )

    log_level: Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "TRACE"] = "INFO"
    db_dsn: str = Field(..., alias="DB_DSN")

    coinbase_api_key: Optional[SecretStr] = Field(None, alias="COINBASE_API_KEY")
    coinbase_api_secret: Optional[SecretStr] = Field(None, alias="COINBASE_API_SECRET")
    coinbase_passphrase: Optional[SecretStr] = Field(None, alias="COINBASE_PASSPHRASE")
    coinbase_portfolio_id: Optional[str] = Field(None, alias="COINBASE_PORTFOLIO_ID")
    coinbase_base_url: HttpUrl = Field("https://api.exchange.coinbase.com", alias="COINBASE_BASE_URL")
    coinbase_wallet_secret: Optional[SecretStr] = Field(None, alias="COINBASE_WALLET_SECRET")

    default_safety_buffer: float = Field(0.10, alias="DEFAULT_SAFETY_BUFFER", ge=0.0, le=1.0)
    prometheus_listen_addr: str = Field("0.0.0.0", alias="PROMETHEUS_LISTEN_ADDR")
    prometheus_listen_port: int = Field(9090, alias="PROMETHEUS_LISTEN_PORT")

    idempotency_ttl_seconds: int = Field(3600, alias="IDEMPOTENCY_TTL_SECONDS", ge=60)
    http_timeout_seconds: float = Field(15.0, alias="HTTP_TIMEOUT_SECONDS", gt=0)
    http_retry_attempts: int = Field(5, alias="HTTP_RETRY_ATTEMPTS", ge=0)
    http_retry_backoff_seconds: float = Field(0.5, alias="HTTP_RETRY_BACKOFF_SECONDS", gt=0)

    metrics_namespace: str = Field("cta", alias="METRICS_NAMESPACE")
    ledger_trading_wallet_id: Optional[int] = Field(default=None, alias="LEDGER_TRADING_WALLET_ID")
    ledger_trading_wallet_name: str = Field(default="mock_trading", alias="LEDGER_TRADING_WALLET_NAME")
    ledger_equity_wallet_name: str = Field(default="system_equity", alias="LEDGER_EQUITY_WALLET_NAME")
    ledger_db_dsn: Optional[str] = Field(default=None, alias="LEDGER_DB_DSN")
    use_ledger_test_dsn: bool = Field(default=False, alias="USE_LEDGER_TEST_DSN")

    @field_validator("coinbase_api_secret", mode="before")
    @classmethod
    def _expand_secret_newlines(cls, value: Optional[str | SecretStr]) -> Optional[str | SecretStr]:
        """Allow PEM blocks stored with literal '\n' sequences in .env files."""
        if isinstance(value, SecretStr):
            value = value.get_secret_value()
        if isinstance(value, str):
            return value.replace("\\n", "\n")
        return value

    @field_validator("ledger_trading_wallet_id", mode="before")
    @classmethod
    def _normalize_ledger_wallet_id(cls, value: Optional[str | int]) -> Optional[int]:
        if value in (None, "", "null", "None"):
            return None
        if isinstance(value, int):
            return value
        return int(value)


@lru_cache(1)
def get_settings() -> Settings:
    """Return cached Settings instance."""

    return Settings()  # type: ignore[call-arg]


__all__ = ["Settings", "get_settings"]
