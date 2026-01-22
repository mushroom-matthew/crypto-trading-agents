import os
from pathlib import Path
from typing import AsyncIterator

import pytest
import pytest_asyncio

from app.core.config import Settings, get_settings
from agents.activities import ledger as ledger_activity
from app.db.models import Base
from app.db.repo import Database


@pytest.fixture(autouse=True)
def configure_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Settings:
    """Provide deterministic settings for tests."""

    live_flag = os.getenv("RUN_COINBASE_LIVE_TEST") == "1"

    if not live_flag:
        monkeypatch.setenv("COINBASE_API_KEY", "test-key")
        monkeypatch.setenv("COINBASE_API_SECRET", "test-secret")
        monkeypatch.setenv("COINBASE_PASSPHRASE", "test-pass")
        monkeypatch.setenv("COINBASE_BASE_URL", "https://example.com")
    else:
        if not os.getenv("COINBASE_API_KEY") and os.getenv("COINBASEEXCHANGE_API_KEY"):
            monkeypatch.setenv("COINBASE_API_KEY", os.environ["COINBASEEXCHANGE_API_KEY"])
        if not os.getenv("COINBASE_API_SECRET") and os.getenv("COINBASEEXCHANGE_SECRET"):
            monkeypatch.setenv("COINBASE_API_SECRET", os.environ["COINBASEEXCHANGE_SECRET"])
        for key in ("COINBASE_API_KEY", "COINBASE_API_SECRET"):
            if not os.getenv(key):
                pytest.skip(f"{key} is required for live Coinbase credential checks")

    db_path = tmp_path / "ledger.db"
    monkeypatch.setenv("DB_DSN", f"sqlite+aiosqlite:///{db_path}")
    monkeypatch.setenv("DEFAULT_SAFETY_BUFFER", "0.10")
    monkeypatch.setenv("HTTP_TIMEOUT_SECONDS", "1")
    monkeypatch.setenv("HTTP_RETRY_ATTEMPTS", "1")
    monkeypatch.setenv("HTTP_RETRY_BACKOFF_SECONDS", "0.1")
    monkeypatch.setenv("IDEMPOTENCY_TTL_SECONDS", "3600")
    monkeypatch.setenv("METRICS_NAMESPACE", "test")
    monkeypatch.setenv("PROMETHEUS_LISTEN_ADDR", "0.0.0.0")
    monkeypatch.setenv("PROMETHEUS_LISTEN_PORT", "9100")
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    monkeypatch.setenv("LEDGER_TRADING_WALLET_ID", "")
    monkeypatch.setenv("LEDGER_TRADING_WALLET_NAME", "mock_trading")
    monkeypatch.setenv("LEDGER_EQUITY_WALLET_NAME", "system_equity")
    ledger_activity._STATE = None  # type: ignore[attr-defined]

    get_settings.cache_clear()
    get_settings()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings() -> Settings:
    """Return settings configured for the test."""

    return get_settings()


@pytest_asyncio.fixture
async def db(settings: Settings) -> AsyncIterator[Database]:
    """Provision a fresh database per test."""

    database = Database(settings)
    async with database.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield database
    await database.dispose()
