"""Database session management and repository helpers."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator, Awaitable, Callable, TypeVar

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import Settings, get_settings
from app.core.errors import DatabaseError
from app.core.logging import get_logger


LOG = get_logger(__name__)
T = TypeVar("T")


def _build_engine(settings: Settings) -> AsyncEngine:
    """Create an AsyncEngine from configuration."""

    connect_args: dict[str, object] = {}
    if settings.db_dsn.startswith("postgresql"):
        connect_args["server_settings"] = {"application_name": "cta-ledger"}

    return create_async_engine(
        settings.db_dsn,
        pool_pre_ping=True,
        connect_args=connect_args,
    )


class Database:
    """Wrapper around SQLAlchemy async engine providing session helpers."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._engine = _build_engine(self._settings)
        self._session_factory = async_sessionmaker(self._engine, expire_on_commit=False)

    @property
    def engine(self) -> AsyncEngine:
        return self._engine

    async def dispose(self) -> None:
        """Cleanly dispose the engine."""

        await self._engine.dispose()

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Yield a new AsyncSession with automatic rollback on errors."""

        session = self._session_factory()
        try:
            yield session
            await session.commit()
        except Exception:  # pragma: no cover - difficult to unit test rollback path
            await session.rollback()
            raise
        finally:
            await session.close()

    async def run_in_transaction(self, fn: Callable[[AsyncSession], Awaitable[T]]) -> T:
        """Execute an async callable within a managed transaction."""

        async with self.session() as session:
            return await fn(session)


__all__ = ["Database"]
