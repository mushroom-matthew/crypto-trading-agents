"""Maker/taker fee lookups with persistence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Literal, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.coinbase.client import CoinbaseClient
from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.models import FeesSnapshot
from app.db.repo import Database


LOG = get_logger(__name__)


@dataclass(slots=True)
class FeeRates:
    maker_rate: Decimal
    taker_rate: Decimal
    tier_name: str | None = None


class FeeService:
    """Fetch and cache Coinbase fee tiers."""

    def __init__(self, db: Database, client: CoinbaseClient, *, ttl: int = 3600) -> None:
        self._db = db
        self._client = client
        self._ttl = ttl
        self._settings = get_settings()

    async def lookup_current_rate(self, *, will_rest: bool) -> Decimal:
        """Return maker or taker rate depending on order behaviour."""

        snapshot = await self._ensure_snapshot()
        return snapshot.maker_rate if will_rest else snapshot.taker_rate

    async def _ensure_snapshot(self) -> FeeRates:
        async with self._db.session() as session:
            snapshot = await self._latest_snapshot(session)
            if snapshot and not self._is_expired(snapshot.fetched_at):
                return FeeRates(
                    maker_rate=snapshot.maker_rate,
                    taker_rate=snapshot.taker_rate,
                    tier_name=snapshot.tier_name,
                )

        # Refresh snapshot outside previous session to avoid nested transactions.
        refreshed = await self._fetch_remote()
        async with self._db.session() as session:
            session.add(
                FeesSnapshot(
                    portfolio_id=self._settings.coinbase_portfolio_id or "default",
                    maker_rate=refreshed.maker_rate,
                    taker_rate=refreshed.taker_rate,
                    tier_name=refreshed.tier_name,
                )
            )
            await session.flush()
        return refreshed

    async def _latest_snapshot(self, session: AsyncSession) -> Optional[FeesSnapshot]:
        stmt = (
            select(FeesSnapshot)
            .where(FeesSnapshot.portfolio_id == (self._settings.coinbase_portfolio_id or "default"))
            .order_by(FeesSnapshot.fetched_at.desc())
            .limit(1)
        )
        return await session.scalar(stmt)

    def _is_expired(self, fetched_at: datetime) -> bool:
        return fetched_at + timedelta(seconds=self._ttl) < datetime.now(timezone.utc)

    async def _fetch_remote(self) -> FeeRates:
        resp = await self._client.get("/api/v3/brokerage/transaction_summary")
        fee_tier = resp.get("fee_tier") or {}
        maker_rate = Decimal(fee_tier.get("maker_fee_rate", "0.0005"))
        taker_rate = Decimal(fee_tier.get("taker_fee_rate", "0.0006"))
        tier_name = fee_tier.get("pricing_tier")
        LOG.info("Fetched fee tier", maker=str(maker_rate), taker=str(taker_rate), tier=tier_name)
        return FeeRates(maker_rate=maker_rate, taker_rate=taker_rate, tier_name=tier_name)


__all__ = ["FeeService", "FeeRates"]
