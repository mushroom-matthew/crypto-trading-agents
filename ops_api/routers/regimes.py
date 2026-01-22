"""API router for market regime reference data."""

from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter
from pydantic import BaseModel

from backtesting.regimes import REGIMES, REGIME_METADATA

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/regimes", tags=["regimes"])


class RegimeInfo(BaseModel):
    """Information about a market regime period."""

    id: str
    name: str
    description: str
    character: str  # "bull", "bear", "volatile", "ranging"
    start_date: str
    end_date: str


class RegimesListResponse(BaseModel):
    """Response model for listing available regimes."""

    regimes: List[RegimeInfo]


@router.get("/", response_model=RegimesListResponse)
async def list_regimes() -> RegimesListResponse:
    """List all available market regime periods for backtesting.

    Returns regime IDs, human-readable names, descriptions,
    character (bull/bear/volatile/ranging), and date ranges.
    """
    regimes = []
    for regime_id, (start_date, end_date) in REGIMES.items():
        meta = REGIME_METADATA.get(regime_id, {})
        regimes.append(
            RegimeInfo(
                id=regime_id,
                name=meta.get("name", regime_id.replace("_", " ").title()),
                description=meta.get("description", f"Market regime: {regime_id}"),
                character=meta.get("character", "unknown"),
                start_date=start_date,
                end_date=end_date,
            )
        )

    # Sort by start date for chronological ordering
    regimes.sort(key=lambda r: r.start_date)

    return RegimesListResponse(regimes=regimes)
