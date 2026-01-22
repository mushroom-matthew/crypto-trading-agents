"""Coinbase transfer helpers."""

from __future__ import annotations

from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from app.coinbase.client import CoinbaseClient
from app.core.idempotency import IdempotencyKey


class TransferStatus(BaseModel):
    """Transfer status payload."""

    model_config = ConfigDict(extra="ignore")

    transfer_id: str = Field(..., alias="id")
    amount: Decimal
    currency: str
    status: str
    completed_at: Optional[str] = Field(default=None, alias="completed_at")


class TransferResponse(BaseModel):
    """Transfer initiation response."""

    model_config = ConfigDict(extra="ignore")

    transfer: TransferStatus


async def create_internal(
    client: CoinbaseClient,
    *,
    from_account: str,
    to_account: str,
    currency: str,
    amount: Decimal,
    idempotency_key: IdempotencyKey,
) -> TransferStatus:
    """Initiate an internal Coinbase transfer."""

    payload = {
        "from_account": from_account,
        "to_account": to_account,
        "currency": currency,
        "amount": str(amount),
    }
    raw = await client.post(
        "/api/v3/brokerage/transfer",
        json_payload=payload,
        idempotency_key=idempotency_key,
    )
    return TransferStatus.model_validate(raw.get("transfer", raw))


async def poll_status(client: CoinbaseClient, transfer_id: str) -> TransferStatus:
    """Poll Coinbase for transfer status updates."""

    raw = await client.get(f"/api/v3/brokerage/transfer/{transfer_id}")
    return TransferStatus.model_validate(raw.get("transfer", raw))


__all__ = ["TransferStatus", "create_internal", "poll_status"]
