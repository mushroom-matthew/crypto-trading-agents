"""Coinbase account and balance operations."""

from __future__ import annotations

from decimal import Decimal
from typing import List

from pydantic import BaseModel, ConfigDict, Field

from app.coinbase.client import CoinbaseClient


class Account(BaseModel):
    """Coinbase account metadata."""

    model_config = ConfigDict(extra="ignore")

    uuid: str = Field(..., alias="uuid")
    name: str | None = None
    currency: str
    available_balance: "MoneyAmount"
    hold: "MoneyAmount | None" = None
    type: str | None = Field(default=None, alias="type")
    portfolio_uuid: str | None = Field(default=None, alias="portfolio_uuid")


class MoneyAmount(BaseModel):
    """Coinbase money amount."""

    model_config = ConfigDict(extra="ignore")

    value: Decimal
    currency: str


class AccountBalances(BaseModel):
    """Aggregated balance details for an account."""

    model_config = ConfigDict(extra="ignore")

    account: Account
    balances: List[MoneyAmount]


async def list_accounts(client: CoinbaseClient) -> list[Account]:
    """Return Coinbase accounts accessible with the API key."""

    raw = await client.get("/api/v3/brokerage/accounts")
    accounts = raw.get("accounts", [])
    return [Account.model_validate(account) for account in accounts]


async def get_balances(client: CoinbaseClient, account_id: str) -> AccountBalances:
    """Return balances for a specific Coinbase account."""

    raw = await client.get(f"/api/v3/brokerage/accounts/{account_id}")
    account_payload = raw.get("account", raw)
    account = Account.model_validate(account_payload)
    balances_raw = raw.get("balances", [])
    balances = [
        MoneyAmount.model_validate(bal if isinstance(bal, dict) else {"currency": bal[0], "value": bal[1]})
        for bal in balances_raw
    ]
    return AccountBalances(account=account, balances=balances)


__all__ = ["Account", "MoneyAmount", "AccountBalances", "list_accounts", "get_balances"]
