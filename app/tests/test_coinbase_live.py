import os

import pytest

from app.coinbase import accounts
from app.coinbase.client import CoinbaseClient
from app.core.config import get_settings


@pytest.mark.asyncio
async def test_coinbase_credentials_and_balances():
    if os.getenv("RUN_COINBASE_LIVE_TEST") != "1":
        pytest.skip("Set RUN_COINBASE_LIVE_TEST=1 to run live Coinbase checks")

    missing = [key for key in ("COINBASE_API_KEY", "COINBASE_API_SECRET") if not os.getenv(key)]
    if missing:
        pytest.skip(f"Missing required Coinbase env vars: {', '.join(missing)}")

    settings = get_settings()
    async with CoinbaseClient(settings) as client:
        account_list = await accounts.list_accounts(client)
        assert account_list, "No Coinbase accounts returned; check credentials"

        summaries: list[dict[str, object]] = []
        populated = None
        for acct in account_list:
            balances = await accounts.get_balances(client, acct.uuid)
            summaries.append(
                {
                    "uuid": acct.uuid,
                    "name": acct.name,
                    "currency": acct.currency,
                    "available": str(acct.available_balance.value),
                    "hold": str(acct.hold.value if acct.hold else "0"),
                    "balance_entries": [f"{bal.currency}:{bal.value}" for bal in balances.balances],
                }
            )
            if acct.available_balance.value > 0:
                populated = acct
                break

        if populated is None:
            print("Coinbase account summaries:", summaries)
        assert populated is not None, "No Coinbase account reported a positive available balance; verify funding"
