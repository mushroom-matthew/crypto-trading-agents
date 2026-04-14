from __future__ import annotations

from types import SimpleNamespace

import pytest

from agents.activities import ledger as ledger_activities
from agents.workflows.execution_ledger_workflow import ExecutionLedgerWorkflow


def test_split_trading_pair_accepts_slash_and_dash_formats():
    assert ledger_activities._split_trading_pair("BTC/USD") == ("BTC", "USD")
    assert ledger_activities._split_trading_pair("BTC-USD") == ("BTC", "USD")


@pytest.mark.asyncio
async def test_persist_fill_activity_accepts_coinbase_dash_symbol(monkeypatch: pytest.MonkeyPatch):
    recorded_postings = []

    class DummyLedger:
        async def post_double_entry(self, postings):
            recorded_postings.extend(postings)

    async def fake_wallet(*args, **kwargs):
        return SimpleNamespace(wallet_id=kwargs.get("wallet_id", 1))

    monkeypatch.setattr(
        ledger_activities,
        "_get_state",
        lambda: (
            SimpleNamespace(
                ledger_trading_wallet_id=None,
                ledger_trading_wallet_name="mock_trading",
                ledger_equity_wallet_name="system_equity",
            ),
            None,
            DummyLedger(),
        ),
    )
    monkeypatch.setattr(ledger_activities, "_get_wallet_by_id_or_name", fake_wallet)
    monkeypatch.setattr(
        ledger_activities,
        "_ensure_equity_wallet",
        lambda name: fake_wallet(wallet_id=2),
    )

    await ledger_activities.persist_fill_activity(
        {
            "fill": {
                "symbol": "BTC-USD",
                "side": "BUY",
                "qty": 0.25,
                "fill_price": 40000.0,
                "cost": 10000.0,
            },
            "workflow_id": "paper-trading-ledger",
            "sequence": 1,
            "recorded_at": 0.0,
        }
    )

    assert len(recorded_postings) == 4
    assert {posting.currency for posting in recorded_postings} == {"BTC", "USD"}


@pytest.mark.asyncio
async def test_persist_fill_logs_plain_exception_message(monkeypatch: pytest.MonkeyPatch):
    workflow = ExecutionLedgerWorkflow()
    messages: list[str] = []

    async def fake_execute_activity(*args, **kwargs):
        raise RuntimeError("boom")

    class FakeLogger:
        def error(self, message, *args):
            messages.append(message % args if args else message)

    monkeypatch.setattr(
        "agents.workflows.execution_ledger_workflow.workflow.execute_activity",
        fake_execute_activity,
    )
    monkeypatch.setattr(
        "agents.workflows.execution_ledger_workflow.workflow.logger",
        FakeLogger(),
    )

    await workflow._persist_fill({"fill": {}})

    assert messages == ["Failed to persist fill to ledger: boom"]
