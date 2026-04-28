from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from ops_api.routers import paper_trading
from schemas.session_intent import SessionIntent


class _FakeHandle:
    def __init__(self, *, status_name: str = "RUNNING", queries: dict | None = None) -> None:
        self._status_name = status_name
        self._queries = queries or {}
        self.signals: list[tuple[str, object]] = []

    async def describe(self):
        return SimpleNamespace(status=SimpleNamespace(name=self._status_name))

    async def query(self, name: str, *args):
        return self._queries[name]

    async def signal(self, name: str, payload=None):
        self.signals.append((name, payload))


class _FakeClient:
    def __init__(self) -> None:
        self.handles: dict[str, _FakeHandle] = {}
        self.start_calls: list[dict] = []

    def get_workflow_handle(self, workflow_id: str) -> _FakeHandle:
        return self.handles.setdefault(workflow_id, _FakeHandle())

    async def start_workflow(self, workflow, *args, **kwargs):
        self.start_calls.append({
            "workflow": workflow,
            "args": args,
            "kwargs": kwargs,
        })


@pytest.mark.asyncio
async def test_start_session_generates_session_intent_before_workflow_start(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    prompt_file = tmp_path / "strategist.txt"
    prompt_file.write_text("prompt-body")

    fake_client = _FakeClient()
    planner_calls: list[dict] = []

    async def _fake_get_temporal_client():
        return fake_client

    async def _fake_fetch_indicator_snapshots(symbols, timeframe, lookback_candles):
        assert symbols == ["BTC-USD", "ETH-USD"]
        assert timeframe == "1h"
        assert lookback_candles == 300
        return {
            "BTC-USD": {"close": 1.0},
            "ETH-USD": {"close": 2.0},
            "_raw_ohlcv_data": {"BTC-USD": {}, "ETH-USD": {}},
        }

    async def _fake_build_session_intent_from_indicator_snapshots(
        *,
        symbols,
        indicator_snapshots_raw,
        portfolio_state,
        session_config,
        llm_model,
    ):
        planner_calls.append({
            "symbols": symbols,
            "indicator_snapshots_raw": indicator_snapshots_raw,
            "portfolio_state": portfolio_state,
            "session_config": session_config,
            "llm_model": llm_model,
        })
        return SessionIntent(
            selected_symbols=["sol-usd"],
            symbol_intents=[],
            regime_summary="test regime",
            planner_rationale="test rationale",
            is_fallback=True,
        )

    monkeypatch.setattr("ops_api.temporal_client.get_temporal_client", _fake_get_temporal_client)
    monkeypatch.setattr(
        "tools.paper_trading.fetch_indicator_snapshots_activity",
        _fake_fetch_indicator_snapshots,
    )
    monkeypatch.setattr(
        "services.session_planner.build_session_intent_from_indicator_snapshots",
        _fake_build_session_intent_from_indicator_snapshots,
    )
    monkeypatch.setattr(
        "ops_api.routers.prompts.current_strategist_prompt_path",
        lambda: prompt_file,
    )

    response = await paper_trading.start_session(
        paper_trading.PaperTradingSessionConfig(
            symbols=["btc", "eth"],
            strategy_id="default",
            use_ai_planner=True,
            indicator_timeframe="1h",
            initial_cash=12_500,
            direction_bias="neutral",
            screener_regime="bull_trending",
            llm_model="planner-model",
        )
    )

    assert response.status == "running"
    assert len(planner_calls) == 1
    assert planner_calls[0]["symbols"] == ["BTC-USD", "ETH-USD"]
    assert planner_calls[0]["indicator_snapshots_raw"] == {
        "BTC-USD": {"close": 1.0},
        "ETH-USD": {"close": 2.0},
    }
    assert planner_calls[0]["portfolio_state"] == {
        "cash": 12_500.0,
        "positions": {},
        "equity": 12_500.0,
    }
    assert planner_calls[0]["session_config"] == {
        "indicator_timeframe": "1h",
        "screener_regime": "bull_trending",
        "direction_bias": "neutral",
    }
    assert planner_calls[0]["llm_model"] == "planner-model"

    assert len(fake_client.start_calls) == 1
    workflow_config = fake_client.start_calls[0]["kwargs"]["args"][0]
    assert workflow_config["strategy_prompt"] == "prompt-body"
    assert workflow_config["use_ai_planner"] is True
    assert workflow_config["symbols"] == ["SOL-USD"]
    assert workflow_config["session_intent"]["selected_symbols"] == ["SOL-USD"]


@pytest.mark.asyncio
async def test_refresh_session_intent_signals_workflow(monkeypatch: pytest.MonkeyPatch):
    session_id = "paper-trading-test"
    fake_client = _FakeClient()
    session_handle = _FakeHandle(
        status_name="RUNNING",
        queries={
            "get_session_status": {
                "symbols": ["BTC-USD"],
                "indicator_timeframe": "1h",
                "direction_bias": "long",
                "screener_regime": "bull_trending",
            }
        },
    )
    ledger_handle = _FakeHandle(
        queries={
            "get_portfolio_status": {
                "cash": 9500.0,
                "positions": {"BTC-USD": 0.25},
                "position_meta": {},
                "total_equity": 10_250.0,
            }
        },
    )
    fake_client.handles[session_id] = session_handle
    fake_client.handles[f"{session_id}-ledger"] = ledger_handle

    async def _fake_get_temporal_client():
        return fake_client

    async def _fake_fetch_indicator_snapshots(symbols, timeframe, lookback_candles):
        assert symbols == ["BTC-USD"]
        assert timeframe == "1h"
        assert lookback_candles == 300
        return {"BTC-USD": {"close": 1.0}, "_raw_ohlcv_data": {"BTC-USD": {}}}

    async def _fake_build_session_intent_from_indicator_snapshots(
        *,
        symbols,
        indicator_snapshots_raw,
        portfolio_state,
        session_config,
        llm_model,
    ):
        assert symbols == ["BTC-USD"]
        assert indicator_snapshots_raw == {"BTC-USD": {"close": 1.0}}
        assert portfolio_state == {
            "cash": 9500.0,
            "positions": {"BTC-USD": {"qty": 0.25, "side": "long"}},
            "equity": 10_250.0,
        }
        assert session_config == {
            "indicator_timeframe": "1h",
            "direction_bias": "long",
            "screener_regime": "bull_trending",
        }
        assert llm_model is None
        return SessionIntent(
            selected_symbols=["ETH-USD", "BTC-USD"],
            symbol_intents=[],
            regime_summary="rotating leadership",
            planner_rationale="refresh rationale",
        )

    monkeypatch.setattr("ops_api.temporal_client.get_temporal_client", _fake_get_temporal_client)
    monkeypatch.setattr(
        "tools.paper_trading.fetch_indicator_snapshots_activity",
        _fake_fetch_indicator_snapshots,
    )
    monkeypatch.setattr(
        "services.session_planner.build_session_intent_from_indicator_snapshots",
        _fake_build_session_intent_from_indicator_snapshots,
    )

    response = await paper_trading.refresh_session_intent(session_id)

    assert response["status"] == "session_intent_refreshed"
    assert response["symbols"] == ["ETH-USD", "BTC-USD"]
    assert session_handle.signals == [
        (
            "update_session_intent",
            response["session_intent"],
        )
    ]
