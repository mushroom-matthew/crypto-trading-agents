from __future__ import annotations

from datetime import datetime, timedelta, timezone

from tools.paper_trading import PaperTradingConfig, SessionState, _normalize_live_plan


def _plan_dict(exit_rule: str) -> dict:
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return {
        "plan_id": "plan-live",
        "run_id": "paper-live",
        "generated_at": now.isoformat(),
        "valid_until": (now + timedelta(hours=4)).isoformat(),
        "global_view": "test",
        "regime": "range",
        "triggers": [
            {
                "id": "btc_probe",
                "symbol": "BTC-USD",
                "direction": "long",
                "timeframe": "5m",
                "entry_rule": "close > ema_short",
                "exit_rule": exit_rule,
                "category": "mean_reversion",
                "stop_loss_pct": 1.5,
            }
        ],
        "risk_constraints": {
            "max_position_risk_pct": 1.0,
            "max_symbol_exposure_pct": 25.0,
            "max_portfolio_exposure_pct": 80.0,
            "max_daily_loss_pct": 3.0,
        },
        "sizing_rules": [
            {
                "symbol": "BTC-USD",
                "sizing_mode": "fixed_fraction",
                "target_risk_pct": 1.0,
            }
        ],
        "max_trades_per_day": 5,
    }


def test_normalize_live_plan_sanitizes_position_age_exit_rule():
    plan = _plan_dict("not is_flat and (stop_hit or (position_age_minutes > 10))")

    normalized = _normalize_live_plan(plan, "5m")

    assert normalized["triggers"][0]["exit_rule"] == "not is_flat and (stop_hit or target_hit)"


def test_paper_trading_config_defaults_to_exact_exit_binding():
    config = PaperTradingConfig(
        session_id="paper-trading-test",
        symbols=["BTC-USD"],
    )
    assert config.exit_binding_mode == "exact"


def test_session_state_defaults_to_exact_exit_binding():
    state = SessionState(
        session_id="paper-trading-test",
        symbols=["BTC-USD"],
        strategy_prompt=None,
        plan_interval_hours=4.0,
    )
    assert state.exit_binding_mode == "exact"
