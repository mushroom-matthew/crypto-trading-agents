from datetime import datetime, timedelta, timezone

from tools.strategy_spec import (
    EntryCondition,
    ExitCondition,
    PositionState,
    RiskSpec,
    StrategySpec,
    deserialize_strategy,
    serialize_strategy,
)


def _build_sample_spec() -> StrategySpec:
    return StrategySpec(
        strategy_id="sample-1",
        market="ETH-USD",
        timeframe="15m",
        mode="trend",
        entry_conditions=[
            EntryCondition(
                type="crossover",
                indicator="ema",
                lookback=20,
                direction="above",
                side="buy",
            )
        ],
        exit_conditions=[
            ExitCondition(type="take_profit", take_profit_pct=5.0),
            ExitCondition(type="stop_loss", stop_loss_pct=2.0),
        ],
        risk=RiskSpec(
            max_fraction_of_balance=0.3,
            risk_per_trade_fraction=0.01,
            max_drawdown_pct=20.0,
            leverage=1.0,
        ),
    )


def test_strategy_spec_round_trip_serialization() -> None:
    spec = _build_sample_spec()
    payload = serialize_strategy(spec)
    restored = deserialize_strategy(payload)
    assert restored.market == "ETH-USD"
    assert restored.entry_conditions[0].indicator == "ema"
    assert restored.risk.max_fraction_of_balance == 0.3


def test_strategy_spec_expiry_check() -> None:
    expires = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    spec = _build_sample_spec()
    spec.expiry_ts = expires
    assert spec.is_expired()


def test_position_state_helpers() -> None:
    state = PositionState(market="ETH-USD", side="buy", qty=1.0, avg_entry_price=2000)
    assert not state.is_flat()
    flat = PositionState(market="ETH-USD")
    assert flat.is_flat()
