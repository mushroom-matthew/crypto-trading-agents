from tools.strategy_executor import (
    TradeSignal,
    compute_ema,
    compute_rsi,
    compute_sma,
    evaluate_signals,
)
from tools.strategy_spec import (
    EntryCondition,
    ExitCondition,
    PositionState,
    RiskSpec,
    StrategySpec,
)


def _build_spec() -> StrategySpec:
    return StrategySpec(
        strategy_id="spec-1",
        market="ETH-USD",
        timeframe="15m",
        mode="trend",
        entry_conditions=[
            EntryCondition(
                type="crossover", indicator="sma", lookback=3, direction="above"
            )
        ],
        exit_conditions=[
            ExitCondition(type="take_profit", take_profit_pct=5.0),
            ExitCondition(type="stop_loss", stop_loss_pct=2.0),
        ],
        risk=RiskSpec(
            max_fraction_of_balance=0.2,
            risk_per_trade_fraction=0.05,
            max_drawdown_pct=15.0,
            leverage=1.0,
        ),
    )


def test_indicator_helpers() -> None:
    series = [10, 11, 12, 13, 14]
    assert compute_sma(series, 3) == 13
    ema = compute_ema(series, 3)
    assert ema is not None and ema > 13
    rsi = compute_rsi([40, 41, 42, 45, 44, 46, 47, 49], 5)
    assert rsi is not None
    assert 0 <= rsi <= 100


def test_evaluate_signals_generates_entry() -> None:
    spec = _build_spec()
    features = {
        "close_prices": [100, 101, 102, 104],
        "price": 104,
        "indicators": {"sma": {3: 102}},
    }
    position = PositionState(market="ETH-USD")
    signals = evaluate_signals(spec, features, position)
    assert signals
    first = signals[0]
    assert first.side == "buy"
    assert first.take_profit and first.take_profit > features["price"]
    assert first.stop_loss and first.stop_loss < features["price"]


def test_evaluate_signals_generates_exit_on_take_profit() -> None:
    spec = _build_spec()
    features = {"price": 106, "close_prices": [100, 103, 106]}
    position = PositionState(
        market="ETH-USD", side="buy", qty=1, avg_entry_price=100, opened_ts=0
    )
    signals = evaluate_signals(spec, features, position)
    assert signals
    assert signals[0].side == "close"
    assert signals[0].reason == "exit:take_profit"
