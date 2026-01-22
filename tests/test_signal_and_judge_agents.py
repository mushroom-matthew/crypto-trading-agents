from trading_core.config import DEFAULT_STRATEGY_CONFIG, StrategyConfig, AssetConfig
from trading_core.signal_agent import MarketSnapshot, generate_intents
from trading_core.judge_agent import PortfolioState, evaluate_intents


def _mock_config() -> StrategyConfig:
    cfg = DEFAULT_STRATEGY_CONFIG
    cfg.assets = [
        AssetConfig(symbol="BTC-USD", leader_weight=0.7, follower_weight=0.3),
        AssetConfig(symbol="ETH-USD", leader_weight=0.3, follower_weight=0.7),
    ]
    return cfg


def test_generate_intents_breakout_entry():
    cfg = _mock_config()
    snapshots = {
        "BTC-USD": MarketSnapshot(
            symbol="BTC-USD",
            price=110,
            rolling_high=100,
            rolling_low=90,
            recent_max=108,
            atr=2,
            atr_band=3,
            volume_multiple=2,
            volume_floor=1,
            is_leader=True,
        )
    }
    intents = generate_intents(cfg, snapshots)
    assert intents[0].action == "BUY"
    assert intents[0].reason == "breakout_entry"


def test_generate_intents_exit_to_cash():
    cfg = _mock_config()
    snapshots = {
        "BTC-USD": MarketSnapshot(
            symbol="BTC-USD",
            price=95,
            rolling_high=110,
            rolling_low=90,
            recent_max=110,
            atr=5,
            atr_band=3,
            volume_multiple=0.5,
            volume_floor=1,
            is_leader=True,
            go_to_cash=True,
        )
    }
    intent = generate_intents(cfg, snapshots)[0]
    assert intent.action == "CLOSE"


def test_judge_blocks_when_drawdown_exceeded():
    cfg = _mock_config()
    cfg.risk.max_portfolio_drawdown_before_kill = 0.1
    portfolio = PortfolioState(cash=400, positions={}, equity=900, max_equity=1000)
    intent = MarketSnapshot(
        symbol="BTC-USD",
        price=110,
        rolling_high=100,
        rolling_low=90,
        recent_max=108,
        atr=2,
        atr_band=3,
        volume_multiple=2,
        volume_floor=1,
    )
    buy_intent = generate_intents(cfg, {"BTC-USD": intent})[0]
    judgements = evaluate_intents(cfg, portfolio, [buy_intent])
    assert not judgements[0].approved
    assert judgements[0].reason == "drawdown_halt"


def test_judge_allows_sell_when_position_exists():
    cfg = _mock_config()
    portfolio = PortfolioState(cash=1000, positions={"BTC-USD": 0.5}, equity=1200, max_equity=1300)
    snapshots = {
        "BTC-USD": MarketSnapshot(
            symbol="BTC-USD",
            price=90,
            rolling_high=110,
            rolling_low=85,
            recent_max=115,
            atr=4,
            atr_band=3,
            volume_multiple=1,
            volume_floor=1,
            is_leader=True,
        )
    }
    intent = generate_intents(cfg, snapshots)[0]
    assert intent.action == "SELL"
    judgement = evaluate_intents(cfg, portfolio, [intent])[0]
    assert judgement.approved
