from pathlib import Path

from services.strategy_config_store import StrategyConfigStore


def test_store_saves_and_loads(tmp_path: Path) -> None:
    store = StrategyConfigStore(tmp_path / "plans.json")
    plan = {"strategy": "demo", "param": 1}
    store.save("BTC-USD", plan)
    assert store.load("BTC-USD") == plan
    assert store.load("ETH-USD") is None
