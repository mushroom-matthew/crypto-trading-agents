"""Run a suite of backtests defined in a JSON configuration."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from .simulator import run_backtest, run_portfolio_backtest
from .strategies import StrategyWrapperConfig, StrategyParameters


def _parse_date(value: str) -> datetime:
    return datetime.fromisoformat(value)


def run_case(defaults: Dict[str, Any], case: Dict[str, Any]) -> Dict[str, Any]:
    start = _parse_date(case.get("start", defaults["start"]))
    end = _parse_date(case.get("end", defaults["end"]))
    fee_rate = case.get("fee_rate", defaults["fee_rate"])
    initial_cash = case.get("initial_cash", defaults["initial_cash"])
    params = StrategyParameters(
        atr_band_mult=case.get("atr_mult", defaults.get("atr_mult", 1.5)),
        volume_floor=case.get("volume_floor", defaults.get("volume_floor", 1.0)),
        go_to_cash=case.get("go_to_cash", defaults.get("go_to_cash", False)),
    )
    strategy_cfg = StrategyWrapperConfig(parameters=params)
    if "pairs" in case:
        result = run_portfolio_backtest(
            pairs=case["pairs"],
            start=start,
            end=end,
            initial_cash=initial_cash,
            fee_rate=fee_rate,
            strategy_config=strategy_cfg,
            weights=case.get("weights"),
        )
        summary = result.summary
    else:
        result = run_backtest(
            pair=case["pair"],
            start=start,
            end=end,
            initial_cash=initial_cash,
            fee_rate=fee_rate,
            strategy_config=strategy_cfg,
        )
        summary = result.summary
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run batch of backtests from JSON config")
    parser.add_argument("--config", required=True, help="Path to JSON file describing test suite")
    args = parser.parse_args()

    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    defaults = {
        "start": config["start"],
        "end": config["end"],
        "fee_rate": config.get("fee_rate", 0.001),
        "initial_cash": config.get("initial_cash", 1400.0),
        "atr_mult": config.get("atr_mult", 1.5),
        "volume_floor": config.get("volume_floor", 1.0),
        "go_to_cash": config.get("go_to_cash", False),
    }

    print(f"Backtest suite: {config_path}")
    headers = ["name", "return_pct", "sharpe_ratio", "max_drawdown_pct", "win_rate", "profit_factor"]
    print(" | ".join(f"{h:>18}" for h in headers))
    print("-" * (22 * len(headers)))

    for case in config["cases"]:
        summary = run_case(defaults, case)
        print(
            f"{case.get('name', case.get('pair', '+'.join(case.get('pairs', [])))):>18} | "
            f"{summary.get('return_pct', 0):>18.2f} | "
            f"{summary.get('sharpe_ratio', 0):>18.2f} | "
            f"{summary.get('max_drawdown_pct', 0):>18.2f} | "
            f"{summary.get('win_rate', 0):>18.2f} | "
            f"{summary.get('profit_factor', 0):>18.2f}"
        )


if __name__ == "__main__":
    main()
