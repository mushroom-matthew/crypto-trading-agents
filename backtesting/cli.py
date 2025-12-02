"""CLI entry point for running strategy backtests over historical data."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

if __package__ is None:  # allow running as `python backtesting/cli.py`
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from services.strategy_config_store import load_all_plans
from agents.strategies.llm_client import LLMClient
from backtesting.risk_config import resolve_risk_limits
from schemas.strategy_run import RiskLimitSettings
from .llm_strategist_runner import LLMStrategistBacktester
from .simulator import run_backtest, run_portfolio_backtest
from .strategies import StrategyWrapperConfig, StrategyParameters


def main() -> None:
    parser = argparse.ArgumentParser(description="Run execution-agent backtest")
    parser.add_argument("--pair", default="BTC-USD")
    parser.add_argument("--pairs", nargs="+", help="List of symbols to backtest as independent sleeves")
    parser.add_argument("--weights", nargs="+", type=float, help="Weights corresponding to --pairs")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--initial-cash", type=float, default=1400.0)
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--atr-mult", type=float, default=1.5)
    parser.add_argument("--volume-floor", type=float, default=1.0)
    parser.add_argument("--go-to-cash", action="store_true")
    parser.add_argument("--plan-file", help="Path to JSON file with planner plans")
    parser.add_argument("--use-saved-plans", action="store_true", help="Load plan metadata from data/strategy_configs.json")
    parser.add_argument("--llm-strategist", choices=["enabled", "disabled"], default="disabled", help="Enable the LLM strategist workflow")
    parser.add_argument("--llm-calls-per-day", type=int, default=8)
    parser.add_argument("--llm-cache-dir", default=".cache/strategy_plans")
    parser.add_argument("--llm-run-id", default="default")
    parser.add_argument("--llm-prompt", help="Optional prompt template path for the strategist")
    parser.add_argument("--max-position-risk-pct", type=float, default=None, help="Override per-trade risk (% of equity)")
    parser.add_argument("--max-symbol-exposure-pct", type=float, default=None, help="Override max exposure per symbol (% of equity)")
    parser.add_argument("--max-portfolio-exposure-pct", type=float, default=None, help="Override gross portfolio exposure (% of equity)")
    parser.add_argument("--max-daily-loss-pct", type=float, default=None, help="Override daily loss cap (% drawdown from daily anchor)")
    parser.add_argument("--risk-config", help="Optional JSON/YAML file containing risk limits (keys: max_position_risk_pct, etc.)")
    parser.add_argument("--timeframes", nargs="+", default=["1h", "4h", "1d"], help="Timeframe list for strategist indicators")
    args = parser.parse_args()

    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)

    plan_map: Dict[str, Dict[str, Any]] = {}
    if args.use_saved_plans:
        plan_map.update(load_all_plans())
    if args.plan_file:
        with open(args.plan_file, "r", encoding="utf-8") as f:
            file_plans = json.load(f)
            if isinstance(file_plans, dict):
                plan_map.update(file_plans)

    def plan_to_parameters(symbol: str) -> Optional[StrategyParameters]:
        entry = plan_map.get(symbol)
        if not entry:
            return None
        metadata = entry.get("metadata", {})
        volatility = metadata.get("volatility", {})
        volume_meta = metadata.get("volume", {})
        atr_mult = volatility.get("atr_band_mult", args.atr_mult)
        vol_floor = volume_meta.get("volume_floor", args.volume_floor)
        go_to_cash_val = metadata.get("allow_go_to_cash", args.go_to_cash)
        return StrategyParameters(atr_band_mult=atr_mult, volume_floor=vol_floor, go_to_cash=go_to_cash_val)

    strategy_cfg = StrategyWrapperConfig(
        parameters=StrategyParameters(
            atr_band_mult=args.atr_mult,
            volume_floor=args.volume_floor,
            go_to_cash=args.go_to_cash,
        )
    )
    if plan_map:
        per_symbol_params = {}
        for symbol in (args.pairs or [args.pair]):
            params = plan_to_parameters(symbol)
            if params:
                per_symbol_params[symbol] = params
        strategy_cfg.per_symbol_parameters = per_symbol_params

    if args.llm_strategist == "enabled":
        cli_risk_overrides = {
            "max_position_risk_pct": args.max_position_risk_pct,
            "max_symbol_exposure_pct": args.max_symbol_exposure_pct,
            "max_portfolio_exposure_pct": args.max_portfolio_exposure_pct,
            "max_daily_loss_pct": args.max_daily_loss_pct,
        }
        risk_limits: RiskLimitSettings = resolve_risk_limits(
            Path(args.risk_config) if args.risk_config else None,
            cli_risk_overrides,
        )
        risk_params = risk_limits.to_risk_params()
        pairs = args.pairs or [args.pair]
        prompt_path = Path(args.llm_prompt) if args.llm_prompt else None
        backtester = LLMStrategistBacktester(
            pairs=pairs,
            start=start,
            end=end,
            initial_cash=args.initial_cash,
            fee_rate=args.fee_rate,
            llm_client=LLMClient(),
            cache_dir=Path(args.llm_cache_dir),
            llm_calls_per_day=args.llm_calls_per_day,
            risk_params=risk_params,
            prompt_template_path=prompt_path,
            timeframes=args.timeframes,
        )
        result = backtester.run(run_id=args.llm_run_id)
        print("=== LLM Strategist Summary ===")
        for key, value in result.summary.items():
            print(f"{key}: {value}")
        print("\nLLM cost estimates:", result.llm_costs)
        if result.plan_log:
            print("\nPlan log (first 5 entries):")
            for entry in result.plan_log[:5]:
                print(entry)
        if not result.fills.empty:
            print("\nSample fills:")
            print(result.fills.tail())
        return

    if args.pairs:
        weights: Optional[List[float]] = args.weights
        result = run_portfolio_backtest(
            pairs=args.pairs,
            start=start,
            end=end,
            initial_cash=args.initial_cash,
            fee_rate=args.fee_rate,
            strategy_config=strategy_cfg,
            weights=weights,
        )
        print("=== Portfolio Summary ===")
        for key, value in result.summary.items():
            print(f"{key}: {value}")
        print("\n=== Per Pair Summaries ===")
        for pair, pair_result in result.per_pair.items():
            print(f"{pair}: return={pair_result.summary['return_pct']:.2f}% sharpe={pair_result.summary.get('sharpe_ratio', 0):.2f}")
        print("\nLast 5 aggregated equity points:")
        print(result.equity_curve.tail())
    else:
        result = run_backtest(
            pair=args.pair,
            start=start,
            end=end,
            initial_cash=args.initial_cash,
            fee_rate=args.fee_rate,
            strategy_config=strategy_cfg,
        )
        print("=== Summary ===")
        for key, value in result.summary.items():
            print(f"{key}: {value}")
        print("\nLast 5 equity points:")
        print(result.equity_curve.tail())
        if not result.trades.empty:
            print("\nSample trades:")
            print(result.trades.tail())


if __name__ == "__main__":
    main()
