"""CLI entry point for running strategy backtests over historical data."""

from __future__ import annotations

import csv
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
from backtesting.logging_config import setup_backtest_logging
from schemas.strategy_run import RiskLimitSettings
from .llm_strategist_runner import LLMStrategistBacktester
from .simulator import run_backtest, run_portfolio_backtest
from .strategies import StrategyWrapperConfig, StrategyParameters


def _parse_session_multipliers(raw: str | None) -> list[dict[str, float]] | None:
    """Parse a comma-separated list of session windows into multiplier mappings."""

    if not raw:
        return None
    schedule: list[dict[str, float]] = []
    for chunk in raw.split(","):
        piece = chunk.strip()
        if not piece or ":" not in piece or "-" not in piece:
            continue
        window, mult_str = piece.split(":", 1)
        start_str, end_str = window.split("-", 1)
        try:
            start = int(start_str)
            end = int(end_str)
            multiplier = float(mult_str)
        except ValueError:
            continue
        schedule.append({"start_hour": start, "end_hour": end, "multiplier": multiplier})
    return schedule or None


def _emit_limit_debug(
    daily_reports: List[Dict[str, Any]],
    mode: str,
    output_dir: Path,
    run_id: str,
) -> None:
    if mode == "off":
        return
    if not daily_reports:
        print("\n(No daily limit stats available.)")
        return

    print("\n=== Limit Enforcement Stats ===")
    header = f"{'date':<12} {'attempted':>9} {'executed':>9} {'blocked_cap':>12} {'blocked_risk':>13} {'blocked_plan':>13}"
    print(header)
    print("-" * len(header))
    for report in daily_reports:
        stats = report.get("limit_stats") or {}
        attempted = report.get("attempted_triggers", stats.get("attempted_triggers", 0))
        executed = report.get("executed_trades", stats.get("executed_trades", 0))
        print(
            f"{report.get('date',''): <12}"
            f"{attempted:>9}"
            f"{executed:>9}"
            f"{stats.get('blocked_by_daily_cap', 0):>12}"
            f"{stats.get('blocked_by_risk_limits', 0):>13}"
            f"{stats.get('blocked_by_plan_limits', 0):>13}"
        )
    print("\n--- Trigger Stats ---")
    for report in daily_reports:
        stats = report.get("trigger_stats") or {}
        if not stats:
            continue
        print(f"{report.get('date','')}:")
        for trigger_id, payload in sorted(stats.items()):
            blocked_reasons = payload.get("blocked_by_reason") or {}
            print(
                f"  {trigger_id:<30} exec={payload.get('executed', 0):>3} blocked={payload.get('blocked', 0):>3} "
                f"cap={blocked_reasons.get('daily_cap', 0):>3} risk={blocked_reasons.get('risk_budget', 0):>3}"
            )

    if mode != "verbose":
        return
    target_dir = output_dir / run_id
    target_dir.mkdir(parents=True, exist_ok=True)
    csv_path = target_dir / "limits.csv"
    fieldnames = [
        "date",
        "timestamp",
        "symbol",
        "side",
        "price",
        "quantity",
        "timeframe",
        "trigger_id",
        "outcome",
        "reason",
        "detail",
        "source",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for report in daily_reports:
            date = report.get("date")
            stats = report.get("limit_stats") or {}
            for entry in stats.get("blocked_details", []):
                writer.writerow(
                    {
                        "date": date,
                        "timestamp": entry.get("timestamp"),
                        "symbol": entry.get("symbol"),
                        "side": entry.get("side"),
                        "price": entry.get("price"),
                        "quantity": entry.get("quantity"),
                        "timeframe": entry.get("timeframe"),
                        "trigger_id": entry.get("trigger_id"),
                        "outcome": "blocked",
                        "reason": entry.get("reason"),
                        "detail": entry.get("detail"),
                        "source": entry.get("source"),
                    }
                )
            for entry in stats.get("executed_details", []):
                writer.writerow(
                    {
                        "date": date,
                        "timestamp": entry.get("timestamp"),
                        "symbol": entry.get("symbol"),
                        "side": entry.get("side"),
                        "price": entry.get("price"),
                        "quantity": entry.get("quantity"),
                        "timeframe": entry.get("timeframe"),
                        "trigger_id": entry.get("trigger_id"),
                        "outcome": "executed",
                        "reason": "",
                        "detail": "",
                        "source": entry.get("source"),
                    }
                )
    print(f"Verbose limit log written to {csv_path}")
    trigger_path = target_dir / "trigger_stats.csv"
    trigger_fields = ["date", "trigger_id", "executed", "blocked", "blocked_by_daily_cap", "blocked_by_risk_budget", "blocked_by_risk"]
    with trigger_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=trigger_fields)
        writer.writeheader()
        for report in daily_reports:
            date = report.get("date")
            stats = report.get("trigger_stats") or {}
            for trigger_id, payload in stats.items():
                blocked_reasons = payload.get("blocked_by_reason") or {}
                writer.writerow(
                    {
                        "date": date,
                        "trigger_id": trigger_id,
                        "executed": payload.get("executed", 0),
                        "blocked": payload.get("blocked", 0),
                        "blocked_by_daily_cap": blocked_reasons.get("daily_cap", 0),
                        "blocked_by_risk_budget": blocked_reasons.get("risk_budget", 0),
                        "blocked_by_risk": blocked_reasons.get("risk", 0),
                    }
                )
    print(f"Verbose trigger stats written to {trigger_path}")


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
    parser.add_argument("--max-daily-risk-budget-pct", type=float, default=None, help="Optional cap on cumulative per-trade risk allocated each day")
    parser.add_argument("--risk-config", help="Optional JSON/YAML file containing risk limits (keys: max_position_risk_pct, etc.)")
    parser.add_argument("--timeframes", nargs="+", default=["1h", "4h", "1d"], help="Timeframe list for strategist indicators")
    parser.add_argument("--log-level", default="INFO", help="Backtest log level (DEBUG, INFO, etc.)")
    parser.add_argument("--log-file", help="Optional file path to append logs")
    parser.add_argument("--log-json", action="store_true", help="Emit JSON-formatted logs")
    parser.add_argument("--debug-limits", choices=["off", "basic", "verbose"], default="off", help="Enable limit-enforcement diagnostics")
    parser.add_argument("--debug-output-dir", default=".debug/backtests", help="Directory for verbose limit debug files")
    parser.add_argument("--flatten-daily", action="store_true", help="Flatten all open positions at the end of each trading day")
    parser.add_argument("--flatten-threshold", type=float, default=0.0, help="Only flatten positions with notional above this USD value")
    parser.add_argument("--flatten-session-hour", type=int, default=None, help="Optional UTC hour to flatten positions (e.g., 0 for midnight session close)")
    parser.add_argument(
        "--session-trade-multipliers",
        help="Optional session cap schedule, e.g., '0-4:1.5,4-24:0.75' to raise caps early UTC and throttle later hours",
    )
    args = parser.parse_args()

    setup_backtest_logging(level=args.log_level, log_file=args.log_file, json_logs=args.log_json)

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
            "max_daily_risk_budget_pct": args.max_daily_risk_budget_pct,
        }
        risk_limits: RiskLimitSettings = resolve_risk_limits(
            Path(args.risk_config) if args.risk_config else None,
            cli_risk_overrides,
        )
        risk_params = risk_limits.to_risk_params()
        pairs = args.pairs or [args.pair]
        prompt_path = Path(args.llm_prompt) if args.llm_prompt else None
        session_multipliers = _parse_session_multipliers(args.session_trade_multipliers)
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
            flatten_positions_daily=args.flatten_daily,
            flatten_notional_threshold=args.flatten_threshold,
            flatten_session_boundary_hour=args.flatten_session_hour,
            session_trade_multipliers=session_multipliers,
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
        if args.debug_limits != "off":
            _emit_limit_debug(result.daily_reports, args.debug_limits, Path(args.debug_output_dir), args.llm_run_id)
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
            flatten_positions_daily=args.flatten_daily,
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
            flatten_positions_daily=args.flatten_daily,
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
