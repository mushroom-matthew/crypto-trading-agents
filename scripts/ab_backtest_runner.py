"""Helper to run A/B backtest matrices across risk, flatten, and cadence knobs."""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Sequence


BASE_CMD = ["uv", "run", "python", "-m", "backtesting.cli"]


REGIMES: Dict[str, tuple[str, str]] = {
    "bull_2020_2021": ("2020-10-01", "2021-04-14"),
    "late_bull_2021": ("2021-07-20", "2021-11-10"),
    "bear_2018": ("2018-01-01", "2018-12-15"),
    "covid_crash": ("2020-02-20", "2020-03-15"),
    "bear_2022": ("2022-04-01", "2022-11-10"),
    "range_2019": ("2019-01-01", "2019-10-01"),
    "mid_2021_consolidation": ("2021-05-01", "2021-07-20"),
    "late_2023_consolidation": ("2023-06-01", "2023-10-01"),
    "vol_china_elon": ("2021-05-01", "2021-06-01"),
    "ftx_collapse": ("2022-11-01", "2022-11-20"),
}


def _scenario(
    key: str,
    desc: str,
    start: str,
    end: str,
    extra_args: Sequence[str],
) -> dict:
    return {
        "key": key,
        "desc": desc,
        "start": start,
        "end": end,
        "args": list(extra_args),
    }


SCENARIOS: Dict[str, dict] = {
    # Phase 0: mechanical baseline (no LLM)
    "P0": _scenario(
        "P0",
        "Baseline mechanical run (LLM disabled, saved plan)",
        "2021-01-01",
        "2021-01-14",
        [
            "--pair",
            "BTC-USD",
            "--initial-cash",
            "1400",
            "--fee-rate",
            "0.001",
            "--llm-strategist",
            "disabled",
            "--use-saved-plans",
            "--max-position-risk-pct",
            "0.5",
            "--max-daily-risk-budget-pct",
            "2.0",
            "--max-daily-loss-pct",
            "3.0",
            "--flatten-policy",
            "daily_close",
            "--flatten-session-hour",
            "0",
            "--debug-limits",
            "basic",
        ],
    ),
    # Phase 1: daily budget sweep
    "P1A": _scenario(
        "P1A",
        "LLM on, tight daily budget 1.0%",
        "2021-02-01",
        "2021-02-14",
        [
            "--pair",
            "BTC-USD",
            "--initial-cash",
            "1400",
            "--fee-rate",
            "0.001",
            "--llm-strategist",
            "enabled",
            "--llm-calls-per-day",
            "1",
            "--max-position-risk-pct",
            "0.5",
            "--max-daily-risk-budget-pct",
            "1.0",
            "--max-daily-loss-pct",
            "3.0",
            "--flatten-policy",
            "daily_close",
            "--debug-limits",
            "basic",
        ],
    ),
    "P1B": _scenario(
        "P1B",
        "LLM on, normal daily budget 3.75%",
        "2021-02-01",
        "2021-02-14",
        [
            "--pair",
            "BTC-USD",
            "--initial-cash",
            "1400",
            "--fee-rate",
            "0.001",
            "--llm-strategist",
            "enabled",
            "--llm-calls-per-day",
            "1",
            "--max-position-risk-pct",
            "0.5",
            "--max-daily-risk-budget-pct",
            "3.75",
            "--max-daily-loss-pct",
            "3.0",
            "--flatten-policy",
            "daily_close",
            "--debug-limits",
            "basic",
        ],
    ),
    "P1C": _scenario(
        "P1C",
        "LLM on, loose daily budget 7.5%",
        "2021-02-01",
        "2021-02-14",
        [
            "--pair",
            "BTC-USD",
            "--initial-cash",
            "1400",
            "--fee-rate",
            "0.001",
            "--llm-strategist",
            "enabled",
            "--llm-calls-per-day",
            "1",
            "--max-position-risk-pct",
            "0.5",
            "--max-daily-risk-budget-pct",
            "7.5",
            "--max-daily-loss-pct",
            "3.0",
            "--flatten-policy",
            "daily_close",
            "--debug-limits",
            "basic",
        ],
    ),
    # Phase 2: per-trade risk sweep (daily budget fixed at 3.75%)
    "P2D": _scenario(
        "P2D",
        "Tiny per-trade risk 0.25% with 3.75% daily budget",
        "2021-02-01",
        "2021-02-14",
        [
            "--pair",
            "BTC-USD",
            "--initial-cash",
            "1400",
            "--fee-rate",
            "0.001",
            "--llm-strategist",
            "enabled",
            "--llm-calls-per-day",
            "1",
            "--max-position-risk-pct",
            "0.25",
            "--max-daily-risk-budget-pct",
            "3.75",
            "--max-daily-loss-pct",
            "3.0",
            "--flatten-policy",
            "daily_close",
            "--debug-limits",
            "basic",
        ],
    ),
    "P2E": _scenario(
        "P2E",
        "Medium per-trade risk 0.75% with 3.75% daily budget",
        "2021-02-01",
        "2021-02-14",
        [
            "--pair",
            "BTC-USD",
            "--initial-cash",
            "1400",
            "--fee-rate",
            "0.001",
            "--llm-strategist",
            "enabled",
            "--llm-calls-per-day",
            "1",
            "--max-position-risk-pct",
            "0.75",
            "--max-daily-risk-budget-pct",
            "3.75",
            "--max-daily-loss-pct",
            "3.0",
            "--flatten-policy",
            "daily_close",
            "--debug-limits",
            "basic",
        ],
    ),
    "P2F": _scenario(
        "P2F",
        "Aggressive per-trade risk 1.5% with 3.75% daily budget",
        "2021-02-01",
        "2021-02-14",
        [
            "--pair",
            "BTC-USD",
            "--initial-cash",
            "1400",
            "--fee-rate",
            "0.001",
            "--llm-strategist",
            "enabled",
            "--llm-calls-per-day",
            "1",
            "--max-position-risk-pct",
            "1.5",
            "--max-daily-risk-budget-pct",
            "3.75",
            "--max-daily-loss-pct",
            "3.0",
            "--flatten-policy",
            "daily_close",
            "--debug-limits",
            "basic",
        ],
    ),
    # Phase 3: daily loss cap stress
    "P3G": _scenario(
        "P3G",
        "No daily loss cap (risk budget 3.75%)",
        "2022-11-01",
        "2022-11-20",
        [
            "--pair",
            "BTC-USD",
            "--initial-cash",
            "1400",
            "--fee-rate",
            "0.001",
            "--llm-strategist",
            "enabled",
            "--llm-calls-per-day",
            "1",
            "--max-position-risk-pct",
            "0.75",
            "--max-daily-risk-budget-pct",
            "3.75",
            "--flatten-policy",
            "daily_close",
            "--debug-limits",
            "basic",
        ],
    ),
    "P3H": _scenario(
        "P3H",
        "Tight daily loss cap 1.0% with 3.75% budget",
        "2022-11-01",
        "2022-11-20",
        [
            "--pair",
            "BTC-USD",
            "--initial-cash",
            "1400",
            "--fee-rate",
            "0.001",
            "--llm-strategist",
            "enabled",
            "--llm-calls-per-day",
            "1",
            "--max-position-risk-pct",
            "0.75",
            "--max-daily-risk-budget-pct",
            "3.75",
            "--max-daily-loss-pct",
            "1.0",
            "--flatten-policy",
            "daily_close",
            "--debug-limits",
            "basic",
        ],
    ),
    # Phase 4: flatten policy matrix
    "P4I": _scenario(
        "P4I",
        "No forced flattening",
        "2021-05-01",
        "2021-05-14",
        [
            "--pair",
            "BTC-USD",
            "--initial-cash",
            "1400",
            "--fee-rate",
            "0.001",
            "--llm-strategist",
            "enabled",
            "--llm-calls-per-day",
            "1",
            "--max-position-risk-pct",
            "0.75",
            "--max-daily-risk-budget-pct",
            "3.75",
            "--max-daily-loss-pct",
            "3.0",
            "--flatten-policy",
            "none",
            "--debug-limits",
            "basic",
        ],
    ),
    "P4J": _scenario(
        "P4J",
        "Daily close flatten at 00:00 UTC",
        "2021-05-01",
        "2021-05-14",
        [
            "--pair",
            "BTC-USD",
            "--initial-cash",
            "1400",
            "--fee-rate",
            "0.001",
            "--llm-strategist",
            "enabled",
            "--llm-calls-per-day",
            "1",
            "--max-position-risk-pct",
            "0.75",
            "--max-daily-risk-budget-pct",
            "3.75",
            "--max-daily-loss-pct",
            "3.0",
            "--flatten-policy",
            "daily_close",
            "--flatten-session-hour",
            "0",
            "--debug-limits",
            "basic",
        ],
    ),
    "P4K": _scenario(
        "P4K",
        "Session flatten aligned to early UTC",
        "2021-05-01",
        "2021-05-14",
        [
            "--pair",
            "BTC-USD",
            "--initial-cash",
            "1400",
            "--fee-rate",
            "0.001",
            "--llm-strategist",
            "enabled",
            "--llm-calls-per-day",
            "1",
            "--max-position-risk-pct",
            "0.75",
            "--max-daily-risk-budget-pct",
            "3.75",
            "--max-daily-loss-pct",
            "3.0",
            "--flatten-policy",
            "session_close_utc",
            "--flatten-session-hour",
            "4",
            "--debug-limits",
            "basic",
        ],
    ),
    # Phase 5A: timeframe caps
    "P5L": _scenario(
        "P5L",
        "Uncapped timeframes",
        "2021-06-01",
        "2021-06-14",
        [
            "--pair",
            "BTC-USD",
            "--initial-cash",
            "1400",
            "--fee-rate",
            "0.001",
            "--llm-strategist",
            "enabled",
            "--llm-calls-per-day",
            "1",
            "--max-position-risk-pct",
            "0.75",
            "--max-daily-risk-budget-pct",
            "3.75",
            "--max-daily-loss-pct",
            "3.0",
            "--timeframes",
            "1h",
            "4h",
            "1d",
            "--debug-limits",
            "basic",
        ],
    ),
    "P5M": _scenario(
        "P5M",
        "Favor 1h, throttle 4h/1d",
        "2021-06-01",
        "2021-06-14",
        [
            "--pair",
            "BTC-USD",
            "--initial-cash",
            "1400",
            "--fee-rate",
            "0.001",
            "--llm-strategist",
            "enabled",
            "--llm-calls-per-day",
            "1",
            "--max-position-risk-pct",
            "0.75",
            "--max-daily-risk-budget-pct",
            "3.75",
            "--max-daily-loss-pct",
            "3.0",
            "--timeframes",
            "1h",
            "4h",
            "1d",
            "--timeframe-trigger-caps",
            "1h:8,4h:2,1d:1",
            "--debug-limits",
            "basic",
        ],
    ),
    # Phase 5B: session multipliers
    "P5N": _scenario(
        "P5N",
        "No session multipliers",
        "2021-06-01",
        "2021-06-14",
        [
            "--pair",
            "BTC-USD",
            "--initial-cash",
            "1400",
            "--fee-rate",
            "0.001",
            "--llm-strategist",
            "enabled",
            "--llm-calls-per-day",
            "1",
            "--max-position-risk-pct",
            "0.75",
            "--max-daily-risk-budget-pct",
            "3.75",
            "--max-daily-loss-pct",
            "3.0",
            "--debug-limits",
            "basic",
        ],
    ),
    "P5O": _scenario(
        "P5O",
        "Aggressive early UTC, throttled later",
        "2021-06-01",
        "2021-06-14",
        [
            "--pair",
            "BTC-USD",
            "--initial-cash",
            "1400",
            "--fee-rate",
            "0.001",
            "--llm-strategist",
            "enabled",
            "--llm-calls-per-day",
            "1",
            "--max-position-risk-pct",
            "0.75",
            "--max-daily-risk-budget-pct",
            "3.75",
            "--max-daily-loss-pct",
            "3.0",
            "--session-trade-multipliers",
            "0-4:1.5,4-24:0.5",
            "--debug-limits",
            "basic",
        ],
    ),
    # Phase 6: LLM cadence
    "P6P": _scenario(
        "P6P",
        "Strategist off baseline (mechanical)",
        "2021-07-01",
        "2021-07-14",
        [
            "--pair",
            "BTC-USD",
            "--initial-cash",
            "1400",
            "--fee-rate",
            "0.001",
            "--llm-strategist",
            "disabled",
            "--use-saved-plans",
            "--max-position-risk-pct",
            "0.75",
            "--max-daily-risk-budget-pct",
            "3.75",
            "--max-daily-loss-pct",
            "3.0",
            "--flatten-policy",
            "daily_close",
            "--debug-limits",
            "basic",
        ],
    ),
    "P6Q": _scenario(
        "P6Q",
        "LLM strategist 1 call/day",
        "2021-07-01",
        "2021-07-14",
        [
            "--pair",
            "BTC-USD",
            "--initial-cash",
            "1400",
            "--fee-rate",
            "0.001",
            "--llm-strategist",
            "enabled",
            "--llm-calls-per-day",
            "1",
            "--max-position-risk-pct",
            "0.75",
            "--max-daily-risk-budget-pct",
            "3.75",
            "--max-daily-loss-pct",
            "3.0",
            "--flatten-policy",
            "daily_close",
            "--debug-limits",
            "basic",
        ],
    ),
    "P6R": _scenario(
        "P6R",
        "LLM strategist 4 calls/day",
        "2021-07-01",
        "2021-07-14",
        [
            "--pair",
            "BTC-USD",
            "--initial-cash",
            "1400",
            "--fee-rate",
            "0.001",
            "--llm-strategist",
            "enabled",
            "--llm-calls-per-day",
            "4",
            "--max-position-risk-pct",
            "0.75",
            "--max-daily-risk-budget-pct",
            "3.75",
            "--max-daily-loss-pct",
            "3.0",
            "--flatten-policy",
            "daily_close",
            "--debug-limits",
            "basic",
        ],
    ),
}


def _maybe_shorten(start: str, end: str, smoke_days: int | None) -> tuple[str, str]:
    if not smoke_days:
        return start, end
    try:
        s = datetime.fromisoformat(start)
        e = datetime.fromisoformat(end)
    except ValueError:
        return start, end
    span = (e - s).days
    if span <= smoke_days:
        return start, end
    window = max(1, smoke_days)
    offset = (span - window)
    # deterministic-ish pick: middle window
    s_smoke = s + timedelta(days=offset // 2)
    e_smoke = s_smoke + timedelta(days=window)
    return s_smoke.date().isoformat(), e_smoke.date().isoformat()


def build_command(
    scenario: dict,
    regime_override: str | None,
    log_dir: Path,
    smoke_days: int | None,
    factor_opts: dict[str, str | None] | None = None,
) -> List[str]:
    start, end = scenario["start"], scenario["end"]
    if regime_override:
        if regime_override not in REGIMES:
            raise ValueError(f"Unknown regime '{regime_override}'")
        start, end = REGIMES[regime_override]
    start, end = _maybe_shorten(start, end, smoke_days)
    run_id = f"{scenario['key'].lower()}_{start}_{end}".replace("-", "")
    log_path = log_dir / f"{run_id}.log"
    args = list(BASE_CMD)
    args.extend(
        [
            "--start",
            start,
            "--end",
            end,
            "--log-level",
            "INFO",
            "--log-json",
            "--log-file",
            str(log_path),
            "--llm-run-id",
            run_id,
        ]
    )
    args.extend(scenario["args"])
    if factor_opts:
        if factor_opts.get("auto_fetch"):
            args.append("--factor-auto-fetch")
        if factor_opts.get("factor_data"):
            args.extend(["--factor-data", factor_opts["factor_data"]])
        if factor_opts.get("btc_csv"):
            args.extend(["--factor-btc-csv", factor_opts["btc_csv"]])
        if factor_opts.get("eth_csv"):
            args.extend(["--factor-eth-csv", factor_opts["eth_csv"]])
        if factor_opts.get("total_csv"):
            args.extend(["--factor-total-csv", factor_opts["total_csv"]])
        if factor_opts.get("factor_start"):
            args.extend(["--factor-start", factor_opts["factor_start"]])
        if factor_opts.get("factor_end"):
            args.extend(["--factor-end", factor_opts["factor_end"]])
        if factor_opts.get("fetch_days"):
            args.extend(["--factor-fetch-days", str(factor_opts["fetch_days"])])
        if factor_opts.get("fetch_interval"):
            args.extend(["--factor-fetch-interval", factor_opts["fetch_interval"]])
        if factor_opts.get("auto_hedge_market"):
            args.append("--auto-hedge-market")
        if factor_opts.get("use_backtest_cache"):
            args.append("--factor-use-backtest-cache")
    return args


def main() -> None:
    parser = argparse.ArgumentParser(description="Run A/B backtest scenarios.")
    parser.add_argument("--phases", nargs="+", default=["P0"], help="Scenario keys to run (e.g., P0 P1A P1B).")
    parser.add_argument("--regime", help="Optional regime override key.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    parser.add_argument("--log-dir", default="logs/ab_runs", help="Where to write per-run logs.")
    parser.add_argument("--smoke-days", type=int, help="Optional max days for smoke runs; trims windows longer than this.")
    parser.add_argument("--factor-auto-fetch", action="store_true", help="Auto-fetch/build factor data per run.")
    parser.add_argument("--factor-data", help="Existing factor file to use (skips auto-fetch).")
    parser.add_argument("--factor-btc-csv", help="Local BTC CSV for factor auto-fetch.")
    parser.add_argument("--factor-eth-csv", help="Local ETH CSV for factor auto-fetch.")
    parser.add_argument("--factor-total-csv", help="Local total mcap CSV for factor auto-fetch.")
    parser.add_argument("--factor-start", help="Optional ISO start for factor slicing.")
    parser.add_argument("--factor-end", help="Optional ISO end for factor slicing.")
    parser.add_argument("--factor-fetch-days", type=int, default=180, help="Days to fetch if using CoinGecko fallback.")
    parser.add_argument("--factor-fetch-interval", default="hourly", choices=["hourly", "daily"], help="CoinGecko interval fallback.")
    parser.add_argument("--auto-hedge-market", action="store_true", help="Enable auto-hedge flag for all scenarios.")
    parser.add_argument("--factor-use-backtest-cache", action="store_true", help="Build factors from backtest OHLCV cache/fetch.")
    args = parser.parse_args()

    missing = [p for p in args.phases if p not in SCENARIOS]
    if missing:
        raise ValueError(f"Unknown scenarios: {', '.join(missing)}")

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    for key in args.phases:
        scenario = SCENARIOS[key]
        factor_opts = None
        if args.factor_auto_fetch or args.factor_data or args.factor_btc_csv or args.factor_eth_csv:
            factor_opts = {
                "auto_fetch": args.factor_auto_fetch,
                "factor_data": args.factor_data,
                "btc_csv": args.factor_btc_csv,
                "eth_csv": args.factor_eth_csv,
                "total_csv": args.factor_total_csv,
                "factor_start": args.factor_start,
                "factor_end": args.factor_end,
                "fetch_days": args.factor_fetch_days,
                "fetch_interval": args.factor_fetch_interval,
                "auto_hedge_market": args.auto_hedge_market,
                "use_backtest_cache": args.factor_use_backtest_cache,
            }
        cmd = build_command(scenario, args.regime, log_dir, args.smoke_days, factor_opts=factor_opts)
        printable = " ".join(shlex.quote(part) for part in cmd)
        print(f"\n[{key}] {scenario['desc']}\n{printable}")
        if args.dry_run:
            continue
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f"Scenario {key} failed with exit code {result.returncode}")


if __name__ == "__main__":
    main()
