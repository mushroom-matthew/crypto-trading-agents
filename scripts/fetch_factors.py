#!/usr/bin/env python3
"""Fetch BTC/ETH factor proxies from CoinGecko and cache to disk."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_loader.factors import build_factor_df, build_factors_from_csvs, fetch_coin_market_chart, resample_factors, save_factors


def _load_price_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
    elif "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.set_index("time")
    else:
        raise ValueError(f"{path} missing timestamp/time column")
    if "close" not in df.columns:
        raise ValueError(f"{path} must include a close column")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build factor proxies (BTC/ETH market, dominance, eth_beta).")
    parser.add_argument("--days", type=int, default=180, help="Number of days to fetch (CoinGecko fallback)")
    parser.add_argument("--interval", default="hourly", choices=["hourly", "daily"], help="CoinGecko sampling interval")
    parser.add_argument("--timeframe", default=None, help="Optional resample timeframe (e.g., 1h, 4h, 1d)")
    parser.add_argument("--out", type=Path, default=Path("data/factors.parquet"), help="Output path (.parquet or .csv)")
    parser.add_argument("--btc-csv", type=Path, help="Optional local CSV for BTC with columns timestamp,close[,market_cap]")
    parser.add_argument("--eth-csv", type=Path, help="Optional local CSV for ETH with columns timestamp,close[,market_cap]")
    parser.add_argument("--total-csv", type=Path, help="Optional local CSV for total market cap (timestamp,market_cap)")
    parser.add_argument("--start", help="Optional start datetime (ISO) to slice local CSVs")
    parser.add_argument("--end", help="Optional end datetime (ISO) to slice local CSVs")
    parser.add_argument("--use-backtest-cache", action="store_true", help="Load BTC/ETH from backtesting dataset cache/fetch instead of CSVs/Coingecko.")
    parser.add_argument("--btc-symbol", default="BTC-USD", help="BTC symbol when using backtest cache.")
    parser.add_argument("--eth-symbol", default="ETH-USD", help="ETH symbol when using backtest cache.")
    args = parser.parse_args()

    if args.btc_csv and args.eth_csv:
        factors = build_factors_from_csvs(
            btc_csv=args.btc_csv,
            eth_csv=args.eth_csv,
            total_csv=args.total_csv,
            timeframe=args.timeframe,
            start=pd.to_datetime(args.start, utc=True) if args.start else None,
            end=pd.to_datetime(args.end, utc=True) if args.end else None,
            use_backtest_cache=args.use_backtest_cache,
            btc_symbol=args.btc_symbol,
            eth_symbol=args.eth_symbol,
        )
    else:
        if args.use_backtest_cache and args.start and args.end:
            from backtesting.dataset import load_ohlcv

            btc = load_ohlcv(args.btc_symbol, start=pd.to_datetime(args.start), end=pd.to_datetime(args.end), timeframe=args.timeframe or "1h")
            eth = load_ohlcv(args.eth_symbol, start=pd.to_datetime(args.start), end=pd.to_datetime(args.end), timeframe=args.timeframe or "1h")
            factors = build_factor_df(btc, eth)
            if args.timeframe:
                factors = resample_factors(factors, timeframe=args.timeframe)
        else:
            btc = fetch_coin_market_chart("bitcoin", days=args.days, interval=args.interval)
            eth = fetch_coin_market_chart("ethereum", days=args.days, interval=args.interval)
            total = None
            factors = build_factor_df(btc, eth, total_mcap=total)
            if args.timeframe:
                factors = resample_factors(factors, timeframe=args.timeframe)
    save_factors(factors, args.out)
    print(f"Wrote factors to {args.out} (rows={len(factors)})")


if __name__ == "__main__":
    main()
