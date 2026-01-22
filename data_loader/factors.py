"""CoinGecko-based factor fetcher and normalizer."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import requests

DEFAULT_HEADERS = {"Accept": "application/json", "User-Agent": "crypto-trading-agents/ factor fetcher"}


def _to_frame(payload: Dict[str, Any]) -> pd.DataFrame:
    """Parse CoinGecko market_chart payload into a DataFrame with close and market_cap."""

    prices = payload.get("prices", [])
    caps = payload.get("market_caps", [])
    if not prices:
        raise ValueError("prices field missing from market_chart response")
    price_df = pd.DataFrame(prices, columns=["timestamp", "close"])
    price_df["timestamp"] = pd.to_datetime(price_df["timestamp"], unit="ms", utc=True)
    price_df = price_df.set_index("timestamp")
    if caps:
        cap_df = pd.DataFrame(caps, columns=["timestamp", "market_cap"])
        cap_df["timestamp"] = pd.to_datetime(cap_df["timestamp"], unit="ms", utc=True)
        cap_df = cap_df.set_index("timestamp")
        price_df = price_df.join(cap_df, how="left")
    return price_df


def fetch_coin_market_chart(
    coin_id: str,
    vs_currency: str = "usd",
    days: int = 90,
    interval: str = "hourly",
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    """Fetch CoinGecko market_chart data for a coin (close + market cap)."""

    sess = session or requests.Session()
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": vs_currency, "days": days, "interval": interval}
    headers = dict(DEFAULT_HEADERS)
    api_key = os.getenv("COINGECKO_API_KEY")
    if api_key:
        headers["x-cg-pro-api-key"] = api_key
    resp = sess.get(url, params=params, headers=headers, timeout=30)
    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        if resp.status_code == 401:
            raise RuntimeError("CoinGecko returned 401; set COINGECKO_API_KEY to use this fetcher.") from exc
        raise
    payload = resp.json()
    return _to_frame(payload)


def build_factor_df(btc: pd.DataFrame, eth: pd.DataFrame, total_mcap: pd.DataFrame | None = None) -> pd.DataFrame:
    """Create factor dataframe with market, dominance, eth_beta columns."""

    aligned = pd.concat(
        {
            "btc_close": btc["close"],
            "eth_close": eth["close"],
            "btc_mcap": btc["market_cap"] if "market_cap" in btc.columns else None,
            "eth_mcap": eth["market_cap"] if "market_cap" in eth.columns else None,
        },
        axis=1,
        join="inner",
    )
    aligned = aligned.dropna(subset=["btc_close", "eth_close"])
    if aligned.empty:
        return aligned
    factor_series: dict[str, pd.Series] = {}
    market = (aligned["btc_close"] + aligned["eth_close"]) / 2.0
    factor_series["market"] = market.pct_change()
    eth_ratio = aligned["eth_close"] / aligned["btc_close"]
    factor_series["eth_beta"] = eth_ratio.pct_change()
    if "btc_mcap" in aligned.columns and "eth_mcap" in aligned.columns and not aligned["btc_mcap"].isna().all():
        total_cap = None
        if total_mcap is not None and not total_mcap.empty and "market_cap" in total_mcap.columns:
            total_cap = total_mcap["market_cap"].reindex(aligned.index, method="ffill")
        if total_cap is None or total_cap.empty:
            total_cap = aligned["btc_mcap"].fillna(0) + aligned["eth_mcap"].fillna(0)
        dominance = aligned["btc_mcap"] / total_cap.replace(0, pd.NA)
        factor_series["dominance"] = dominance.pct_change()
    factors = pd.DataFrame(factor_series, index=aligned.index)
    return factors.dropna(how="all")


def resample_factors(df: pd.DataFrame, timeframe: str = "1h") -> pd.DataFrame:
    """Resample factor dataframe to a desired timeframe."""

    return df.resample(timeframe).mean().dropna()


def save_factors(df: pd.DataFrame, path: Path) -> None:
    """Persist factor dataframe to parquet or CSV based on extension."""

    path.parent.mkdir(parents=True, exist_ok=True)
    index_label = df.index.name or "timestamp"
    if path.suffix == ".parquet":
        try:
            df.to_parquet(path)
            return
        except ImportError:
            # Fallback to CSV when parquet engine is unavailable.
            fallback = path.with_suffix(".csv")
            fallback.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(fallback, index_label=index_label)
            return
    elif path.suffix == ".csv":
        df.to_csv(path, index_label=index_label)
    else:
        reset = df.reset_index()
        if "index" in reset.columns and index_label != "index":
            reset = reset.rename(columns={"index": index_label})
        path.write_text(json.dumps(reset.to_dict(orient="list")), encoding="utf-8")


def load_cached_factors(path: Path) -> pd.DataFrame:
    """Load factors dataframe from disk; supports parquet or CSV."""

    if not path.exists():
        if path.suffix == ".parquet":
            csv_path = path.with_suffix(".csv")
            if csv_path.exists():
                path = csv_path
            else:
                raise FileNotFoundError(path)
    if path.suffix == ".parquet":
        try:
            return pd.read_parquet(path)
        except ImportError:
            # Try CSV fallback if parquet engine missing.
            csv_path = path.with_suffix(".csv")
            if csv_path.exists():
                return pd.read_csv(csv_path, parse_dates=["timestamp"], index_col="timestamp")
            raise
    if path.suffix == ".csv":
        df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
        return df
    payload = json.loads(path.read_text(encoding="utf-8"))
    df = pd.DataFrame(payload)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
    return df


def build_factors_from_csvs(
    btc_csv: Path,
    eth_csv: Path,
    total_csv: Path | None = None,
    timeframe: str | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pd.DataFrame:
    """
    Build factors from local BTC/ETH CSVs (timestamp, close[, market_cap]) for a given window.

    Args:
        btc_csv: path to BTC OHLCV CSV.
        eth_csv: path to ETH OHLCV CSV.
        total_csv: optional total market cap CSV (timestamp, market_cap).
        timeframe: optional resample (e.g., 1h, 4h, 1d).
        start/end: optional datetime bounds to slice data before factor calc.
    """

    def _load(path: Path) -> pd.DataFrame:
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

    btc = _load(btc_csv)
    eth = _load(eth_csv)
    total = _load(total_csv) if total_csv else None
    if start:
        btc = btc[btc.index >= pd.to_datetime(start, utc=True)]
        eth = eth[eth.index >= pd.to_datetime(start, utc=True)]
        if total is not None:
            total = total[total.index >= pd.to_datetime(start, utc=True)]
    if end:
        btc = btc[btc.index <= pd.to_datetime(end, utc=True)]
        eth = eth[eth.index <= pd.to_datetime(end, utc=True)]
        if total is not None:
            total = total[total.index <= pd.to_datetime(end, utc=True)]
    factors = build_factor_df(btc, eth, total_mcap=total)
    if timeframe:
        factors = resample_factors(factors, timeframe=timeframe)
    return factors


def fetch_or_build_factors(
    out_path: Path,
    days: int = 180,
    interval: str = "hourly",
    timeframe: str | None = None,
    btc_csv: Path | None = None,
    eth_csv: Path | None = None,
    total_csv: Path | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    use_backtest_cache: bool = False,
    btc_symbol: str = "BTC-USD",
    eth_symbol: str = "ETH-USD",
) -> Path:
    """
    Build factors from local CSVs if provided; otherwise fetch CoinGecko market data.

    Saves to out_path and returns the path. If saving fails (e.g., missing parquet engine),
    falls back to CSV with the same stem.
    """

    if btc_csv and eth_csv:
        factors = build_factors_from_csvs(
            btc_csv=btc_csv,
            eth_csv=eth_csv,
            total_csv=total_csv,
            timeframe=timeframe,
            start=start,
            end=end,
        )
    elif use_backtest_cache and start and end:
        try:
            from backtesting.dataset import load_ohlcv
        except Exception:
            load_ohlcv = None
        if load_ohlcv is None:
            btc = eth = None
        else:
            btc = load_ohlcv(btc_symbol, start=start, end=end, timeframe=timeframe or "1h")
            eth = load_ohlcv(eth_symbol, start=start, end=end, timeframe=timeframe or "1h")
        if btc is None or eth is None:
            raise RuntimeError("Backtest cache load failed; provide CSVs or disable use_backtest_cache.")
        total = None
        factors = build_factor_df(btc, eth, total_mcap=None)
        if timeframe:
            factors = resample_factors(factors, timeframe=timeframe)
    else:
        btc = fetch_coin_market_chart("bitcoin", days=days, interval=interval)
        eth = fetch_coin_market_chart("ethereum", days=days, interval=interval)
        factors = build_factor_df(btc, eth)
        if timeframe:
            factors = resample_factors(factors, timeframe=timeframe)
    try:
        save_factors(factors, out_path)
        return out_path
    except Exception:
        # Fallback: write CSV next to requested path
        fallback = out_path.with_suffix(".csv")
        save_factors(factors, fallback)
        return fallback
