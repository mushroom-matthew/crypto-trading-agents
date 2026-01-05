"""Temporal activities for backtest execution.

These activities handle all non-deterministic I/O operations:
- Data loading (API calls, file I/O)
- Simulation execution (long-running computation)
- Result persistence (disk I/O)
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from temporalio import activity

logger = logging.getLogger(__name__)


@activity.defn
async def load_ohlcv_activity(config: Dict[str, Any]) -> Dict[str, Any]:
    """Load OHLCV data from API or cache (non-deterministic I/O).

    Args:
        config: Backtest configuration with symbols, date range, timeframe

    Returns:
        Dict mapping symbol -> {data: List[Dict], total_candles: int}

    Timeout: 60 seconds (data loading from API/cache)
    Retry: 3 attempts with exponential backoff
    """
    from backtesting.dataset import load_ohlcv

    symbols = config["symbols"]
    start = datetime.fromisoformat(config["start_date"])
    end = datetime.fromisoformat(config["end_date"])
    timeframe = config.get("timeframe", "1h")

    logger.info(f"Loading OHLCV data for {len(symbols)} symbols from {start} to {end}")

    # Non-deterministic: May hit API or read from disk cache
    ohlcv_dict = {}
    for symbol in symbols:
        try:
            df = await asyncio.to_thread(
                load_ohlcv,
                pair=symbol,
                start=start,
                end=end,
                timeframe=timeframe
            )
            ohlcv_dict[symbol] = {
                "data": df.to_dict(orient="records"),
                "total_candles": len(df)
            }
            logger.info(f"Loaded {len(df)} candles for {symbol}")
        except Exception as e:
            logger.error(f"Failed to load OHLCV for {symbol}: {e}")
            raise

    return ohlcv_dict


@activity.defn
def run_simulation_chunk_activity(
    config: Dict[str, Any],
    ohlcv_data: Dict[str, Any],
    offset: int = 0,
    chunk_size: int = 5000,
) -> Dict[str, Any]:
    """Run simulation for a chunk of candles (deterministic, but long-running).

    Args:
        config: Backtest configuration
        ohlcv_data: OHLCV data dict from load_ohlcv_activity
        offset: Starting candle index (for continue-as-new)
        chunk_size: Max candles to process in this chunk

    Returns:
        Dict with equity_curve, trades, candles_processed, has_more

    Timeout: 15 minutes (long computation)
    Heartbeat: 30 seconds (prevents timeout during long runs)
    """
    from backtesting.simulator import run_backtest, run_portfolio_backtest
    from backtesting.strategies import StrategyWrapperConfig
    import pandas as pd

    logger.info(f"Running simulation chunk: offset={offset}, chunk_size={chunk_size}")

    # Determine backtest type
    symbols = config["symbols"]
    is_portfolio = len(symbols) > 1

    # Get total candles (assume all symbols have same length for now)
    first_symbol = symbols[0]
    total_candles = ohlcv_data[first_symbol]["total_candles"]

    # Calculate chunk boundaries
    chunk_start_idx = offset
    chunk_end_idx = min(offset + chunk_size, total_candles)
    has_more = chunk_end_idx < total_candles

    logger.info(f"Processing candles {chunk_start_idx} to {chunk_end_idx} of {total_candles}")

    # Progress callback with activity heartbeats
    def progress_callback(idx, total, timestamp):
        """Send heartbeat to prevent activity timeout."""
        try:
            # Calculate global progress (20-95% of overall backtest)
            global_idx = offset + idx
            global_progress = 20 + (global_idx / total_candles) * 75

            activity.heartbeat({
                "progress": global_progress,
                "candles_processed": global_idx,
                "timestamp": timestamp,
                "current_phase": "Simulating"
            })
        except Exception as e:
            logger.warning(f"Heartbeat failed: {e}")

    # Run appropriate simulation
    if is_portfolio:
        # Portfolio backtest
        result = run_portfolio_backtest(
            pairs=symbols,
            start=datetime.fromisoformat(config["start_date"]),
            end=datetime.fromisoformat(config["end_date"]),
            initial_cash=config.get("initial_cash", 10000.0),
            fee_rate=config.get("fee_rate", 0.001),
            strategy_config=StrategyWrapperConfig(**config.get("strategy_config", {})),
            flatten_positions_daily=config.get("flatten_positions_daily", False),
            risk_limits=None,  # TODO: Support risk limits
            progress_callback=progress_callback,
        )

        # Extract data
        equity_curve = result.equity_curve.to_dict(orient="records")
        trades = result.trades.to_dict(orient="records") if not result.trades.empty else []

    else:
        # Single-pair backtest
        symbol = symbols[0]

        # Slice OHLCV data for this chunk (if needed)
        # For now, run full backtest - chunking optimization can come later
        result = run_backtest(
            pair=symbol,
            start=datetime.fromisoformat(config["start_date"]),
            end=datetime.fromisoformat(config["end_date"]),
            initial_cash=config.get("initial_cash", 10000.0),
            fee_rate=config.get("fee_rate", 0.001),
            strategy_config=StrategyWrapperConfig(**config.get("strategy_config", {})),
            flatten_positions_daily=config.get("flatten_positions_daily", False),
            risk_limits=None,
            progress_callback=progress_callback,
        )

        equity_curve = result.equity_curve.to_dict(orient="records")
        trades = result.trades.to_dict(orient="records") if not result.trades.empty else []

    logger.info(f"Chunk complete: {len(equity_curve)} equity points, {len(trades)} trades")

    return {
        "equity_curve": equity_curve,
        "trades": trades,
        "candles_processed": chunk_end_idx - chunk_start_idx,
        "has_more": has_more,
        "summary": result.summary if hasattr(result, 'summary') else {}
    }


@activity.defn
async def persist_results_activity(
    run_id: str,
    results: Dict[str, Any]
) -> None:
    """Save backtest results to disk (non-deterministic I/O).

    Args:
        run_id: Backtest run identifier
        results: Complete backtest results to persist

    Timeout: 10 seconds (disk I/O)
    """
    from ops_api.routers.backtests import save_backtest_to_disk

    logger.info(f"Persisting backtest results for {run_id}")

    # Write to disk cache
    await asyncio.to_thread(
        save_backtest_to_disk,
        run_id=run_id,
        data=results
    )

    logger.info(f"Successfully persisted backtest {run_id}")
