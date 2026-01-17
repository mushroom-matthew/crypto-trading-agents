"""Temporal activities for backtest execution.

These activities handle all non-deterministic I/O operations:
- Data loading (API calls, file I/O)
- Simulation execution (long-running computation)
- Result persistence (disk I/O)
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
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
    from data_loader.utils import ensure_utc

    symbols = config["symbols"]
    start = ensure_utc(datetime.fromisoformat(config["start_date"]))
    end = ensure_utc(datetime.fromisoformat(config["end_date"]))
    trade_start = ensure_utc(datetime.fromisoformat(config.get("requested_start_date", config["start_date"])))
    trade_end = ensure_utc(datetime.fromisoformat(config.get("requested_end_date", config["end_date"])))
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
            window = df[(df.index >= trade_start) & (df.index <= trade_end)]
            ohlcv_dict[symbol] = {
                "data": df.to_dict(orient="records"),
                "total_candles": len(window)
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

    from data_loader.utils import ensure_utc

    trade_start = ensure_utc(datetime.fromisoformat(config.get("requested_start_date", config["start_date"])))
    trade_end = ensure_utc(datetime.fromisoformat(config.get("requested_end_date", config["end_date"])))

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
            initial_allocations=config.get("initial_allocations"),
            flatten_positions_daily=config.get("flatten_positions_daily", False),
            risk_limits=None,  # TODO: Support risk limits
            progress_callback=progress_callback,
            trade_start=trade_start,
            trade_end=trade_end,
        )

        # Extract data - equity_curve is a Series, not DataFrame
        equity_curve = [
            {"timestamp": str(ts), "equity": float(val)}
            for ts, val in result.equity_curve.items()
        ]
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
            initial_allocations=config.get("initial_allocations"),
            flatten_positions_daily=config.get("flatten_positions_daily", False),
            risk_limits=None,
            progress_callback=progress_callback,
            trade_start=trade_start,
            trade_end=trade_end,
        )

        # equity_curve is a Series, not DataFrame
        equity_curve = [
            {"timestamp": str(ts), "equity": float(val)}
            for ts, val in result.equity_curve.items()
        ]
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
def run_llm_backtest_activity(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run LLM strategist backtest for the full range (no chunking)."""
    from agents.strategies.llm_client import LLMClient
    from backtesting.llm_strategist_runner import LLMStrategistBacktester
    from backtesting.dataset import load_ohlcv
    from data_loader.utils import ensure_utc
    from backtesting.simulator import _compute_metrics_summary
    import pandas as pd

    symbols = config["symbols"]
    start = ensure_utc(datetime.fromisoformat(config["start_date"]))
    end = ensure_utc(datetime.fromisoformat(config["end_date"]))
    timeframe = config.get("timeframe", "1h")
    initial_cash = config.get("initial_cash", 10000.0)
    initial_allocations = config.get("initial_allocations")
    fee_rate = config.get("fee_rate", 0.001)
    llm_calls_per_day = int(config.get("llm_calls_per_day", 1))
    llm_model = config.get("llm_model")
    llm_cache_dir = config.get("llm_cache_dir", ".cache/strategy_plans")
    risk_params = config.get("risk_params")
    timeframes = config.get("timeframes") or [timeframe]
    flatten_positions_daily = config.get("flatten_positions_daily", False)
    flatten_notional_threshold = config.get("flatten_notional_threshold", 0.0)
    strategy_prompt = config.get("strategy_prompt")

    # Whipsaw / anti-flip-flop controls (new parameters)
    min_hold_hours = config.get("min_hold_hours")
    if min_hold_hours is None:
        min_hold_hours = 2.0  # Default
    min_flat_hours = config.get("min_flat_hours")
    if min_flat_hours is None:
        min_flat_hours = 2.0  # Default
    confidence_override_threshold = config.get("confidence_override_threshold", "A")

    # Walk-away threshold
    walk_away_enabled = config.get("walk_away_enabled", False)
    walk_away_profit_target_pct = config.get("walk_away_profit_target_pct", 25.0)

    # Trade frequency limits
    max_trades_per_day = config.get("max_trades_per_day")
    max_triggers_per_symbol_per_day = config.get("max_triggers_per_symbol_per_day")

    # Debug trigger evaluation sampling
    debug_trigger_sample_rate = config.get("debug_trigger_sample_rate", 0.0)
    debug_trigger_max_samples = config.get("debug_trigger_max_samples", 100)

    # Vector store for trigger examples
    use_trigger_vector_store = config.get("use_trigger_vector_store", False)

    logger.info(
        "Running LLM strategist backtest",
        extra={"symbols": symbols, "start": start.isoformat(), "end": end.isoformat(), "timeframes": timeframes},
    )

    llm_client = LLMClient(model=llm_model) if llm_model else LLMClient()
    market_data: Dict[str, Dict[str, pd.DataFrame]] = {}
    for symbol in symbols:
        df = load_ohlcv(symbol, start, end, timeframe=timeframe)
        if df.empty:
            raise ValueError(f"No OHLCV data available for {symbol}")
        market_data[symbol] = {timeframe: df}

    trade_start = ensure_utc(datetime.fromisoformat(config.get("requested_start_date", config["start_date"])))
    trade_end = ensure_utc(datetime.fromisoformat(config.get("requested_end_date", config["end_date"])))

    backtester = LLMStrategistBacktester(
        pairs=symbols,
        start=trade_start,
        end=trade_end,
        initial_cash=initial_cash,
        fee_rate=fee_rate,
        llm_client=llm_client,
        cache_dir=Path(llm_cache_dir),
        llm_calls_per_day=llm_calls_per_day,
        risk_params=risk_params,
        timeframes=timeframes,
        market_data=market_data,
        flatten_positions_daily=flatten_positions_daily,
        flatten_notional_threshold=flatten_notional_threshold,
        min_hold_hours=float(min_hold_hours),
        min_flat_hours=float(min_flat_hours),
        confidence_override_threshold=confidence_override_threshold,
        initial_allocations=initial_allocations,
        strategy_prompt=strategy_prompt,
        # Walk-away threshold (passed to backtester for tracking)
        walk_away_enabled=walk_away_enabled,
        walk_away_profit_target_pct=walk_away_profit_target_pct,
        max_trades_per_day=max_trades_per_day,
        max_triggers_per_symbol_per_day=max_triggers_per_symbol_per_day,
        # Debug trigger evaluation sampling
        debug_trigger_sample_rate=debug_trigger_sample_rate,
        debug_trigger_max_samples=debug_trigger_max_samples,
        # Vector store for trigger examples
        use_trigger_vector_store=use_trigger_vector_store,
    )
    result_holder: Dict[str, Any] = {}
    error_holder: Dict[str, Exception] = {}

    def _run_backtest() -> None:
        try:
            result_holder["result"] = backtester.run(run_id=config.get("run_id", "llm-backtest"))
        except Exception as exc:
            error_holder["error"] = exc

    worker_thread = threading.Thread(target=_run_backtest, name="llm-backtest-worker", daemon=True)
    worker_thread.start()

    try:
        while worker_thread.is_alive():
            try:
                activity.heartbeat({"phase": "llm_backtest", "status": "running"})
            except Exception as exc:
                logger.warning("LLM backtest heartbeat failed: %s", exc)
            time.sleep(10)
        worker_thread.join()
        if error_holder.get("error"):
            raise error_holder["error"]
        result = result_holder["result"]
    finally:
        worker_thread.join(timeout=1)

    equity_curve = [
        {"timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts), "equity": float(val)}
        for ts, val in result.equity_curve.items()
    ]
    fills_df = result.fills
    trades: List[Dict[str, Any]] = []
    if not fills_df.empty:
        for record in fills_df.to_dict(orient="records"):
            ts = record.get("timestamp")
            if hasattr(ts, "isoformat"):
                record["timestamp"] = ts.isoformat()
            trades.append(record)

    if fills_df.empty:
        trades_for_metrics = pd.DataFrame(columns=["time", "symbol", "side", "qty", "price", "fee"])
    else:
        trades_for_metrics = fills_df.rename(columns={"timestamp": "time"}).copy()
        trades_for_metrics["time"] = pd.to_datetime(trades_for_metrics["time"], utc=True)

    summary = _compute_metrics_summary(result.equity_curve, trades_for_metrics, initial_cash)
    if isinstance(result.summary, dict) and result.summary.get("run_summary") is not None:
        summary["run_summary"] = result.summary["run_summary"]
    summary["llm_costs"] = result.llm_costs
    llm_data = {
        "plan_log": result.plan_log,
        "llm_costs": result.llm_costs,
        "daily_reports": result.daily_reports,
        "final_cash": result.final_cash,
        "final_positions": result.final_positions,
    }

    candles_processed = len(equity_curve)
    return {
        "equity_curve": equity_curve,
        "trades": trades,
        "candles_processed": candles_processed,
        "candles_total": candles_processed,
        "has_more": False,
        "summary": summary,
        "llm_data": llm_data,
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
