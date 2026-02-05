"""Temporal activities for backtest execution.

These activities handle all non-deterministic I/O operations:
- Data loading (API calls, file I/O)
- Simulation execution (long-running computation)
- Result persistence (disk I/O)
"""

from __future__ import annotations

import asyncio
import logging
import os
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
    from backtesting.llm_shim import make_strategist_shim_transport
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
    # Adaptive judge workflow parameters
    # In adaptive mode, judge evaluates at judge_cadence_hours interval
    # Each judge eval can trigger a strategy update (replan) if score is below threshold
    judge_cadence_hours = float(config.get("judge_cadence_hours", 4.0))
    # Number of judge evaluations per day
    judge_evals_per_day = int(24 / max(1, judge_cadence_hours))
    # LLM budget = 1 (initial plan) + possible replans (one per judge eval in worst case)
    llm_calls_per_day = 1 + judge_evals_per_day
    llm_model = config.get("llm_model")
    use_llm_shim = bool(config.get("use_llm_shim"))
    use_judge_shim = bool(config.get("use_judge_shim"))
    llm_cache_dir = config.get("llm_cache_dir", ".cache/strategy_plans")
    risk_params = config.get("risk_params")
    timeframes = config.get("timeframes") or [timeframe]
    flatten_positions_daily = config.get("flatten_positions_daily", False)
    flatten_notional_threshold = config.get("flatten_notional_threshold", 0.0)
    strategy_prompt = config.get("strategy_prompt")

    # Whipsaw / anti-flip-flop controls (new parameters)
    min_hold_hours = config.get("min_hold_hours")
    min_flat_hours = config.get("min_flat_hours")
    if min_flat_hours is None:
        min_flat_hours = 2.0  # Default
    confidence_override_threshold = config.get("confidence_override_threshold", "A")
    priority_skip_confidence_threshold = config.get("priority_skip_confidence_threshold")
    exit_binding_mode = config.get("exit_binding_mode", "category")
    conflicting_signal_policy = config.get("conflicting_signal_policy", "reverse")

    # Walk-away threshold
    walk_away_enabled = config.get("walk_away_enabled", False)
    walk_away_profit_target_pct = config.get("walk_away_profit_target_pct", 25.0)

    # Trade frequency limits
    max_trades_per_day = config.get("max_trades_per_day")
    max_triggers_per_symbol_per_day = config.get("max_triggers_per_symbol_per_day")

    # Debug trigger evaluation sampling
    debug_trigger_sample_rate = config.get("debug_trigger_sample_rate", 0.0)
    debug_trigger_max_samples = config.get("debug_trigger_max_samples", 100)
    indicator_debug_mode = config.get("indicator_debug_mode")
    indicator_debug_keys = config.get("indicator_debug_keys")

    # Vector store for trigger examples
    use_trigger_vector_store = config.get("use_trigger_vector_store", False)

    # Adaptive judge workflow - remaining parameters
    adaptive_replanning = config.get("adaptive_replanning", True)  # Default: enabled
    judge_replan_threshold = config.get("judge_replan_threshold", 40.0)
    judge_check_after_trades = config.get("judge_check_after_trades", 3)
    replan_on_day_boundary = bool(config.get("replan_on_day_boundary", True))

    logger.info(
        "Running LLM strategist backtest",
        extra={
            "symbols": symbols,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "timeframes": timeframes,
            "adaptive_replanning": adaptive_replanning,
            "judge_cadence_hours": judge_cadence_hours,
            "llm_budget": llm_calls_per_day,
        },
    )

    if use_llm_shim:
        llm_client = LLMClient(
            transport=make_strategist_shim_transport(),
            model=llm_model or "shim-strategist",
            allow_fallback=False,
        )
    else:
        llm_client = LLMClient(model=llm_model) if llm_model else LLMClient()
    market_data: Dict[str, Dict[str, pd.DataFrame]] = {}
    for symbol in symbols:
        df = load_ohlcv(symbol, start, end, timeframe=timeframe)
        if df.empty:
            raise ValueError(f"No OHLCV data available for {symbol}")
        market_data[symbol] = {timeframe: df}

    trade_start = ensure_utc(datetime.fromisoformat(config.get("requested_start_date", config["start_date"])))
    trade_end = ensure_utc(datetime.fromisoformat(config.get("requested_end_date", config["end_date"])))

    # Calculate total candles upfront for accurate progress tracking
    all_timestamps = sorted({
        ts for pair_data in market_data.values()
        for df in pair_data.values()
        for ts in df.index
        if trade_start <= ts <= trade_end
    })
    total_candles = len(all_timestamps)
    logger.info("Total candles to process: %d", total_candles)

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
        min_hold_hours=min_hold_hours,
        min_flat_hours=float(min_flat_hours),
        confidence_override_threshold=confidence_override_threshold,
        priority_skip_confidence_threshold=priority_skip_confidence_threshold,
        exit_binding_mode=exit_binding_mode,
        conflicting_signal_policy=conflicting_signal_policy,
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
        indicator_debug_mode=indicator_debug_mode,
        indicator_debug_keys=indicator_debug_keys,
        # Vector store for trigger examples
        use_trigger_vector_store=use_trigger_vector_store,
        # Adaptive judge workflow
        adaptive_replanning=adaptive_replanning,
        judge_cadence_hours=judge_cadence_hours,
        judge_replan_threshold=judge_replan_threshold,
        judge_check_after_trades=judge_check_after_trades,
        replan_on_day_boundary=replan_on_day_boundary,
        use_judge_shim=use_judge_shim,
    )

    from temporalio.client import Client

    # Thread-safe progress state for heartbeats
    progress_state: Dict[str, Any] = {
        "candles_processed": 0,
        "candles_total": total_candles,
        "progress_pct": 0.0,
        "current_timestamp": None,
        "recent_events": [],
    }
    progress_signal_enabled = True
    signal_loop: asyncio.AbstractEventLoop | None = None
    signal_client: Client | None = None
    signal_handle = None
    last_signal_payload: Dict[str, Any] = {}

    def _init_progress_signal() -> None:
        nonlocal signal_loop, signal_client, signal_handle, progress_signal_enabled
        if not progress_signal_enabled or signal_handle is not None:
            return
        try:
            info = activity.info()
            temporal_address = os.environ.get("TEMPORAL_ADDRESS", "localhost:7233")
            temporal_namespace = os.environ.get("TEMPORAL_NAMESPACE", "default")
            signal_loop = asyncio.new_event_loop()
            signal_client = signal_loop.run_until_complete(
                Client.connect(temporal_address, namespace=temporal_namespace)
            )
            signal_handle = signal_client.get_workflow_handle(info.workflow_id)
        except Exception as exc:
            progress_signal_enabled = False
            logger.debug("Progress signal setup failed: %s", exc)

    def _signal_progress() -> None:
        nonlocal last_signal_payload
        _init_progress_signal()
        if not signal_handle or not signal_loop:
            return
        raw_pct = progress_state.get("progress_pct", 0.0) or 0.0
        scaled_progress = 20.0 + min(max(raw_pct, 0.0), 100.0) * 0.75
        payload = {
            "progress": scaled_progress,
            "candles_processed": progress_state.get("candles_processed", 0),
            "candles_total": progress_state.get("candles_total", total_candles),
            "timestamp": progress_state.get("current_timestamp"),
            "current_phase": "Simulating (LLM)",
        }
        if payload == last_signal_payload:
            return
        last_signal_payload = payload.copy()
        try:
            signal_loop.run_until_complete(signal_handle.signal("update_progress", payload))
        except Exception as exc:
            logger.debug("Progress signal failed: %s", exc)

    def progress_callback(progress: Dict[str, Any]) -> None:
        """Update progress state from backtester (thread-safe)."""
        progress_state.update(progress)

    # Attach progress callback to backtester
    backtester.progress_callback = progress_callback

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
        _signal_progress()
        while worker_thread.is_alive():
            try:
                # Send heartbeat with real progress info
                activity.heartbeat({
                    "phase": "llm_backtest",
                    "status": "running",
                    "candles_processed": progress_state.get("candles_processed", 0),
                    "candles_total": progress_state.get("candles_total", total_candles),
                    "progress_pct": progress_state.get("progress_pct", 0.0),
                    "current_timestamp": progress_state.get("current_timestamp"),
                    "current_day": progress_state.get("current_day"),
                    "recent_events": progress_state.get("recent_events", [])[-5:],
                })
                _signal_progress()
            except Exception as exc:
                logger.warning("LLM backtest heartbeat failed: %s", exc)
            time.sleep(5)  # More frequent heartbeats for better progress updates
        worker_thread.join()
        if error_holder.get("error"):
            raise error_holder["error"]
        result = result_holder["result"]
    finally:
        worker_thread.join(timeout=1)
        if signal_loop:
            if signal_client is not None:
                close_client = getattr(signal_client, "close", None)
                if callable(close_client):
                    signal_loop.run_until_complete(close_client())
            signal_loop.close()

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
        "bar_decisions": result.bar_decisions,
        "final_cash": result.final_cash,
        "final_positions": result.final_positions,
        "intraday_judge_history": result.intraday_judge_history,
        "judge_triggered_replans": result.judge_triggered_replans,
    }
    if isinstance(result.summary, dict):
        if result.summary.get("trigger_evaluation_samples") is not None:
            llm_data["trigger_evaluation_samples"] = result.summary["trigger_evaluation_samples"]
        if result.summary.get("trigger_evaluation_sample_count") is not None:
            llm_data["trigger_evaluation_sample_count"] = result.summary["trigger_evaluation_sample_count"]

    run_id = config.get("run_id", "llm-backtest")
    full_payload = {
        "run_id": run_id,
        "status": "completed",
        "config": config,
        "strategy": config.get("strategy"),
        "equity_curve": equity_curve,
        "trades": trades,
        "trade_log": result.trade_log,
        "candles_processed": backtester.candles_processed,
        "candles_total": total_candles,
        "summary": summary,
        "llm_data": llm_data,
    }
    try:
        from ops_api.routers.backtests import save_backtest_to_disk, BACKTEST_CACHE_DIR

        save_backtest_to_disk(run_id, full_payload)
        artifact_path = str(BACKTEST_CACHE_DIR / f"{run_id}.pkl")
    except Exception as exc:
        logger.warning("Failed to persist LLM backtest artifacts for %s: %s", run_id, exc)
        artifact_path = ""

    return {
        "summary": summary,
        "candles_processed": backtester.candles_processed,
        "candles_total": total_candles,
        "has_more": False,
        "artifact_path": artifact_path,
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

    Timeout: 120 seconds (DB persist + disk I/O for large payloads)
    """
    from ops_api.routers.backtests import save_backtest_to_disk
    from backtesting.persistence import persist_backtest_results

    logger.info(f"Persisting backtest results for {run_id}")

    results_payload = results
    llm_data = results.get("llm_data") or {}
    artifact_path = llm_data.get("artifact_path") if isinstance(llm_data, dict) else None
    if artifact_path:
        try:
            from ops_api.routers.backtests import load_backtest_from_disk

            disk_payload = load_backtest_from_disk(run_id)
        except Exception as exc:
            logger.warning("Failed to load artifact payload for %s: %s", run_id, exc)
            disk_payload = None
        if disk_payload:
            results_payload = dict(disk_payload)
            for key in (
                "run_id",
                "status",
                "started_at",
                "completed_at",
                "candles_total",
                "candles_processed",
                "config",
                "strategy",
                "summary",
            ):
                if results.get(key) is not None:
                    results_payload[key] = results[key]
            if isinstance(results_payload.get("llm_data"), dict):
                results_payload["llm_data"].setdefault("artifact_path", artifact_path)
                results_payload["llm_data"].setdefault("stored", True)

    try:
        await persist_backtest_results(run_id, results_payload)
    except Exception as exc:
        logger.warning("Failed to persist backtest %s to DB: %s", run_id, exc)

    # Write to disk cache unless artifacts were already stored in the activity.
    llm_data = results_payload.get("llm_data") or {}
    artifact_path = llm_data.get("artifact_path") if isinstance(llm_data, dict) else None
    if artifact_path:
        logger.info("Skipping disk write for %s; artifacts already stored at %s", run_id, artifact_path)
    else:
        await asyncio.to_thread(
            save_backtest_to_disk,
            run_id=run_id,
            data=results
        )

    logger.info(f"Successfully persisted backtest {run_id}")
