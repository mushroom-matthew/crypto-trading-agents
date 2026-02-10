"""Temporal workflow for backtest execution.

Provides durable, fault-tolerant backtest execution with:
- Continue-as-new for long-running backtests
- Real-time progress tracking via queries
- Deterministic replay for audit trails
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from temporalio import workflow
from temporalio.common import RetryPolicy

# Import activities
with workflow.unsafe.imports_passed_through():
    from backtesting.activities import (
        load_ohlcv_activity,
        run_simulation_chunk_activity,
        run_llm_backtest_activity,
        persist_results_activity,
    )

logger = logging.getLogger(__name__)


@workflow.defn
class BacktestWorkflow:
    """Durable backtest execution workflow.

    Handles:
    - Data loading via activity
    - Chunked simulation with continue-as-new for large backtests
    - Progress tracking via signals and queries
    - Result persistence to disk

    Continue-as-new trigger: Every 5000 candles to prevent unbounded history
    """

    def __init__(self) -> None:
        self.run_id: str = ""
        self.progress: float = 0.0
        self.status: str = "queued"
        self.current_phase: str = "Initializing"
        self.candles_total: int = 0
        self.candles_processed: int = 0
        self.current_timestamp: str = ""
        self.started_at: Optional[str] = None
        self.completed_at: Optional[str] = None
        self.error: Optional[str] = None

        # Config storage (for persist)
        self._config: Dict[str, Any] = {}

        # Results accumulation (for continue-as-new)
        self.equity_curve: List[Dict] = []
        self.trades: List[Dict] = []
        self.summary: Dict[str, Any] = {}
        self.llm_data: Dict[str, Any] | None = None

    @workflow.run
    async def run(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute backtest with continue-as-new for long runs.

        Args:
            config: Backtest configuration with:
                - run_id: Unique identifier
                - symbols: List of symbols to backtest
                - start_date, end_date: ISO format datetime strings
                - timeframe: e.g., "1h", "1d"
                - initial_cash: Starting capital
                - strategy: Strategy name (optional)
                - resume_state: State from previous continue-as-new (optional)

        Returns:
            Dict with final backtest results
        """
        self.run_id = config["run_id"]
        self._config = {
            key: value
            for key, value in config.items()
            if key not in {"resume_state", "ohlcv_data"}
        }
        self.status = "running"
        self.started_at = workflow.now().isoformat()
        strategy = config.get("strategy", "baseline")

        try:
            # Check if resuming from continue-as-new
            if config.get("resume_state"):
                workflow.logger.info(f"Resuming backtest {self.run_id} from continue-as-new")
                self._restore_state(config["resume_state"])
            else:
                workflow.logger.info(f"Starting new backtest {self.run_id}")

            if strategy == "llm_strategist":
                # LLM strategist runs in a single activity (no chunking/continue-as-new).
                await self._run_llm_backtest(config)
            else:
                # Phase 1: Load data (5-20%)
                if not self.equity_curve:  # Only load if not resuming
                    self.current_phase = "Loading Data"
                    self.progress = 5.0
                    ohlcv_data = await self._load_data(config)
                else:
                    # Use cached data from resume_state
                    ohlcv_data = config["resume_state"]["ohlcv_data"]

                # Phase 2: Run simulation (20-95%)
                await self._run_simulation(config, ohlcv_data)

            # Phase 3: Persist results (95-100%)
            self.current_phase = "Saving Results"
            self.progress = 95.0
            completed_at = workflow.now().isoformat()
            final_results = await self._persist_results(completed_at)

            # Complete
            self.status = "completed"
            self.progress = 100.0
            self.completed_at = completed_at

            workflow.logger.info(f"Backtest {self.run_id} completed successfully")

            return final_results

        except Exception as e:
            workflow.logger.error(f"Backtest {self.run_id} failed: {e}")
            self.status = "failed"
            self.error = str(e)
            self.completed_at = workflow.now().isoformat()
            raise

    async def _load_data(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Load OHLCV data via activity.

        Returns:
            Dict mapping symbol -> {data, total_candles}
        """
        workflow.logger.info("Loading OHLCV data")

        ohlcv_data = await workflow.execute_activity(
            load_ohlcv_activity,
            args=[config],
            schedule_to_close_timeout=timedelta(seconds=60),
            retry_policy=RetryPolicy(
                maximum_attempts=3,
                initial_interval=timedelta(seconds=1),
                maximum_interval=timedelta(seconds=10),
            ),
        )

        # Update total candles
        if ohlcv_data:
            first_symbol = config["symbols"][0]
            self.candles_total = ohlcv_data[first_symbol]["total_candles"]
            workflow.logger.info(f"Loaded {self.candles_total} candles")

        self.progress = 20.0
        return ohlcv_data

    async def _run_simulation(
        self,
        config: Dict[str, Any],
        ohlcv_data: Dict[str, Any]
    ) -> None:
        """Run simulation with continue-as-new for large backtests.

        Args:
            config: Backtest configuration
            ohlcv_data: Loaded OHLCV data
        """
        self.current_phase = "Simulating"
        chunk_size = 500  # Continue-as-new threshold (reduced for better progress granularity)

        workflow.logger.info(f"Running simulation (offset={self.candles_processed}, chunk_size={chunk_size})")

        # Execute simulation activity
        result = await workflow.execute_activity(
            run_simulation_chunk_activity,
            args=[config, ohlcv_data, self.candles_processed, chunk_size],
            schedule_to_close_timeout=timedelta(minutes=15),
            heartbeat_timeout=timedelta(seconds=30),
            retry_policy=RetryPolicy(maximum_attempts=1),  # No retry (deterministic)
        )

        # Update state from activity
        self.equity_curve.extend(result["equity_curve"])
        self.trades.extend(result["trades"])
        self.candles_processed += result["candles_processed"]
        self.summary = result.get("summary", {})

        # Update progress
        if self.candles_total > 0:
            self.progress = 20 + (self.candles_processed / self.candles_total) * 75

        workflow.logger.info(
            f"Chunk complete: {self.candles_processed}/{self.candles_total} candles, "
            f"{len(self.trades)} trades"
        )

        # Continue-as-new if more data remains
        if result.get("has_more", False):
            workflow.logger.info("More data to process, triggering continue-as-new")

            await workflow.continue_as_new(
                args=[{
                    **config,
                    "resume_state": {
                        "equity_curve": self.equity_curve,
                        "trades": self.trades,
                        "candles_processed": self.candles_processed,
                        "summary": self.summary,
                        "ohlcv_data": ohlcv_data,  # Carry forward
                        "config": self._config,  # Preserve config across continue-as-new
                    }
                }]
            )

    async def _run_llm_backtest(self, config: Dict[str, Any]) -> None:
        """Run the LLM strategist backtest in a single activity call."""
        self.current_phase = "Simulating (LLM)"
        self.progress = 20.0
        if self.candles_total == 0:
            try:
                from data_loader.utils import ensure_utc, timeframe_to_seconds

                start = config.get("requested_start_date") or config.get("start_date")
                end = config.get("requested_end_date") or config.get("end_date")
                timeframe = self._config.get("timeframe", "1h")
                if start and end and timeframe:
                    start_dt = ensure_utc(datetime.fromisoformat(start))
                    end_dt = ensure_utc(datetime.fromisoformat(end))
                    tf_seconds = timeframe_to_seconds(timeframe)
                    self.candles_total = int((end_dt - start_dt).total_seconds() // tf_seconds) + 1
            except Exception as exc:
                workflow.logger.debug("Failed to estimate candles_total: %s", exc)

        result = await workflow.execute_activity(
            run_llm_backtest_activity,
            args=[config],
            schedule_to_close_timeout=timedelta(minutes=60),
            heartbeat_timeout=timedelta(seconds=30),
            retry_policy=RetryPolicy(maximum_attempts=1),
        )

        self.summary = result.get("summary", {})
        artifact_path = result.get("artifact_path")
        if artifact_path:
            self.llm_data = {"artifact_path": artifact_path, "stored": True}
        else:
            self.llm_data = result.get("llm_data")
        self.equity_curve = result.get("equity_curve", [])
        self.trades = result.get("trades", [])
        self.candles_processed = result.get("candles_processed", 0)
        self.candles_total = result.get("candles_total", self.candles_processed)
        self.progress = 95.0

    async def _persist_results(self, completed_at: str) -> Dict[str, Any]:
        """Persist results to disk and return summary.

        Returns:
            Dict with final backtest metrics
        """
        workflow.logger.info("Persisting results")

        # Calculate final metrics
        final_results = self._calculate_metrics(
            status_override="completed",
            completed_at_override=completed_at,
        )

        # Persist to disk via activity
        await workflow.execute_activity(
            persist_results_activity,
            args=[self.run_id, final_results],
            schedule_to_close_timeout=timedelta(seconds=120),
            retry_policy=RetryPolicy(
                maximum_attempts=3,
                initial_interval=timedelta(seconds=2),
                maximum_interval=timedelta(seconds=10),
            ),
        )

        self.progress = 100.0
        return final_results

    def _restore_state(self, resume_state: Dict[str, Any]) -> None:
        """Restore workflow state from continue-as-new.

        Args:
            resume_state: State dict from previous execution
        """
        self.equity_curve = resume_state.get("equity_curve", [])
        self.trades = resume_state.get("trades", [])
        self.candles_processed = resume_state.get("candles_processed", 0)
        self.summary = resume_state.get("summary", {})
        # Restore config if present
        if resume_state.get("config"):
            self._config = resume_state["config"]

        workflow.logger.info(
            f"Restored state: {len(self.equity_curve)} equity points, "
            f"{len(self.trades)} trades, {self.candles_processed} candles processed"
        )

    def _calculate_metrics(
        self,
        status_override: Optional[str] = None,
        completed_at_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate final backtest metrics.

        Returns:
            Dict with all backtest results and metrics
        """
        import pandas as pd

        # Convert lists back to DataFrames for metric calculation
        equity_df = pd.DataFrame(self.equity_curve) if self.equity_curve else pd.DataFrame()
        trades_df = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()

        # Extract metrics from summary (calculated by simulator)
        metrics = {
            "run_id": self.run_id,
            "status": status_override or self.status,
            "started_at": self.started_at,
            "completed_at": completed_at_override or self.completed_at,
            "error": self.error,

            # Config (needed by playback endpoints)
            "config": self._config,
            "strategy": self._config.get("strategy"),
            "llm_data": self.llm_data,

            # Performance metrics
            "final_equity": self.summary.get("final_equity"),
            "equity_return_pct": self.summary.get("equity_return_pct"),
            "sharpe_ratio": self.summary.get("sharpe_ratio"),
            "max_drawdown_pct": self.summary.get("max_drawdown_pct"),

            # Trading metrics
            "total_trades": self.summary.get("total_trades", len(self.trades)),
            "win_rate": self.summary.get("win_rate"),
            "avg_win": self.summary.get("avg_win"),
            "avg_loss": self.summary.get("avg_loss"),
            "profit_factor": self.summary.get("profit_factor"),

            # Data for detailed queries
            "equity_curve": self.equity_curve,
            "trades": self.trades,
            "candles_total": self.candles_total,
            "candles_processed": self.candles_processed,
        }

        return metrics

    @workflow.signal
    def update_progress(self, progress_data: Dict[str, Any]) -> None:
        """Receive progress updates from activity heartbeats.

        Args:
            progress_data: Dict with progress, candles_processed, timestamp, etc.
        """
        progress = progress_data.get("progress")
        if progress is None:
            progress = progress_data.get("progress_pct")
        if progress is not None:
            self.progress = progress
        self.candles_processed = progress_data.get("candles_processed", self.candles_processed)
        candles_total = progress_data.get("candles_total")
        if candles_total:
            self.candles_total = candles_total
        timestamp = progress_data.get("timestamp") or progress_data.get("current_timestamp")
        if timestamp:
            self.current_timestamp = timestamp
        self.current_phase = progress_data.get("current_phase", self.current_phase)

    @workflow.query
    def get_status(self) -> Dict[str, Any]:
        """Return current backtest status.

        Returns:
            Dict with run_id, status, progress, phase, etc.
        """
        return {
            "run_id": self.run_id,
            "status": self.status,
            "progress": self.progress,
            "current_phase": self.current_phase,
            "candles_total": self.candles_total,
            "candles_processed": self.candles_processed,
            "current_timestamp": self.current_timestamp,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "strategy": self._config.get("strategy"),
            "timeframe": self._config.get("timeframe"),
            "start_date": self._config.get("start_date"),
            "end_date": self._config.get("end_date"),
            "requested_start_date": self._config.get("requested_start_date"),
            "requested_end_date": self._config.get("requested_end_date"),
        }

    @workflow.query
    def get_results(self) -> Dict[str, Any]:
        """Return backtest results (only valid when completed).

        Returns:
            Dict with final metrics, equity curve, trades
        """
        if self.status != "completed":
            return {
                "error": f"Backtest not completed (status: {self.status})",
                "run_id": self.run_id,
                "status": self.status,
            }

        return self._calculate_metrics()

    @workflow.query
    def get_equity_curve(self) -> List[Dict[str, Any]]:
        """Return equity curve data.

        Returns:
            List of {timestamp, equity} dicts
        """
        return self.equity_curve

    @workflow.query
    def get_trades(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Return trade log with pagination.

        Args:
            limit: Max trades to return
            offset: Starting index

        Returns:
            List of trade dicts
        """
        return self.trades[offset:offset + limit]
