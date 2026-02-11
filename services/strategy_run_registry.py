"""File-backed registry for StrategyRun objects."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

from schemas.strategy_run import StrategyRun, StrategyRunConfig


class StrategyRunRegistry:
    """Lightweight persistence layer for strategy runs."""

    def __init__(self, storage_dir: Path | None = None) -> None:
        self.storage_dir = storage_dir or Path("data/strategy_runs")
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, run_id: str) -> Path:
        return self.storage_dir / f"{run_id}.json"

    def _persist(self, run: StrategyRun) -> StrategyRun:
        run.updated_at = datetime.now(timezone.utc)
        path = self._path(run.run_id)
        path.write_text(run.model_dump_json(indent=2), encoding="utf-8")
        return run

    def create_strategy_run(self, config: StrategyRunConfig, run_id: Optional[str] = None) -> StrategyRun:
        run_id = run_id or f"run_{uuid4().hex}"
        now = datetime.now(timezone.utc)
        run = StrategyRun(
            run_id=run_id,
            config=config,
            created_at=now,
            updated_at=now,
        )
        return self._persist(run)

    def get_strategy_run(self, run_id: str) -> StrategyRun:
        path = self._path(run_id)
        if not path.exists():
            raise KeyError(f"StrategyRun {run_id} not found")
        with path.open("r", encoding="utf-8") as f:
            return StrategyRun.model_validate_json(f.read())

    def update_strategy_run(self, run: StrategyRun) -> StrategyRun:
        stored = self.get_strategy_run(run.run_id)
        if stored.is_locked:
            comparable_fields = ("current_plan_id", "latest_judge_feedback", "latest_judge_action", "config")
            for field in comparable_fields:
                if getattr(run, field) != getattr(stored, field):
                    raise ValueError(f"StrategyRun {run.run_id} is locked and cannot be modified")
            run.is_locked = True
        run.created_at = stored.created_at
        return self._persist(run)

    def lock_run(self, run_id: str) -> StrategyRun:
        run = self.get_strategy_run(run_id)
        if run.is_locked:
            return run
        run.is_locked = True
        return self.update_strategy_run(run)


registry = StrategyRunRegistry()
strategy_run_registry = registry
