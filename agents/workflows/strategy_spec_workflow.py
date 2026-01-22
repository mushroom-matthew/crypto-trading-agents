"""Workflow for storing and retrieving deterministic strategy specs."""

from __future__ import annotations

from typing import Any, Dict, Optional

from temporalio import workflow
from pydantic import ValidationError

from tools.strategy_spec import StrategySpec, serialize_strategy


@workflow.defn
class StrategySpecWorkflow:
    """Temporal workflow storing StrategySpec objects keyed by market/timeframe."""

    def __init__(self) -> None:
        self.specs: Dict[str, StrategySpec] = {}
        self.strategy_index: Dict[str, str] = {}  # strategy_id -> spec key

    def _store_spec(self, spec: StrategySpec) -> None:
        spec_key = spec.spec_key()
        self.specs[spec_key] = spec
        self.strategy_index[spec.strategy_id] = spec_key
        workflow.logger.info(
            "StrategySpec stored",
            extra={"strategy_id": spec.strategy_id, "market": spec.market, "timeframe": spec.timeframe},
        )

    def _get_spec(self, market: str, timeframe: Optional[str] = None) -> Optional[StrategySpec]:
        if timeframe:
            return self.specs.get(f"{market}:{timeframe}")
        # Return newest spec for market if timeframe omitted
        if not market:
            return None
        for key in sorted(self.specs.keys(), reverse=True):
            if key.startswith(f"{market}:"):
                return self.specs[key]
        return None

    @workflow.signal
    def update_strategy_spec(self, spec_data: Dict[str, Any]) -> None:
        """Create or update a strategy spec entry."""
        try:
            spec = StrategySpec.model_validate(spec_data)
        except ValidationError as exc:
            workflow.logger.error("Invalid StrategySpec payload", exc_info=exc)
            return
        self._store_spec(spec)

    @workflow.signal
    def delete_strategy_spec(self, market: str, timeframe: Optional[str] = None) -> None:
        """Delete a stored spec."""
        spec = self._get_spec(market, timeframe)
        if not spec:
            return
        key = spec.spec_key()
        self.specs.pop(key, None)
        if spec.strategy_id in self.strategy_index:
            del self.strategy_index[spec.strategy_id]
        workflow.logger.info(
            "StrategySpec removed",
            extra={"strategy_id": spec.strategy_id, "market": spec.market, "timeframe": spec.timeframe},
        )

    @workflow.query
    def get_strategy_spec(self, market: str, timeframe: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Return serialized strategy spec for a market/timeframe."""
        spec = self._get_spec(market, timeframe)
        return serialize_strategy(spec) if spec else None

    @workflow.query
    def get_strategy_by_id(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """Return spec by unique strategy id."""
        spec_key = self.strategy_index.get(strategy_id)
        if not spec_key:
            return None
        spec = self.specs.get(spec_key)
        return serialize_strategy(spec) if spec else None

    @workflow.query
    def list_strategy_specs(self) -> Dict[str, Dict[str, Any]]:
        """Return all specs keyed by market/timeframe string."""
        return {key: serialize_strategy(spec) for key, spec in self.specs.items()}

    @workflow.run
    async def run(self) -> None:
        await workflow.wait_condition(lambda: False)
