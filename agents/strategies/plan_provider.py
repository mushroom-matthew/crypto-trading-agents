"""StrategyPlan caching and LLM-call budgeting."""

from __future__ import annotations

import hashlib
import logging
import math
import os
from uuid import uuid4
from datetime import timezone
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from datetime import datetime
from pathlib import Path

from schemas.llm_strategist import LLMInput, StrategyPlan

from .llm_client import LLMClient
from ops_api.event_store import EventStore
from ops_api.schemas import Event


logger = logging.getLogger(__name__)


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


@lru_cache(maxsize=1)
def _prompt_catalog() -> dict[str, str]:
    """Load known prompt templates and return sha256 hashes keyed by template id."""
    base_dir = Path(__file__).resolve().parents[2] / "prompts"
    catalog: dict[str, str] = {}
    base_prompt = base_dir / "llm_strategist_prompt.txt"
    if base_prompt.exists():
        catalog["default"] = hashlib.sha256(base_prompt.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
    strategies_dir = base_dir / "strategies"
    if strategies_dir.exists():
        for path in sorted(strategies_dir.glob("*.txt")):
            catalog[path.stem] = hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
    return catalog


@lru_cache(maxsize=1)
def _schema_prompt() -> str:
    """Load shared StrategyPlan schema block for all strategist prompts."""

    schema_path = Path(__file__).resolve().parents[2] / "prompts" / "strategy_plan_schema.txt"
    if schema_path.exists():
        return schema_path.read_text(encoding="utf-8").strip()
    return ""


def _resolve_prompt_template(prompt_template: str | None) -> str:
    """Resolve the actual prompt template using the same fallback order as LLMClient."""
    if prompt_template:
        base = prompt_template
    else:
        env_prompt = os.environ.get("LLM_STRATEGIST_PROMPT")
        if env_prompt:
            base = env_prompt
        else:
            base_prompt = Path(__file__).resolve().parents[2] / "prompts" / "llm_strategist_prompt.txt"
            base = base_prompt.read_text(encoding="utf-8").strip() if base_prompt.exists() else ""
    schema = _schema_prompt()
    if schema and schema not in base:
        return f"{base}\n\n{schema}"
    return base


def _prompt_metadata(prompt_template: str | None) -> dict[str, object]:
    """Return prompt template metadata for logging/telemetry."""
    resolved = _resolve_prompt_template(prompt_template)
    if not resolved:
        return {}
    resolved_hash = hashlib.sha256(resolved.encode("utf-8")).hexdigest()
    template_id = None
    for candidate_id, candidate_hash in _prompt_catalog().items():
        if candidate_hash == resolved_hash:
            template_id = candidate_id
            break
    return {
        "prompt_template_id": template_id,
        "prompt_template_hash": resolved_hash,
        "prompt_template_chars": len(resolved),
    }


def _context_flags(llm_input: LLMInput) -> dict[str, object]:
    context = llm_input.global_context or {}
    return {
        "has_market_structure": bool(llm_input.market_structure),
        "has_market_structure_context": bool(context.get("market_structure")),
        "has_factor_exposures": bool(context.get("factor_exposures")),
        "has_rpr_comparison": bool(context.get("rpr_comparison")),
        "has_judge_feedback": bool(context.get("judge_feedback")),
        "has_strategy_memory": bool(context.get("strategy_memory")),
        "has_risk_adjustments": bool(context.get("risk_adjustments")),
        "auto_hedge_mode": context.get("auto_hedge_mode"),
    }


@dataclass
class LLMCostTracker:
    input_tokens: int = 0
    output_tokens: int = 0
    num_calls: int = 0
    cost_input_per_1k: float = 0.01
    cost_output_per_1k: float = 0.03

    def record(self, prompt_json: str, completion_json: str) -> None:
        self.input_tokens += _estimate_tokens(prompt_json)
        self.output_tokens += _estimate_tokens(completion_json)
        self.num_calls += 1

    @property
    def estimated_cost(self) -> float:
        return (
            (self.input_tokens / 1000.0) * self.cost_input_per_1k
            + (self.output_tokens / 1000.0) * self.cost_output_per_1k
        )

    def snapshot(self) -> dict[str, float]:
        return {
            "num_llm_calls": self.num_calls,
            "tokens_in": self.input_tokens,
            "tokens_out": self.output_tokens,
            "estimated_cost_usd": round(self.estimated_cost, 4),
        }


class StrategyPlanProvider:
    """Loads cached plans or requests fresh ones within a daily call budget."""

    def __init__(self, llm_client: LLMClient, cache_dir: Path, llm_calls_per_day: int = 1) -> None:
        self.llm_client = llm_client
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.llm_calls_per_day = llm_calls_per_day
        self.daily_counts = defaultdict(int)
        self.cost_tracker = LLMCostTracker()
        self.last_generation_info: dict[str, object] = {}
        self._event_store = EventStore()

    def _cache_path(self, run_id: str, plan_date: datetime, llm_input: LLMInput) -> Path:
        digest = hashlib.sha256(llm_input.to_json().encode("utf-8")).hexdigest()
        date_bucket = plan_date.strftime("%Y-%m-%d")
        return self.cache_dir / run_id / date_bucket / f"{digest}.json"

    def _load_cached(self, path: Path) -> StrategyPlan | None:
        if not path.exists():
            return None
        try:
            return StrategyPlan.model_validate_json(path.read_text())
        except Exception as exc:
            logger.warning("Cached plan failed validation (%s); ignoring cache %s", exc, path)
            return None

    def get_plan(
        self,
        run_id: str,
        plan_date: datetime,
        llm_input: LLMInput,
        prompt_template: str | None = None,
        use_vector_store: bool = False,
        event_ts: datetime | None = None,
        emit_events: bool = True,
    ) -> StrategyPlan:
        cache_path = self._cache_path(run_id, plan_date, llm_input)
        emit_ts = event_ts or plan_date
        cached = self._load_cached(cache_path)
        if cached:
            plan = self._enrich_plan(cached, llm_input)
            plan = plan.model_copy(update={"run_id": run_id})
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(plan.to_json(indent=2))
            self.last_generation_info = {
                "source": "cache",
                "fallback_plan_used": False,
                "llm_failed_parse": False,
                "llm_failure_reason": None,
                "raw_output": None,
            }
            object.__setattr__(plan, "_llm_meta", self.last_generation_info)
            if emit_events:
                self._emit_plan_generated(plan, llm_input, run_id, event_ts=emit_ts)
            return plan
        date_key = (run_id, plan_date.strftime("%Y-%m-%d"))
        if self.daily_counts[date_key] >= self.llm_calls_per_day:
            raise RuntimeError(f"LLM call budget exhausted for {date_key[1]}")
        resolved_prompt = _resolve_prompt_template(prompt_template)
        input_hash = hashlib.sha256(llm_input.to_json().encode("utf-8")).hexdigest()
        metadata = self._llm_call_metadata(llm_input, plan_date, prompt_template=resolved_prompt)
        plan = self.llm_client.generate_plan(
            llm_input,
            prompt_template=resolved_prompt,
            run_id=run_id,
            prompt_hash=input_hash,
            metadata=metadata,
            use_vector_store=use_vector_store,
            event_ts=emit_ts,
        )
        plan = self._enrich_plan(plan, llm_input)
        plan = plan.model_copy(update={"run_id": run_id})
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(plan.to_json(indent=2))
        self.daily_counts[date_key] += 1
        self.cost_tracker.record(llm_input.to_json(), plan.to_json())
        meta = getattr(self.llm_client, "last_generation_info", {}) or {}
        self.last_generation_info = meta
        object.__setattr__(plan, "_llm_meta", meta)
        if emit_events:
            self._emit_plan_generated(plan, llm_input, run_id, event_ts=emit_ts)
        return plan

    def _enrich_plan(self, plan: StrategyPlan, llm_input: LLMInput) -> StrategyPlan:
        """Ensure plan defaults (limits and allowed sets) are populated."""

        plan = plan.model_copy(deep=True)
        universe = sorted({asset.symbol for asset in llm_input.assets})
        default_max_trades = int(os.environ.get("STRATEGIST_PLAN_DEFAULT_MAX_TRADES", "10"))
        default_max_triggers = int(os.environ.get("STRATEGIST_PLAN_DEFAULT_MAX_TRIGGERS_PER_SYMBOL", str(default_max_trades)))
        strict_fixed_caps = os.environ.get("STRATEGIST_STRICT_FIXED_CAPS", "false").lower() == "true"
        legacy_cap_floor = os.environ.get("STRATEGIST_PLAN_LEGACY_DERIVED_CAP_FLOOR", "false").lower() == "true"
        try:
            min_cap_floor = int(os.environ.get("STRATEGIST_PLAN_DERIVED_CAP_MIN_FLOOR", "8"))
        except ValueError:
            min_cap_floor = 8

        if plan.max_trades_per_day is None:
            plan.max_trades_per_day = default_max_trades
        if plan.max_triggers_per_symbol_per_day is None:
            plan.max_triggers_per_symbol_per_day = default_max_triggers
        if strict_fixed_caps:
            if plan.max_trades_per_day is not None and plan.max_trades_per_day < default_max_trades:
                plan.max_trades_per_day = default_max_trades
            if (
                plan.max_triggers_per_symbol_per_day is not None
                and plan.max_triggers_per_symbol_per_day < default_max_triggers
            ):
                plan.max_triggers_per_symbol_per_day = default_max_triggers

        derived_cap = None
        derived_trigger_cap = None
        cap_inputs: dict[str, float] | None = None
        if plan.risk_constraints is not None and plan.risk_constraints.max_daily_risk_budget_pct is not None:
            budget_pct = plan.risk_constraints.max_daily_risk_budget_pct
            max_position_risk = plan.risk_constraints.max_position_risk_pct or 0
            # Use the smallest non-zero sizing target as per-trade risk proxy; fall back to max_position_risk_pct.
            target_risks = [rule.target_risk_pct for rule in plan.sizing_rules if rule.target_risk_pct and rule.target_risk_pct > 0]
            per_trade_risk = min(target_risks) if target_risks else max_position_risk
            if max_position_risk > 0:
                per_trade_risk = min(per_trade_risk, max_position_risk)
            if per_trade_risk and per_trade_risk > 0:
                per_trade_risk_for_cap = min(per_trade_risk, budget_pct) if budget_pct else per_trade_risk
                if legacy_cap_floor:
                    derived_cap = max(min_cap_floor, math.ceil(budget_pct / per_trade_risk_for_cap))
                else:
                    derived_cap = max(1, math.floor(budget_pct / per_trade_risk_for_cap))
                    if min_cap_floor > 0 and min_cap_floor * per_trade_risk_for_cap <= budget_pct:
                        derived_cap = max(derived_cap, min_cap_floor)
                derived_trigger_cap = derived_cap
                cap_inputs = {"risk_budget_pct": budget_pct, "per_trade_risk_pct": per_trade_risk_for_cap}
                if not strict_fixed_caps:
                    plan.max_trades_per_day = min(plan.max_trades_per_day or derived_cap, derived_cap)
                    plan.max_triggers_per_symbol_per_day = min(plan.max_triggers_per_symbol_per_day or derived_trigger_cap, derived_trigger_cap)
        if not plan.allowed_symbols:
            plan.allowed_symbols = universe
        normalized_triggers = []
        exit_present = False
        valid_directions = {"long", "short", "exit"}
        dead_suffixes = ("_exit_exit", "_exit_flat")
        for trigger in plan.triggers:
            if any(trigger.id.endswith(suffix) for suffix in dead_suffixes):
                logger.info("Pruning trigger %s due to dead suffix", trigger.id)
                continue
            direction = (trigger.direction or "").lower()
            if direction in {"flat", "flat_exit"}:
                direction = "exit"
            if direction not in valid_directions:
                raise ValueError(f"Unsupported trigger direction '{trigger.direction}' for trigger {trigger.id}")
            if direction == "exit":
                exit_present = True
            confidence = trigger.confidence_grade
            if confidence is None:
                confidence = "A" if trigger.category == "emergency_exit" else "B"
            normalized_triggers.append(
                trigger.model_copy(update={"direction": direction, "confidence_grade": confidence})
            )
        plan.triggers = normalized_triggers
        # Prune triggers that would be blocked by allowed_directions immediately.
        plan.triggers = [tr for tr in plan.triggers if tr.direction in valid_directions]
        # Bias toward the dominant 1h regime: thin non-1h duplicates.
        triggers_by_key: dict[tuple[str, str], list[TriggerCondition]] = defaultdict(list)
        for trig in plan.triggers:
            triggers_by_key[(trig.symbol, trig.timeframe)].append(trig)
        trimmed_triggers: list[TriggerCondition] = []
        for (symbol, timeframe), trig_list in triggers_by_key.items():
            if timeframe != "1h" and len(trig_list) > 1:
                trig_list = trig_list[:1]
            trimmed_triggers.extend(trig_list)
        plan.triggers = trimmed_triggers
        if not plan.allowed_directions:
            plan.allowed_directions = ["long", "short"]
        else:
            allowed = {direction.lower() for direction in plan.allowed_directions if direction and direction.lower() in valid_directions}
            if not allowed:
                raise ValueError("StrategyPlan allowed_directions empty after normalization")
            plan.allowed_directions = sorted(allowed)
        if exit_present and "exit" not in plan.allowed_directions:
            plan.allowed_directions.append("exit")
        plan.allowed_directions = sorted(set(plan.allowed_directions))
        for trigger in plan.triggers:
            if trigger.direction not in plan.allowed_directions:
                raise ValueError(f"Trigger {trigger.id} direction {trigger.direction} not permitted by allowed_directions")
        if derived_cap is not None:
            # Attach derived caps for downstream logging/inspection (not persisted in model_dump).
            object.__setattr__(plan, "_derived_trade_cap", derived_cap)
        if derived_trigger_cap is not None:
            object.__setattr__(plan, "_derived_trigger_cap", derived_trigger_cap)
        if cap_inputs:
            object.__setattr__(plan, "_cap_inputs", cap_inputs)
        if not plan.allowed_trigger_categories:
            plan.allowed_trigger_categories = [
                "trend_continuation",
                "mean_reversion",
                "volatility_breakout",
                "reversal",
                "emergency_exit",
                "other",
            ]
        else:
            plan.allowed_trigger_categories = sorted({category for category in plan.allowed_trigger_categories if category})
        if any(trigger.category is None for trigger in plan.triggers):
            if "other" not in plan.allowed_trigger_categories:
                plan.allowed_trigger_categories.append("other")
        return plan

    def _llm_call_metadata(
        self,
        llm_input: LLMInput,
        plan_date: datetime,
        prompt_template: str | None = None,
    ) -> dict[str, object]:
        symbols = sorted({asset.symbol for asset in llm_input.assets})
        timeframes = sorted(
            {
                indicator.timeframe
                for asset in llm_input.assets
                for indicator in asset.indicators
                if indicator.timeframe
            }
        )
        plan_dt = plan_date if plan_date.tzinfo else plan_date.replace(tzinfo=timezone.utc)
        return {
            "plan_date": plan_dt.astimezone(timezone.utc).isoformat(),
            "symbols": symbols,
            "timeframes": timeframes,
            "asset_count": len(symbols),
            "timeframe_count": len(timeframes),
            "context_flags": _context_flags(llm_input),
            **_prompt_metadata(prompt_template),
        }

    def _emit_plan_generated(
        self,
        plan: StrategyPlan,
        llm_input: LLMInput,
        run_id: str,
        *,
        event_ts: datetime | None = None,
    ) -> None:
        try:
            trigger_summary = [
                {
                    "id": trig.id,
                    "symbol": trig.symbol,
                    "direction": trig.direction,
                    "timeframe": trig.timeframe,
                    "category": trig.category,
                    "confidence": trig.confidence_grade,
                }
                for trig in plan.triggers[:50]
            ]
            payload = {
                "plan_id": plan.plan_id,
                "run_id": run_id,
                "generated_at": plan.generated_at.isoformat(),
                "valid_until": plan.valid_until.isoformat(),
                "regime": plan.regime,
                "num_triggers": len(plan.triggers),
                "triggers": trigger_summary,
                "allowed_symbols": plan.allowed_symbols,
                "allowed_directions": plan.allowed_directions,
                "max_trades_per_day": plan.max_trades_per_day,
                "min_trades_per_day": plan.min_trades_per_day,
                "max_triggers_per_symbol_per_day": plan.max_triggers_per_symbol_per_day,
                "risk_constraints": plan.risk_constraints.model_dump() if plan.risk_constraints else None,
                "context_flags": _context_flags(llm_input),
                "source": (self.last_generation_info or {}).get("source"),
                "llm_meta": self.last_generation_info,
            }
            ts = event_ts
            if ts is None:
                ts = plan.generated_at
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            event = Event(
                event_id=str(uuid4()),
                ts=ts,
                source="llm_strategist",
                type="plan_generated",  # type: ignore[arg-type]
                payload=payload,
                dedupe_key=None,
                run_id=run_id,
                correlation_id=plan.plan_id,
            )
            self._event_store.append(event)
        except Exception:
            logger.debug("Failed to emit plan_generated event", exc_info=True)
