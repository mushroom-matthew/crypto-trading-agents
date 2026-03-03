"""Paper trading workflow and activities for live strategy execution."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, ValidationError, field_validator
from temporalio import activity, workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from agents.constants import (
        MOCK_LEDGER_WORKFLOW_ID,
        STREAM_CONTINUE_EVERY,
        STREAM_HISTORY_LIMIT,
    )
    from agents.strategies.llm_client import LLMClient
    from agents.strategies.trigger_engine import Bar, Order, TriggerEngine
    from agents.strategies.plan_validator import validate_trigger_plan
    from agents.strategies.risk_engine import RiskEngine, RiskProfile
    from agents.strategies.trade_risk import TradeRiskEvaluator
    from agents.analytics import (
        IndicatorWindowConfig,
        build_asset_state,
        compute_indicator_snapshot,
        compute_htf_structural_fields,
        compute_portfolio_state,
    )
    from schemas.llm_strategist import (
        AssetState,
        IndicatorSnapshot,
        LLMInput,
        PortfolioState,
        StrategyPlan,
    )
    from schemas.research_budget import ResearchBudgetState
    from schemas.experiment_spec import ExperimentSpec
    from schemas.position_exit_contract import PositionExitContract, ExitLeg
    from services.exit_contract_builder import build_exit_contract
    from trading_core.trigger_compiler import compile_plan
    from ops_api.event_store import EventStore
    from ops_api.schemas import Event
    # Policy loop cadence (Runbook 61)
    from services.regime_transition_detector import (
        RegimeTransitionDetector,
        RegimeTransitionDetectorState,
        build_regime_fingerprint,
    )
    from services.policy_loop_gate import PolicyLoopGate
    from services.policy_state_machine import PolicyStateMachine
    from schemas.policy_state import PolicyStateMachineRecord
    from schemas.reasoning_cadence import PolicyLoopTriggerEvent

logger = logging.getLogger(__name__)

# Configuration constants
PAPER_TRADING_CONTINUE_EVERY = int(os.environ.get("PAPER_TRADING_CONTINUE_EVERY", "3600"))
PAPER_TRADING_HISTORY_LIMIT = int(os.environ.get("PAPER_TRADING_HISTORY_LIMIT", "9000"))
DEFAULT_PLAN_INTERVAL_HOURS = float(os.environ.get("PAPER_TRADING_PLAN_INTERVAL_HOURS", "4"))
PLAN_CACHE_DIR = Path(os.environ.get("PAPER_TRADING_PLAN_CACHE", ".cache/paper_trading_plans"))


def _timeframe_to_minutes(tf: str) -> int:
    """Convert a Coinbase-style timeframe string to minutes for plan validation."""
    _MAP = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "6h": 360, "1d": 1440}
    return _MAP.get(str(tf).lower(), 60)


def _get_latest_structure_snapshot(
    structure_history: Dict[str, List[Dict[str, Any]]], symbol: str
) -> "Any | None":
    """Return the most recent StructureSnapshot for symbol from session history.

    R64: helper used by _execute_order for structural stop/target candidate selection.
    Returns None if no history is available or validation fails.
    """
    history = structure_history.get(symbol, [])
    if not history:
        return None
    latest = history[-1]
    try:
        from schemas.structure_engine import StructureSnapshot as _StructureSnapshot
        return _StructureSnapshot.model_validate(latest)
    except Exception:
        return None


def _is_missing_stop_validation_error(exc: Exception) -> bool:
    text = str(exc)
    return "must define a stop" in text and "stop_anchor_type" in text


def _missing_stop_repair_instructions(error_text: str) -> str:
    return (
        "REPAIR REQUIRED: The previous plan failed validation because one or more ENTRY triggers "
        "(direction='long' or 'short') omitted a stop.\n"
        "For EVERY entry trigger, you must define exactly one stop mechanism:\n"
        "- Preferred: stop_anchor_type (e.g. 'htf_daily_extreme', 'atr', 'donchian_extreme'), optionally with stop_atr_mult when anchor='atr'\n"
        "- OR: stop_anchor_type='pct' with stop_loss_pct > 0\n"
        "Do not leave stop_anchor_type and stop_loss_pct both null for entry triggers.\n"
        "Return a full valid StrategyPlan JSON; preserve the intended strategy logic while adding stops.\n"
        f"Validation error details:\n{error_text}"
    )


class PaperTradingConfig(BaseModel):
    """Configuration for a paper trading session."""

    session_id: str
    ledger_workflow_id: Optional[str] = None
    symbols: List[str]
    initial_cash: float = 10000.0
    initial_allocations: Optional[Dict[str, float]] = None
    strategy_prompt: Optional[str] = None
    plan_interval_hours: float = DEFAULT_PLAN_INTERVAL_HOURS
    indicator_timeframe: str = Field(
        default="1h",
        description=(
            "OHLCV timeframe used for indicator computation and trigger generation. "
            "Must be a Coinbase-supported granularity: '1m', '5m', '15m', '1h', '6h', '1d'."
        ),
    )

    direction_bias: str = Field(
        default="neutral",
        description="Directional bias for the session: 'long', 'short', or 'neutral'.",
    )

    @field_validator("indicator_timeframe")
    @classmethod
    def validate_indicator_timeframe(cls, v: str) -> str:
        allowed = {"1m", "5m", "15m", "1h", "6h", "1d"}
        if v not in allowed:
            raise ValueError(
                f"indicator_timeframe '{v}' is not supported by Coinbase. "
                f"Allowed values: {sorted(allowed)}"
            )
        return v

    replan_on_day_boundary: bool = True
    enable_symbol_discovery: bool = False
    min_volume_24h: float = 1_000_000
    llm_model: Optional[str] = None
    exit_binding_mode: Literal["none", "category"] = "category"
    conflicting_signal_policy: Literal["ignore", "exit", "reverse", "defer"] = "reverse"
    # Research budget (Runbook 48)
    research_budget_enabled: bool = True
    research_budget_fraction: Optional[float] = None   # None → fall back to env var
    research_max_loss_pct: Optional[float] = None      # % of research capital (None → 50%)


class SessionState(BaseModel):
    """Serializable session state for continue-as-new."""

    session_id: str
    ledger_workflow_id: Optional[str] = None
    symbols: List[str]
    strategy_prompt: Optional[str]
    plan_interval_hours: float
    indicator_timeframe: str = "1h"
    direction_bias: str = "neutral"
    replan_on_day_boundary: bool = True
    current_plan: Optional[Dict[str, Any]] = None
    last_plan_time: Optional[str] = None
    cycle_count: int = 0
    stopped: bool = False
    enable_symbol_discovery: bool = False
    min_volume_24h: float = 1_000_000
    plan_history: List[Dict[str, Any]] = []
    equity_history: List[Dict[str, Any]] = []
    exit_binding_mode: Literal["none", "category"] = "category"
    conflicting_signal_policy: Literal["ignore", "exit", "reverse", "defer"] = "reverse"
    trigger_rule_edits: List[Dict[str, Any]] = []
    # Candle-clock: track last evaluated candle floor per timeframe to avoid
    # re-evaluating the same candle on every 30-second tick.
    last_eval_candle_by_tf: Dict[str, str] = {}
    # Research budget: isolated capital pool for hypothesis testing (Runbook 48)
    research: Optional["ResearchBudgetState"] = None
    active_experiments: List[Dict[str, Any]] = []
    # Bounded indicator/structure snapshot history for UI time-travel lookups.
    structure_history: Dict[str, List[Dict[str, Any]]] = {}
    # Active exit contracts indexed by symbol (Runbook 60 Phase M2).
    exit_contracts: Dict[str, Dict[str, Any]] = {}
    # Policy loop cadence (Runbook 61)
    policy_state_machine_record: Optional[Dict[str, Any]] = None
    regime_detector_state: Optional[Dict[str, Any]] = None
    last_policy_eval_at: Optional[str] = None  # ISO datetime string
    # Adaptive trade management per-position state (Runbook 63)
    adaptive_management_states: Dict[str, Any] = {}
    # In-session episode memory for next plan generation (Runbook 63)
    episode_memory_store_state: List[Dict[str, Any]] = []
    # Exit contract enforcement: tracks which plan opened each position (Runbook 65)
    position_originating_plans: Dict[str, str] = {}  # symbol → plan_id


# ============================================================================
# Activities
# ============================================================================

@activity.defn
async def generate_strategy_plan_activity(
    symbols: List[str],
    portfolio_state: Dict[str, Any],
    strategy_prompt: Optional[str],
    market_context: Dict[str, Any],
    llm_model: Optional[str] = None,
    session_id: Optional[str] = None,
    repair_instructions: Optional[str] = None,
    indicator_timeframe: Optional[str] = None,
    direction_bias: Optional[str] = None,
    preferred_symbol: Optional[str] = None,
    policy_state_machine_record: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Generate a strategy plan using the LLM client.

    Caches plans based on input hash to reduce LLM costs.
    When repair_instructions is provided (after a failed validation), the cache
    is bypassed and the repair prompt is injected into the LLM call.
    """
    # Build cache key from inputs
    cache_key_data = {
        "symbols": sorted(symbols),
        "portfolio": portfolio_state,
        "prompt": strategy_prompt,
        "market": market_context,
    }
    cache_key = hashlib.sha256(json.dumps(cache_key_data, sort_keys=True, default=str).encode()).hexdigest()[:16]

    # Check cache — skip when repair_instructions are provided (plan was rejected)
    if not repair_instructions:
        PLAN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = PLAN_CACHE_DIR / f"plan_{cache_key}.json"
        if cache_file.exists():
            try:
                cached = json.loads(cache_file.read_text())
                logger.info(f"Using cached strategy plan: {cache_key}")
                return cached
            except Exception as e:
                logger.warning(f"Failed to load cached plan: {e}")

    # Build LLM input
    now = datetime.now(timezone.utc)
    assets = []
    for symbol in symbols:
        ctx = market_context.get(symbol, {})
        # If the context came from fetch_indicator_snapshots_activity it has the
        # full IndicatorSnapshot field set.  Reconstruct directly to preserve all
        # indicators (RSI, SMA, ATR, Bollinger, candlestick, HTF anchors, etc.).
        # Fall back to the minimal subset for backwards-compat (e.g. cold-start).
        if ctx.get("close") is not None or ctx.get("as_of") is not None:
            snap_init = {k: v for k, v in ctx.items()
                         if k not in ("trend_state", "vol_state", "price")}
            snap_init.setdefault("symbol", symbol)
            snap_init.setdefault("timeframe", indicator_timeframe or "1h")
            snap_init.setdefault("as_of", now)
            try:
                snapshot = IndicatorSnapshot.model_validate(snap_init)
            except Exception as e:
                logger.warning(f"Full indicator snapshot validation failed for {symbol}, using minimal: {e}")
                close = float(ctx.get("price", ctx.get("close", 0.0)) or 0.0)
                snapshot = IndicatorSnapshot(symbol=symbol, timeframe=indicator_timeframe or "1h", as_of=now, close=close)
        else:
            close = float(ctx.get("price", 0.0) or 0.0)
            snapshot = IndicatorSnapshot(symbol=symbol, timeframe=indicator_timeframe or "1h", as_of=now, close=close)
        assets.append(build_asset_state(symbol, [snapshot]))

    positions_raw = portfolio_state.get("positions", {})
    portfolio = PortfolioState(
        timestamp=now,
        equity=float(portfolio_state.get("total_equity", portfolio_state.get("equity", 10000.0))),
        cash=float(portfolio_state.get("cash", 10000.0)),
        positions={k: float(v) for k, v in positions_raw.items() if isinstance(v, (int, float))},
        realized_pnl_7d=float(portfolio_state.get("realized_pnl", 0.0)),
        realized_pnl_30d=float(portfolio_state.get("realized_pnl", 0.0)),
        sharpe_30d=0.0,
        max_drawdown_90d=0.0,
        win_rate_30d=0.0,
        profit_factor_30d=1.0,
    )

    llm_input = LLMInput(
        portfolio=portfolio,
        assets=assets,
        risk_params={},
    )

    # Step 4: Symbol-local payload shaping — slim non-selected assets to compact
    # summaries (~18 fields vs 70+) to reduce token cost on multi-symbol calls.
    # Auto-activate for single-symbol sessions; use explicit preferred_symbol when
    # a screener recommendation has selected a focal instrument.
    _focal = preferred_symbol or (symbols[0] if len(symbols) == 1 else None)
    if _focal and len(assets) > 1:
        llm_input = llm_input.slim_for_symbol(_focal)
        logger.debug(
            "Payload shaping active: full indicators for %s, compact for %d other asset(s)",
            _focal,
            len(assets) - 1,
        )

    # Generate plan — inject repair instructions into the prompt if provided
    effective_prompt = strategy_prompt
    if repair_instructions:
        logger.warning(f"Generating repair plan with validation error context")
        effective_prompt = f"{repair_instructions}\n\n{strategy_prompt or ''}"

    # Inject a timeframe hint so the LLM generates triggers on the correct candle period
    tf = indicator_timeframe or "1h"
    timeframe_hint = (
        f"\nTIMEFRAME: Use '{tf}' as the timeframe for all triggers in this plan. "
        f"The indicator snapshot was computed on {tf} candles. "
        f"All entry_rule and exit_rule expressions refer to {tf}-close values.\n"
    )
    effective_prompt = (effective_prompt or "") + timeframe_hint

    # R59: Inject direction bias hint so the LLM generates direction-aligned triggers
    _dir = (direction_bias or "neutral").strip().lower()
    if _dir in {"long", "short"}:
        _dir_str = "LONG (upside breakout / buy)" if _dir == "long" else "SHORT (downside break / sell)"
        effective_prompt += (
            f"\nDIRECTION: {_dir_str}. All entry triggers must align with this direction. "
            f"Do NOT generate entry triggers for the opposite direction.\n"
            f"REQUIRED: Every entry trigger must specify a target_anchor_type or numeric target_price_abs.\n"
        )

    # A5 / R63: Inject episode memory context so the LLM can learn from prior outcomes.
    # Loads both DB records and in-session records threaded via market_context.
    # R66: _validation_bundle is captured for the primary symbol to use in judge gate.
    _validation_bundle = None
    if not repair_instructions:  # skip memory injection in repair passes
        try:
            from services.episode_memory_service import EpisodeMemoryStore
            from services.memory_retrieval_service import MemoryRetrievalService
            from schemas.episode_memory import MemoryRetrievalRequest, EpisodeMemoryRecord

            _mem_store = EpisodeMemoryStore()
            # R63: load in-session episode records from market_context (no DB needed)
            _insession_records = market_context.pop("__episode_memory_store_state__", None) or []
            for _ep_dict in _insession_records:
                try:
                    _ep_dict.setdefault("outcome_class", "neutral")
                    _ep_dict.setdefault("episode_id", str(uuid4()))
                    _ep_dict.setdefault("symbol", "")
                    _mem_store.add(EpisodeMemoryRecord.model_validate(_ep_dict))
                except Exception:
                    pass

            _memory_lines: List[str] = []
            for _sym in (symbols or [])[:3]:  # limit to first 3 symbols
                _recent = _mem_store.load_recent(_sym, limit=30)
                if not _recent:
                    # Fall back to in-session records if DB has nothing
                    _recent = _mem_store.get_by_symbol(_sym)
                if not _recent:
                    continue
                for _r in _recent:
                    _mem_store.add(_r)
                _req = MemoryRetrievalRequest(symbol=_sym, regime_fingerprint={})
                _bundle = MemoryRetrievalService(_mem_store).retrieve(_req)
                # R66: capture primary-symbol bundle for judge validation gate
                if _sym == symbols[0] and _validation_bundle is None:
                    _validation_bundle = _bundle
                _wins = _bundle.winning_contexts[:3]
                _losses = _bundle.losing_contexts[:3]
                _failures = _bundle.failure_mode_patterns[:3]
                if _wins or _losses or _failures:
                    _memory_lines.append(f"\n[{_sym}]")
                    if _wins:
                        _memory_lines.append(
                            "  Wins: "
                            + "; ".join(
                                f"r={r.r_achieved:.1f} tf={r.timeframe} playbook={r.playbook_id or 'n/a'}"
                                for r in _wins if r.r_achieved is not None
                            )
                        )
                    if _losses:
                        _memory_lines.append(
                            "  Losses: "
                            + "; ".join(
                                f"r={r.r_achieved:.1f} modes={','.join(r.failure_modes[:2]) or 'n/a'}"
                                for r in _losses if r.r_achieved is not None
                            )
                        )
                    if _failures:
                        _memory_lines.append(
                            "  Failure patterns: "
                            + "; ".join(
                                ",".join(r.failure_modes[:2]) for r in _failures if r.failure_modes
                            )
                        )
            if _memory_lines:
                effective_prompt = (
                    (effective_prompt or "")
                    + "\n\nMEMORY_CONTEXT (recent resolved episodes — calibrate risk accordingly):\n"
                    + "\n".join(_memory_lines)
                    + "\n"
                )
        except Exception as _mem_exc:
            logger.debug("Memory context injection failed (non-fatal): %s", _mem_exc)

    llm_client = LLMClient(model=llm_model) if llm_model else LLMClient()
    try:
        plan = llm_client.generate_plan(
            llm_input,
            prompt_template=effective_prompt,
            run_id=session_id,
            plan_id=str(uuid4()),
        )
    except ValidationError as exc:
        # Schema validation can fail inside LLMClient before the workflow-level
        # validator/repair pass runs. Catch the common missing-stop error and
        # issue one targeted repair request so the activity does not hard-fail.
        if repair_instructions or not _is_missing_stop_validation_error(exc):
            raise
        logger.warning("LLM plan failed missing-stop validation; retrying with targeted repair prompt")
        return await generate_strategy_plan_activity(
            symbols=symbols,
            portfolio_state=portfolio_state,
            strategy_prompt=strategy_prompt,
            market_context=market_context,
            llm_model=llm_model,
            session_id=session_id,
            repair_instructions=_missing_stop_repair_instructions(str(exc)),
            indicator_timeframe=indicator_timeframe,
            direction_bias=direction_bias,
        )

    # R66: Judge validation gate — validate immediately after generation.
    # Skip in repair passes to avoid recursive loops.
    if not repair_instructions:
        try:
            from services.judge_validation_service import JudgePlanValidationService
            from services.judge_revision_loop import JudgePlanRevisionLoopOrchestrator

            # Derive policy state flags from the state machine record
            _psm = policy_state_machine_record or {}
            _current_state = _psm.get("current_state", "IDLE")
            _is_thesis_armed = _current_state == "THESIS_ARMED"
            _is_hold_lock = _current_state == "HOLD_LOCK"

            # Derive eligible regime tags from the plan's playbook, if present
            _playbook_regime_tags: Optional[List[str]] = None
            if plan.playbook_id:
                try:
                    from services.playbook_registry import PlaybookRegistry
                    _pb_def = PlaybookRegistry().get(plan.playbook_id)
                    if _pb_def and _pb_def.regime_eligibility.eligible_regimes:
                        _playbook_regime_tags = list(_pb_def.regime_eligibility.eligible_regimes)
                except Exception:
                    pass

            _validator = JudgePlanValidationService()
            _verdict = _validator.validate_plan(
                plan,
                memory_bundle=_validation_bundle,
                playbook_regime_tags=_playbook_regime_tags,
                is_thesis_armed=_is_thesis_armed,
                is_hold_lock=_is_hold_lock,
            )

            if _verdict.decision != "approve":
                logger.warning(
                    "Plan validation %s (plan_id=%s): %s",
                    _verdict.decision,
                    plan.plan_id,
                    "; ".join(_verdict.reasons[:3]),
                )
                _last_revised: list = [None]

                def _revision_callback(revision_request: Any) -> Optional["StrategyPlan"]:
                    repair = "; ".join(revision_request.failing_criteria[:3])
                    try:
                        _rev = llm_client.generate_plan(
                            llm_input,
                            prompt_template=f"REPAIR: {repair}",
                            run_id=session_id,
                            plan_id=str(uuid4()),
                        )
                        _last_revised[0] = _rev
                        return _rev
                    except Exception:
                        return None

                _loop = JudgePlanRevisionLoopOrchestrator(max_revisions=2)
                _revision_result = _loop.run(
                    plan=plan,
                    revision_callback=_revision_callback,
                    memory_bundle=_validation_bundle,
                    playbook_regime_tags=_playbook_regime_tags,
                    is_thesis_armed=_is_thesis_armed,
                    is_hold_lock=_is_hold_lock,
                )

                if _revision_result.accepted_plan_id:
                    plan = _last_revised[0] or plan
                    logger.info(
                        "Plan revision accepted (plan_id=%s, attempts=%d)",
                        _revision_result.accepted_plan_id,
                        _revision_result.revision_attempts,
                    )
                else:
                    logger.warning(
                        "Plan validation stand-down: %s (attempts=%d)",
                        _revision_result.stand_down_reason,
                        _revision_result.revision_attempts,
                    )
                    return None

        except Exception as _validation_exc:
            logger.warning(
                "Judge validation gate failed (non-fatal, proceeding): %s", _validation_exc
            )

    plan_dict = plan.model_dump()
    # Stash retrieval metadata for the workflow to include in plan_generated event.
    # Prefixed with _ so it can be popped before any Pydantic re-validation.
    plan_dict["_retrieved_template_id"] = (llm_client.last_generation_info or {}).get("retrieved_template_id")

    # R49: build PolicySnapshot and thread provenance into plan dict for event telemetry.
    # Prefixed with _ so callers can pop before Pydantic re-validation.
    try:
        from services.market_snapshot_builder import build_policy_snapshot
        _ps = build_policy_snapshot(llm_input, policy_event_type="plan_generation")
        plan_dict["_snapshot_id"] = _ps.provenance.snapshot_id
        plan_dict["_snapshot_hash"] = _ps.provenance.snapshot_hash
        plan_dict["_snapshot_version"] = _ps.provenance.snapshot_version
        plan_dict["_snapshot_kind"] = _ps.provenance.snapshot_kind
        plan_dict["_snapshot_as_of_ts"] = _ps.provenance.as_of_ts.isoformat()
        plan_dict["_snapshot_staleness_seconds"] = _ps.quality.staleness_seconds
        plan_dict["_snapshot_missing_sections"] = _ps.quality.missing_sections
    except Exception:
        logger.debug("Failed to build policy snapshot in paper trading", exc_info=True)

    # Cache the plan (only when not a repair pass — repair plans are one-shot)
    if not repair_instructions:
        PLAN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = PLAN_CACHE_DIR / f"plan_{cache_key}.json"
    try:
        if not repair_instructions:
            cache_file.write_text(json.dumps(plan_dict, default=str))
        logger.info(f"{'Repair plan generated' if repair_instructions else 'Cached strategy plan'}: {cache_key}")
    except Exception as e:
        logger.warning(f"Failed to cache plan: {e}")

    return plan_dict


@activity.defn
def evaluate_triggers_activity(
    plan_dict: Dict[str, Any],
    market_data: Dict[str, Dict[str, Any]],
    portfolio_state: Dict[str, Any],
    exit_binding_mode: str = "category",
    conflicting_signal_policy: str = "reverse",
    position_originating_plans: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Evaluate strategy triggers against current market data.

    Returns {"orders": [...], "events": [...]} where events contains
    trigger_fired and trade_blocked entries for the activity feed.
    """
    try:
        plan = StrategyPlan.model_validate(plan_dict)
    except Exception as e:
        logger.error(f"Failed to validate plan: {e}")
        return {"orders": [], "events": []}

    # Build risk engine
    risk_engine = RiskEngine(
        plan.risk_constraints,
        {rule.symbol: rule for rule in plan.sizing_rules},
        risk_profile=RiskProfile(),
    )

    # Align sizing with execution: resolve stop distance from the same anchored
    # stop logic used when orders are filled.
    resolve_stop_distance = None
    try:
        from backtesting.llm_strategist_runner import _resolve_stop_price_anchored

        def resolve_stop_distance(trigger, indicator, bar) -> float | None:
            direction = trigger.direction if trigger.direction in {"long", "short"} else None
            if direction is None:
                return None
            stop_abs, _ = _resolve_stop_price_anchored(trigger, bar.close, indicator, direction)
            if stop_abs is None:
                return None
            dist = abs(float(bar.close) - float(stop_abs))
            return dist if dist > 0 else None
    except Exception as e:
        logger.warning(f"Anchored stop resolver unavailable in trigger evaluation: {e}")

    # Build trigger engine
    trigger_engine = TriggerEngine(
        plan,
        risk_engine,
        trade_risk=TradeRiskEvaluator(risk_engine),
        stop_distance_resolver=resolve_stop_distance,
        exit_binding_mode=exit_binding_mode if exit_binding_mode else "category",
        conflicting_signal_policy=conflicting_signal_policy if conflicting_signal_policy else "reverse",
        position_originating_plans=position_originating_plans,  # R65
    )

    all_orders: List[Dict[str, Any]] = []
    all_events: List[Dict[str, Any]] = []

    def _parse_timestamp(value: Any) -> datetime:
        if isinstance(value, datetime):
            return value
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value / 1000.0, tz=timezone.utc)
        if isinstance(value, str):
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    return parsed.replace(tzinfo=timezone.utc)
                return parsed
            except ValueError:
                return datetime.now(timezone.utc)
        return datetime.now(timezone.utc)

    positions = portfolio_state.get("positions") or {}
    cash = float(portfolio_state.get("cash", 0.0))
    equity = float(
        portfolio_state.get("total_equity")
        or portfolio_state.get("total_portfolio_value")
        or cash
    )
    snapshot_ts = None
    for data in market_data.values():
        if isinstance(data, dict) and data.get("timestamp"):
            snapshot_ts = _parse_timestamp(data.get("timestamp"))
            break
    if snapshot_ts is None:
        snapshot_ts = datetime.now(timezone.utc)

    portfolio_snapshot = PortfolioState(
        timestamp=snapshot_ts,
        equity=equity,
        cash=cash,
        positions=positions,
        realized_pnl_7d=0.0,
        realized_pnl_30d=0.0,
        sharpe_30d=0.0,
        max_drawdown_90d=0.0,
        win_rate_30d=0.0,
        profit_factor_30d=0.0,
    )
    position_meta = portfolio_state.get("position_meta") if isinstance(portfolio_state, dict) else None

    for symbol, data in market_data.items():
        if not isinstance(data, dict):
            continue
        bar_ts = _parse_timestamp(data.get("timestamp"))
        close_price = float(data.get("close", data.get("price", 0.0)) or 0.0)
        open_price = float(data.get("open", close_price))
        high_price = float(data.get("high", close_price))
        low_price = float(data.get("low", close_price))
        volume = float(data.get("volume", 0.0) or 0.0)
        timeframes = sorted({t.timeframe for t in plan.triggers if t.symbol == symbol})
        if not timeframes:
            timeframes = [str(data.get("timeframe") or "1m")]

        sma_short = data.get("sma_short") or data.get("sma_20")
        sma_medium = data.get("sma_medium") or data.get("sma_50")
        sma_long = data.get("sma_long") or data.get("sma_200")

        for timeframe in timeframes:
            bar = Bar(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=bar_ts,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume,
            )

            # Build full indicator snapshot — data may contain a complete
            # indicator dict from fetch_indicator_snapshots_activity.
            snap_init = {k: v for k, v in data.items()
                         if k not in ("trend_state", "vol_state", "price",
                                      "timestamp", "open", "high", "low", "volume")}
            snap_init.update({
                "symbol": symbol,
                "timeframe": timeframe,
                "as_of": bar_ts,
                "close": close_price,
                "volume": volume,
                "sma_short": snap_init.get("sma_short") or sma_short,
                "sma_medium": snap_init.get("sma_medium") or sma_medium,
                "sma_long": snap_init.get("sma_long") or sma_long,
            })
            try:
                indicator = IndicatorSnapshot.model_validate(snap_init)
            except Exception:
                indicator = IndicatorSnapshot(
                    symbol=symbol, timeframe=timeframe, as_of=bar_ts,
                    close=close_price, volume=volume,
                    rsi_14=data.get("rsi_14"),
                    sma_short=sma_short, sma_medium=sma_medium, sma_long=sma_long,
                    atr_14=data.get("atr_14"),
                    bollinger_upper=data.get("bollinger_upper"),
                    bollinger_lower=data.get("bollinger_lower"),
                    vwap=data.get("vwap"),
                )

            asset_state = build_asset_state(symbol, [indicator])

            # R64: build TickSnapshot per bar so triggers have access to
            # normalized numeric features and snapshot provenance.
            _tick_snapshot = None
            try:
                from services.market_snapshot_builder import build_tick_snapshot as _build_tick_snapshot
                _tick_snapshot = _build_tick_snapshot(indicator)
            except Exception as _snap_exc:
                logger.warning("build_tick_snapshot failed: %s", _snap_exc)

            orders, block_entries = trigger_engine.on_bar(
                bar,
                indicator,
                portfolio_snapshot,
                asset_state=asset_state,
                market_structure=None,
                position_meta=position_meta,
                tick_snapshot=_tick_snapshot,
            )

            # Build a lookup of trigger dicts by id for signal emission.
            _trigger_map: Dict[str, Any] = {
                t.get("id", ""): t for t in plan_dict.get("triggers", []) if isinstance(t, dict)
            }

            for order in orders:
                order_dict = {
                    "symbol": order.symbol,
                    "side": order.side,
                    "quantity": order.quantity,
                    "price": order.price,
                    "timeframe": order.timeframe,
                    "trigger_id": order.reason,
                    "trigger_category": order.trigger_category,
                    "intent": order.intent,
                    "reason": order.reason,
                }

                # A1: Emit a SignalEvent for each entry order and thread signal_id.
                if getattr(order, "intent", None) == "entry":
                    try:
                        from schemas.signal_event import SignalEvent as _SignalEvent
                        from services.signal_ledger_service import (
                            SignalLedgerService as _SLS,
                            compute_regime_snapshot_hash as _hash_snap,
                        )
                        _trig = _trigger_map.get(order.reason or "", {})
                        _direction = "long" if order.side.lower() == "buy" else "short"
                        _close = float(order.price or close_price)
                        # Best-effort stop from trigger pct (replaced by resolved stop on fill)
                        _slp = _trig.get("stop_loss_pct")
                        if _slp:
                            _stop_est = _close * (1 - float(_slp) / 100) if _direction == "long" else _close * (1 + float(_slp) / 100)
                        else:
                            _stop_est = _close * 0.98  # 2% default placeholder
                        _target_est = float(_trig.get("target_price_abs") or 0.0) or (_close * 1.04)
                        _risk = abs(_close - _stop_est)
                        _reward = abs(_target_est - _close)
                        _r_mult = round(_reward / _risk, 2) if _risk > 0 else 0.0
                        _hold_bars = int(_trig.get("estimated_bars_to_resolution") or 4)
                        _thesis = (
                            _trig.get("thesis")
                            or plan_dict.get("global_view")
                            or f"{_direction} signal from {order.reason}"
                        )
                        _snap_hash = _hash_snap({
                            "close": close_price, "symbol": symbol, "timeframe": timeframe,
                            "rsi_14": data.get("rsi_14"), "atr_14": data.get("atr_14"),
                        })
                        _signal = _SignalEvent(
                            engine_version="1.0.0",
                            ts=bar_ts,
                            valid_until=bar_ts + timedelta(hours=_hold_bars),
                            timeframe=timeframe,
                            symbol=order.symbol,
                            direction=_direction,
                            trigger_id=order.reason or "unknown",
                            strategy_type=(
                                order.trigger_category
                                or _trig.get("category")
                                or plan_dict.get("regime", "unknown")
                            ),
                            regime_snapshot_hash=_snap_hash,
                            entry_price=_close,
                            stop_price_abs=_stop_est,
                            target_price_abs=_target_est,
                            risk_r_multiple=_r_mult,
                            expected_hold_bars=_hold_bars,
                            thesis=str(_thesis)[:500],
                            playbook_id=_trig.get("playbook_id"),
                            strategy_template_version=plan_dict.get("template_id"),
                        )
                        order_dict["signal_id"] = _signal.signal_id
                        order_dict["signal_ts"] = bar_ts.isoformat()
                        order_dict["signal_entry_price"] = _close
                        _SLS().insert_signal(_signal)
                    except Exception as _sig_exc:
                        logger.debug("signal_id emission failed (non-fatal): %s", _sig_exc)

                    # R63: SetupEventGenerator — frozen feature snapshot at trigger fire.
                    try:
                        from agents.analytics.setup_event_generator import SetupEventGenerator as _SetupEventGenerator
                        from services.model_scorer import NullModelScorer as _NullModelScorer
                        from backtesting.constants import ENGINE_SEMVER as _ENGINE_SEMVER
                        _seg = _SetupEventGenerator(
                            engine_semver=_ENGINE_SEMVER,
                            scorer=_NullModelScorer(),
                            strategy_template_version=plan_dict.get("template_id"),
                        )
                        _setup_evts = _seg.on_bar(symbol, timeframe, bar_ts, indicator)
                        if _setup_evts:
                            order_dict["setup_event_id"] = _setup_evts[0].setup_event_id
                            order_dict["setup_event"] = _setup_evts[0].model_dump(mode="json")
                    except Exception as _seg_exc:
                        logger.debug("SetupEventGenerator failed (non-fatal): %s", _seg_exc)

                all_orders.append(order_dict)

                # M1 guardrail (Runbook 60): detect non-emergency direction="exit" triggers
                # that fire via the entry-rule flatten path (reason ends in "_flat").
                # These bypass normal exit-rule binding checks and should migrate to
                # PositionExitContract stop/target legs.
                if (
                    getattr(order, "intent", None) == "exit"
                    and isinstance(order.reason, str)
                    and order.reason.endswith("_flat")
                    and order.trigger_category != "emergency_exit"
                ):
                    all_events.append({
                        "type": "entry_rule_flatten_detected",
                        "payload": {
                            "symbol": order.symbol,
                            "trigger_id": order.reason,
                            "category": order.trigger_category,
                            "exit_class": "strategy_contract_candidate",
                            "detail": (
                                "Non-emergency exit trigger fired via entry-rule flatten path "
                                "(reason ends '_flat'). Migrate to PositionExitContract leg."
                            ),
                        },
                    })

                all_events.append({
                    "type": "trigger_fired",
                    "payload": {
                        "symbol": order.symbol,
                        "side": order.side,
                        "trigger_id": order.reason,
                        "category": order.trigger_category,
                        "price": order.price,
                        "timeframe": timeframe,
                    },
                })

            for block in block_entries:
                all_events.append({
                    "type": "trade_blocked",
                    "payload": {
                        "trigger_id": block.get("trigger_id", ""),
                        "symbol": block.get("symbol", symbol),
                        "reason": block.get("reason", ""),
                        "detail": block.get("detail", ""),
                    },
                })

            # Emit a lightweight eval_summary so the UI knows all symbols are
            # being tracked even when no triggers fire or are blocked.
            symbol_trigger_count = sum(
                1 for t in plan.triggers
                if t.symbol == symbol and t.timeframe == timeframe
            )
            symbol_orders = sum(1 for o in all_orders if o["symbol"] == symbol)
            symbol_blocks = sum(1 for e in all_events if e["type"] == "trade_blocked" and e["payload"].get("symbol") == symbol)
            all_events.append({
                "type": "eval_summary",
                "payload": {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "triggers_evaluated": symbol_trigger_count,
                    "fired": symbol_orders,
                    "blocked": symbol_blocks,
                    "price": close_price,
                },
            })

    # R65: emit trade_blocked event if any exits were suppressed by plan mismatch
    mismatch_count = trigger_engine.exit_binding_mismatch_count
    if mismatch_count > 0:
        all_events.append({
            "type": "trade_blocked",
            "payload": {
                "reason": "exit_binding_mismatch",
                "detail": (
                    f"{mismatch_count} exit trigger(s) blocked: current plan_id does not "
                    "match the plan that opened the position (R65 originating_plan_id pinning)"
                ),
                "count": mismatch_count,
            },
        })

    return {
        "orders": all_orders,
        "events": all_events,
        "exit_binding_mismatch_blocked": mismatch_count,
    }


@activity.defn
async def discover_symbols_activity(
    exchange: str = "coinbase",
    min_volume_24h: float = 1_000_000,
    quote_currency: str = "USD",
) -> List[str]:
    """Discover available trading pairs from exchange.

    Filters by 24h volume and quote currency.
    """
    import ccxt.async_support as ccxt

    if exchange == "coinbase":
        client = ccxt.coinbaseexchange()
    else:
        raise ValueError(f"Unsupported exchange: {exchange}")

    try:
        await client.load_markets()

        eligible_symbols = []
        for symbol, market in client.markets.items():
            # Filter by quote currency
            if market.get("quote") != quote_currency:
                continue

            # Filter by active status
            if not market.get("active", True):
                continue

            # Try to get 24h volume
            try:
                ticker = await client.fetch_ticker(symbol)
                volume_24h = ticker.get("quoteVolume", 0) or 0
                if volume_24h >= min_volume_24h:
                    eligible_symbols.append(symbol)
            except Exception:
                continue

        return sorted(eligible_symbols)
    finally:
        await client.close()


@activity.defn
async def fetch_current_prices_activity(symbols: List[str]) -> Dict[str, float]:
    """Fetch current prices for a list of symbols."""
    import ccxt.async_support as ccxt

    client = ccxt.coinbaseexchange()
    prices = {}

    try:
        for symbol in symbols:
            try:
                ticker = await client.fetch_ticker(symbol)
                prices[symbol] = ticker.get("last", 0) or ticker.get("close", 0) or 0
            except Exception as e:
                logger.warning(f"Failed to fetch price for {symbol}: {e}")
        return prices
    finally:
        await client.close()


@activity.defn
async def fetch_indicator_snapshots_activity(
    symbols: List[str],
    timeframe: str = "1h",
    lookback_candles: int = 300,
) -> Dict[str, Any]:
    """Fetch OHLCV history and compute full indicator snapshots for each symbol.

    Returns {symbol: dict} with all IndicatorSnapshot fields, plus
    'trend_state' and 'vol_state' derived from the indicators.  This gives
    the LLM strategist the same macro/micro picture it gets from the backtester.
    """
    _COINBASE_TIMEFRAMES = {"1m", "5m", "15m", "1h", "6h", "1d"}
    if timeframe not in _COINBASE_TIMEFRAMES:
        raise ValueError(
            f"Timeframe '{timeframe}' is not supported by Coinbase. "
            f"Supported: {sorted(_COINBASE_TIMEFRAMES)}"
        )
    import ccxt.async_support as ccxt
    import numpy as np
    import pandas as pd

    client = ccxt.coinbaseexchange()
    client.enableRateLimit = True
    now = datetime.now(timezone.utc)
    results: Dict[str, Any] = {}

    def _ohlcv_to_df(rows: list) -> pd.DataFrame:
        df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["time"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df.set_index("time").drop(columns=["timestamp"])

    try:
        for symbol in symbols:
            try:
                # Fetch intraday history
                ohlcv = await client.fetch_ohlcv(symbol, timeframe, limit=lookback_candles)
                if not ohlcv:
                    logger.warning(f"No OHLCV data for {symbol} {timeframe}")
                    continue
                df = _ohlcv_to_df(ohlcv)

                # Fetch daily bars for HTF structural anchors (Runbook 41)
                daily_ohlcv = await client.fetch_ohlcv(symbol, "1d", limit=30)
                daily_df = _ohlcv_to_df(daily_ohlcv) if daily_ohlcv else None

                # Compute full indicator snapshot (RSI, SMA, ATR, Bollinger, candlestick, etc.)
                config = IndicatorWindowConfig(timeframe=timeframe)
                snapshot = compute_indicator_snapshot(df, symbol, timeframe, config=config, daily_df=daily_df)

                # Apply HTF structural fields (daily anchor layer)
                if daily_df is not None:
                    htf = compute_htf_structural_fields(now, daily_df)
                    if htf:
                        daily_atr = htf.get("htf_daily_atr", 1.0) or 1.0
                        daily_high = htf.get("htf_daily_high", snapshot.close)
                        daily_low = htf.get("htf_daily_low", snapshot.close)
                        daily_mid = (daily_high + daily_low) / 2.0
                        htf["htf_price_vs_daily_mid"] = (snapshot.close - daily_mid) / max(daily_atr, 1e-9)
                        snapshot = snapshot.model_copy(update=htf)

                snap_dict = snapshot.model_dump()
                results[symbol] = snap_dict

            except Exception as e:
                logger.warning(f"Failed to compute indicators for {symbol}: {e}")
    finally:
        await client.close()

    return results


@activity.defn
async def query_ledger_portfolio_activity(ledger_workflow_id: str) -> Dict[str, Any]:
    """Query the execution ledger for current portfolio status (must run as activity for Temporal client access)."""
    from temporalio.client import Client

    address = os.environ.get("TEMPORAL_ADDRESS", "localhost:7233")
    namespace = os.environ.get("TEMPORAL_NAMESPACE", "default")
    client = await Client.connect(address, namespace=namespace)
    handle = client.get_workflow_handle(ledger_workflow_id)
    return await handle.query("get_portfolio_status")


@activity.defn
async def emit_paper_trading_event_activity(
    session_id: str,
    event_type: str,
    payload: Dict[str, Any],
) -> None:
    """Emit an event to the event store."""
    store = EventStore()
    event = Event(
        event_id=str(uuid4()),
        ts=datetime.now(timezone.utc),
        source="paper_trading",
        type=event_type,
        payload=payload,
        run_id=session_id,
    )
    store.append(event)


@activity.defn
async def record_signal_fill_activity(
    signal_id: str,
    fill_price: float,
    fill_ts_iso: str,
    signal_ts_iso: str,
    signal_entry_price: float,
) -> None:
    """Record fill drift telemetry to the signal_ledger table (non-fatal).

    Called after a paper trading fill to capture slippage_bps and fill_latency_ms.
    """
    try:
        from services.signal_ledger_service import SignalLedgerService
        from datetime import datetime as _dt, timezone as _tz

        fill_ts = _dt.fromisoformat(fill_ts_iso)
        signal_ts = _dt.fromisoformat(signal_ts_iso)
        if fill_ts.tzinfo is None:
            fill_ts = fill_ts.replace(tzinfo=_tz.utc)
        if signal_ts.tzinfo is None:
            signal_ts = signal_ts.replace(tzinfo=_tz.utc)
        SignalLedgerService().record_fill(
            signal_id=signal_id,
            fill_price=fill_price,
            fill_ts=fill_ts,
            signal_ts=signal_ts,
            signal_entry_price=signal_entry_price,
        )
    except Exception as exc:
        logger.debug("record_signal_fill_activity failed (non-fatal): %s", exc)


@activity.defn
async def build_episode_activity(
    signal_id: Optional[str],
    symbol: str,
    direction: str,
    entry_price: float,
    fill_ts_iso: str,
    exit_price: float,
    exit_ts_iso: str,
    stop_price_abs: Optional[float],
    target_price_abs: Optional[float],
    hit: str,
    timeframe: str,
    playbook_id: Optional[str] = None,
    template_id: Optional[str] = None,
) -> None:
    """Build and persist an EpisodeMemoryRecord after a position closes.

    Called when _sweep_stop_target detects a stop or target hit.
    """
    try:
        from services.episode_memory_service import EpisodeMemoryStore, build_episode_record
        from schemas.signal_event import SignalEvent as _SE
        from datetime import datetime as _dt, timezone as _tz

        fill_ts = _dt.fromisoformat(fill_ts_iso)
        exit_ts = _dt.fromisoformat(exit_ts_iso)
        if fill_ts.tzinfo is None:
            fill_ts = fill_ts.replace(tzinfo=_tz.utc)
        if exit_ts.tzinfo is None:
            exit_ts = exit_ts.replace(tzinfo=_tz.utc)

        # Compute outcome metrics
        if direction == "long":
            pnl = exit_price - entry_price
            if stop_price_abs and target_price_abs:
                risk = entry_price - stop_price_abs
                reward = exit_price - entry_price
                r_achieved = reward / risk if risk > 0 else 0.0
            else:
                r_achieved = pnl / max(abs(entry_price * 0.02), 1e-6)
            mfe_pct = max(0.0, (exit_price - entry_price) / entry_price * 100) if hit == "target" else 0.0
            mae_pct = min(0.0, (exit_price - entry_price) / entry_price * 100) if hit == "stop" else 0.0
        else:
            pnl = entry_price - exit_price
            if stop_price_abs and target_price_abs:
                risk = stop_price_abs - entry_price
                reward = entry_price - exit_price
                r_achieved = reward / risk if risk > 0 else 0.0
            else:
                r_achieved = pnl / max(abs(entry_price * 0.02), 1e-6)
            mfe_pct = max(0.0, (entry_price - exit_price) / entry_price * 100) if hit == "target" else 0.0
            mae_pct = min(0.0, (entry_price - exit_price) / entry_price * 100) if hit == "stop" else 0.0

        # Build a minimal SignalEvent to feed build_episode_record
        from services.signal_ledger_service import compute_regime_snapshot_hash as _hash
        _snap_hash = _hash({"symbol": symbol, "entry_price": entry_price})
        _hold_bars = max(1, int((exit_ts - fill_ts).total_seconds() / 3600))
        proxy_signal = _SE(
            signal_id=signal_id or str(uuid4()),
            engine_version="1.0.0",
            ts=fill_ts,
            valid_until=exit_ts,
            timeframe=timeframe,
            symbol=symbol,
            direction=direction,
            trigger_id="paper_trading",
            strategy_type="paper_trading",
            regime_snapshot_hash=_snap_hash,
            entry_price=entry_price,
            stop_price_abs=stop_price_abs or (entry_price * 0.98 if direction == "long" else entry_price * 1.02),
            target_price_abs=target_price_abs or (entry_price * 1.04 if direction == "long" else entry_price * 0.96),
            risk_r_multiple=abs(r_achieved),
            expected_hold_bars=_hold_bars,
            thesis=f"{direction} position closed via {hit}",
            playbook_id=playbook_id,
            strategy_template_version=template_id,
        )

        record = build_episode_record(
            signal_event=proxy_signal,
            exit_ts=exit_ts,
            pnl=pnl,
            r_achieved=r_achieved,
            mfe_pct=mfe_pct,
            mae_pct=mae_pct,
            episode_source="paper",
        )
        store = EpisodeMemoryStore()
        store.add(record)
        store.persist_episode(record)
        logger.info(
            "episode_memory: built episode signal_id=%s symbol=%s hit=%s r=%.2f",
            signal_id, symbol, hit, r_achieved,
        )
    except Exception as exc:
        logger.warning("build_episode_activity failed (non-fatal): %s", exc)


# ============================================================================
# Workflow
# ============================================================================

@workflow.defn
class PaperTradingWorkflow:
    """Long-running workflow for paper trading with live market data."""

    def __init__(self) -> None:
        self.session_id: str = ""
        self.ledger_workflow_id: str = ""
        self.symbols: List[str] = []
        self.strategy_prompt: Optional[str] = None
        self.plan_interval_hours: float = DEFAULT_PLAN_INTERVAL_HOURS
        self.indicator_timeframe: str = "1h"
        self.direction_bias: str = "neutral"
        self.replan_on_day_boundary: bool = True
        self.current_plan: Optional[Dict[str, Any]] = None
        self.last_plan_time: Optional[datetime] = None
        self.cycle_count: int = 0
        self.stopped: bool = False
        self.enable_symbol_discovery: bool = False
        self.min_volume_24h: float = 1_000_000
        self.last_discovery_date: Optional[str] = None
        self.plan_history: List[Dict[str, Any]] = []
        self.equity_history: List[Dict[str, Any]] = []
        self.last_equity_snapshot: Optional[datetime] = None
        self.exit_binding_mode: Literal["none", "category"] = "category"
        self.conflicting_signal_policy: Literal["ignore", "exit", "reverse", "defer"] = "reverse"
        self.trigger_rule_edits: List[Dict[str, Any]] = []
        # Indicator snapshots from last plan generation (used in trigger evaluation)
        self.last_indicators: Dict[str, Any] = {}
        self.min_rr_ratio: float = float(os.environ.get("PAPER_TRADING_MIN_RR_RATIO", "1.2"))
        # Live indicator snapshots refreshed on a 1-minute cadence for UI context.
        # This does not affect trigger evaluation; it is visibility-only.
        self.last_indicators_live: Dict[str, Any] = {}
        self.last_live_indicators_refresh: Optional[datetime] = None
        # Bounded per-symbol snapshot history to support UI "selected candle" lookups.
        self.structure_history: Dict[str, List[Dict[str, Any]]] = {}
        # Candle-clock: tracks the last candle floor we ran trigger evaluation on,
        # keyed by timeframe string (e.g. "1h").  Prevents re-evaluating within the
        # same candle and enforces at least one-candle separation between entry and exit.
        self.last_eval_candle_by_tf: Dict[str, str] = {}
        # Research budget (Runbook 48)
        self.research: Optional["ResearchBudgetState"] = None
        self.active_experiments: List[Dict[str, Any]] = []
        # Active exit contracts indexed by symbol (Runbook 60 Phase M2)
        self.exit_contracts: Dict[str, Dict[str, Any]] = {}
        # Policy loop cadence (Runbook 61)
        self.policy_state_machine_record: Dict[str, Any] = {}
        self.regime_detector_state: Optional[Dict[str, Any]] = None
        self.last_policy_eval_at: Optional[str] = None
        self._position_opened_since_last_eval: bool = False
        self._position_closed_since_last_eval: bool = False
        # Adaptive trade management per-position state (Runbook 63)
        self.adaptive_management_states: Dict[str, Any] = {}
        # In-session episode memory for next plan generation (Runbook 63)
        self.episode_memory_store_state: List[Dict[str, Any]] = []
        # Exit contract enforcement: tracks which plan opened each position (Runbook 65)
        self.position_originating_plans: Dict[str, str] = {}  # symbol → plan_id

    # -------------------------------------------------------------------------
    # Signals
    # -------------------------------------------------------------------------

    @workflow.signal
    def update_symbols(self, symbols: List[str]) -> None:
        """Update the list of symbols to trade."""
        self.symbols = list(symbols)
        workflow.logger.info(f"Updated symbols: {self.symbols}")

    @workflow.signal
    def force_replan(self) -> None:
        """Force regeneration of strategy plan."""
        self.last_plan_time = None
        workflow.logger.info("Forcing strategy replan")

    @workflow.signal
    def stop_session(self) -> None:
        """Stop the paper trading session."""
        self.stopped = True
        workflow.logger.info("Stopping paper trading session")

    @workflow.signal
    def update_strategy_prompt(self, prompt: str) -> None:
        """Update the strategy prompt."""
        self.strategy_prompt = prompt
        self.last_plan_time = None  # Force replan
        workflow.logger.info("Updated strategy prompt, forcing replan")

    def _append_trigger_rule_edit(self, entry: Dict[str, Any]) -> None:
        """Append a trigger-rule edit record and keep bounded history."""
        self.trigger_rule_edits.append(entry)
        if len(self.trigger_rule_edits) > 500:
            self.trigger_rule_edits = self.trigger_rule_edits[-500:]

    @workflow.signal
    def update_trigger_rule(self, update: Dict[str, Any]) -> None:
        """Edit trigger rules in the active plan with deterministic validation.

        Expected payload:
            {
                "request_id": str,
                "trigger_id": str,
                "entry_rule": Optional[str],
                "exit_rule": Optional[str],
                "hold_rule": Optional[str],
                "source": str,
                "reason": str,
            }
        """
        request_id = str(update.get("request_id") or f"edit_{len(self.trigger_rule_edits) + 1}")
        trigger_id = str(update.get("trigger_id") or "").strip()
        source = str(update.get("source") or "unknown")
        reason = str(update.get("reason") or "manual_edit")
        changed_fields = [k for k in ("entry_rule", "exit_rule", "hold_rule") if k in update]

        base_record = {
            "request_id": request_id,
            "timestamp": workflow.now().isoformat(),
            "trigger_id": trigger_id,
            "source": source,
            "reason": reason,
            "changed_fields": changed_fields,
        }

        if not self.current_plan:
            self._append_trigger_rule_edit({
                **base_record,
                "status": "rejected",
                "error": "No active plan to edit",
            })
            return
        if not trigger_id:
            self._append_trigger_rule_edit({
                **base_record,
                "status": "rejected",
                "error": "trigger_id is required",
            })
            return
        if not changed_fields:
            self._append_trigger_rule_edit({
                **base_record,
                "status": "rejected",
                "error": "No rule fields provided (entry_rule/exit_rule/hold_rule)",
            })
            return

        raw_triggers = self.current_plan.get("triggers", []) if isinstance(self.current_plan, dict) else []
        if not isinstance(raw_triggers, list):
            self._append_trigger_rule_edit({
                **base_record,
                "status": "rejected",
                "error": "Malformed plan.triggers payload",
            })
            return

        trigger_index = -1
        for idx, trig in enumerate(raw_triggers):
            if isinstance(trig, dict) and str(trig.get("id", "")).strip() == trigger_id:
                trigger_index = idx
                break
        if trigger_index < 0:
            self._append_trigger_rule_edit({
                **base_record,
                "status": "rejected",
                "error": f"Trigger '{trigger_id}' not found in current plan",
            })
            return

        trigger_before = dict(raw_triggers[trigger_index])
        trigger_after = dict(trigger_before)
        for field_name in changed_fields:
            value = update.get(field_name)
            trigger_after[field_name] = "" if value is None else str(value).strip()

        candidate_plan = dict(self.current_plan)
        candidate_triggers = list(raw_triggers)
        candidate_triggers[trigger_index] = trigger_after
        candidate_plan["triggers"] = candidate_triggers

        try:
            validated_plan = StrategyPlan.model_validate(candidate_plan)
            # Compile validates identifier/syntax safety before accepting the edit.
            compile_plan(validated_plan)
        except Exception as exc:
            self._append_trigger_rule_edit({
                **base_record,
                "status": "rejected",
                "before": {
                    "entry_rule": trigger_before.get("entry_rule", ""),
                    "exit_rule": trigger_before.get("exit_rule", ""),
                    "hold_rule": trigger_before.get("hold_rule", ""),
                },
                "after": {
                    "entry_rule": trigger_after.get("entry_rule", ""),
                    "exit_rule": trigger_after.get("exit_rule", ""),
                    "hold_rule": trigger_after.get("hold_rule", ""),
                },
                "error": str(exc),
            })
            workflow.logger.warning("Rejected trigger rule edit request_id=%s trigger_id=%s: %s", request_id, trigger_id, exc)
            return

        self.current_plan = validated_plan.model_dump()
        if self.plan_history:
            self.plan_history[-1]["triggers"] = self.current_plan.get("triggers", [])

        self._append_trigger_rule_edit({
            **base_record,
            "status": "applied",
            "before": {
                "entry_rule": trigger_before.get("entry_rule", ""),
                "exit_rule": trigger_before.get("exit_rule", ""),
                "hold_rule": trigger_before.get("hold_rule", ""),
            },
            "after": {
                "entry_rule": trigger_after.get("entry_rule", ""),
                "exit_rule": trigger_after.get("exit_rule", ""),
                "hold_rule": trigger_after.get("hold_rule", ""),
            },
        })
        workflow.logger.info("Applied trigger rule edit request_id=%s trigger_id=%s", request_id, trigger_id)

    # -------------------------------------------------------------------------
    # Queries
    # -------------------------------------------------------------------------

    @workflow.query
    def get_session_status(self) -> Dict[str, Any]:
        """Get current session status."""
        research = self.research.model_dump() if self.research else None
        return {
            "session_id": self.session_id,
            "symbols": self.symbols,
            "stopped": self.stopped,
            "cycle_count": self.cycle_count,
            "last_plan_time": self.last_plan_time.isoformat() if self.last_plan_time else None,
            "has_plan": self.current_plan is not None,
            "plan_interval_hours": self.plan_interval_hours,
            "research": research,
            "active_experiments": self.active_experiments,
        }

    @workflow.query
    def get_current_plan(self) -> Optional[Dict[str, Any]]:
        """Get the current strategy plan."""
        return self.current_plan

    @workflow.query
    def get_symbols(self) -> List[str]:
        """Get current symbols being traded."""
        return list(self.symbols)

    @workflow.query
    def get_plan_history(self) -> List[Dict[str, Any]]:
        """Get history of all strategy plans generated for this session."""
        return list(self.plan_history)

    @workflow.query
    def get_trigger_rule_edits(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get manual trigger-rule edit history (newest first)."""
        edits = list(self.trigger_rule_edits)
        edits.sort(key=lambda e: str(e.get("timestamp", "")), reverse=True)
        return edits[:limit]

    @workflow.query
    def get_equity_history(self) -> List[Dict[str, Any]]:
        """Get equity curve history for this session."""
        return list(self.equity_history)

    @workflow.query
    def get_last_indicators(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get latest indicator snapshots used for planning/evaluation."""
        if symbol:
            sym = str(symbol).upper()
            snap = self.last_indicators.get(sym)
            return {sym: snap} if snap is not None else {}
        return dict(self.last_indicators)

    @workflow.query
    def get_live_indicators(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get latest live (1m) indicator snapshots for UI structure context."""
        source = self.last_indicators_live or self.last_indicators
        if symbol:
            sym = str(symbol).upper()
            snap = source.get(sym)
            return {sym: snap} if snap is not None else {}
        return dict(source)

    @workflow.query
    def get_structure_snapshots(
        self,
        symbol: Optional[str] = None,
        as_of: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get structure snapshots for the UI, optionally resolved at-or-before ``as_of``.

        Returns a payload with:
          - indicators: {symbol: snapshot}
          - lookup_mode: "latest" | "at_or_before"
          - requested_as_of: echo of input (when provided)
          - resolved_as_of / resolved_as_of_by_symbol
        """
        sym_filter = str(symbol).upper() if symbol else None
        latest_source = self.last_indicators_live or self.last_indicators

        if not as_of:
            indicators = self._filter_indicator_dict(latest_source, sym_filter)
            payload: Dict[str, Any] = {
                "indicators": indicators,
                "lookup_mode": "latest",
            }
            if sym_filter:
                resolved = self._snapshot_as_of_str(indicators.get(sym_filter))
                if resolved:
                    payload["resolved_as_of"] = resolved
            else:
                payload["resolved_as_of_by_symbol"] = {
                    k: ts for k, ts in (
                        (k, self._snapshot_as_of_str(v)) for k, v in indicators.items()
                    ) if ts
                }
            return payload

        indicators, resolved_map = self._lookup_structure_at_or_before(as_of, sym_filter)
        payload = {
            "indicators": indicators,
            "lookup_mode": "at_or_before",
            "requested_as_of": as_of,
        }
        if sym_filter:
            payload["resolved_as_of"] = resolved_map.get(sym_filter)
        else:
            payload["resolved_as_of_by_symbol"] = resolved_map
        return payload

    # -------------------------------------------------------------------------
    # Main Workflow
    # -------------------------------------------------------------------------

    @workflow.run
    async def run(
        self,
        config: Optional[Dict[str, Any]] = None,
        resume_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run the paper trading session.

        Args:
            config: Initial configuration (used on first run)
            resume_state: State from continue-as-new (used on resume)
        """
        # Initialize state
        if resume_state:
            self._restore_state(resume_state)
            workflow.logger.info(f"Resumed session {self.session_id} at cycle {self.cycle_count}")
        elif config:
            parsed_config = PaperTradingConfig.model_validate(config)
            self.session_id = parsed_config.session_id
            self.ledger_workflow_id = (
                parsed_config.ledger_workflow_id or MOCK_LEDGER_WORKFLOW_ID
            )
            self.symbols = list(parsed_config.symbols)
            self.strategy_prompt = parsed_config.strategy_prompt
            self.plan_interval_hours = parsed_config.plan_interval_hours
            self.indicator_timeframe = parsed_config.indicator_timeframe
            self.direction_bias = parsed_config.direction_bias
            self.replan_on_day_boundary = parsed_config.replan_on_day_boundary
            self.enable_symbol_discovery = parsed_config.enable_symbol_discovery
            self.min_volume_24h = parsed_config.min_volume_24h
            self.exit_binding_mode = parsed_config.exit_binding_mode
            self.conflicting_signal_policy = parsed_config.conflicting_signal_policy

            # Initialize research budget (separate capital pool)
            if not parsed_config.research_budget_enabled:
                self.research = None
                self.active_experiments = []
            else:
                env_fraction = float(os.environ.get("RESEARCH_BUDGET_FRACTION", "0.10"))
                research_fraction = (
                    parsed_config.research_budget_fraction
                    if parsed_config.research_budget_fraction is not None
                    else env_fraction
                )
                research_capital = parsed_config.initial_cash * research_fraction
                env_max_loss = research_capital * 0.5
                max_loss_pct = (
                    parsed_config.research_max_loss_pct
                    if parsed_config.research_max_loss_pct is not None
                    else 50.0
                )
                max_loss_usd = research_capital * (max_loss_pct / 100.0)
                self.research = ResearchBudgetState(
                    initial_capital=research_capital,
                    cash=research_capital,
                    max_loss_usd=max_loss_usd,
                )
                self.active_experiments = []

            # Initialize portfolio with starting allocations
            await self._initialize_portfolio(
                parsed_config.initial_cash,
                parsed_config.initial_allocations or {},
            )

            # Emit session started event
            await workflow.execute_activity(
                emit_paper_trading_event_activity,
                args=[self.session_id, "session_started", {
                    "symbols": self.symbols,
                    "initial_cash": parsed_config.initial_cash,
                    "initial_allocations": parsed_config.initial_allocations,
                    "research_budget": (
                        {
                            "enabled": True,
                            "initial_capital": self.research.initial_capital,
                            "cash": self.research.cash,
                            "max_loss_usd": self.research.max_loss_usd,
                            "paused": self.research.paused,
                        }
                        if self.research is not None
                        else {"enabled": False}
                    ),
                }],
                schedule_to_close_timeout=timedelta(seconds=10),
            )

            workflow.logger.info(f"Started paper trading session: {self.session_id}")
        else:
            raise ValueError("Either config or resume_state must be provided")

        # Main loop
        plan_interval = timedelta(hours=self.plan_interval_hours)
        evaluation_interval = timedelta(seconds=30)  # Evaluate every 30 seconds
        live_indicators_interval = timedelta(minutes=1)
        enable_live_indicator_refresh = workflow.patched("paper-trading-live-indicators-v1")

        while not self.stopped:
            now = workflow.now()

            # Check for continue-as-new
            hist_len = workflow.info().get_current_history_length()
            if (
                hist_len >= PAPER_TRADING_HISTORY_LIMIT
                or self.cycle_count >= PAPER_TRADING_CONTINUE_EVERY
                or workflow.info().is_continue_as_new_suggested()
            ):
                workflow.logger.info(f"Triggering continue-as-new at cycle {self.cycle_count}")
                await workflow.continue_as_new(
                    args=[None, self._snapshot_state()]
                )

            # Daily symbol discovery
            if self.enable_symbol_discovery:
                today = now.strftime("%Y-%m-%d")
                if self.last_discovery_date != today:
                    await self._discover_new_symbols()
                    self.last_discovery_date = today

            # Refresh UI structure snapshots on a 1-minute cadence so context is
            # "live-ish" rather than only updating at plan generation time.
            if enable_live_indicator_refresh and (
                self.last_live_indicators_refresh is None
                or (now - self.last_live_indicators_refresh) >= live_indicators_interval
            ):
                await self._refresh_live_indicators()
                self.last_live_indicators_refresh = now

            # Generate/update strategy plan
            if self.replan_on_day_boundary and self.last_plan_time is not None:
                if self.last_plan_time.date() != now.date():
                    self.last_plan_time = None
            if self.last_plan_time is None or (now - self.last_plan_time) >= plan_interval:
                await self._generate_plan()

            # Evaluate triggers and execute orders
            if self.current_plan:
                await self._evaluate_and_execute()

            # Track equity periodically (every 5 minutes)
            equity_interval = timedelta(minutes=5)
            if self.last_equity_snapshot is None or (now - self.last_equity_snapshot) >= equity_interval:
                await self._record_equity_snapshot()

            self.cycle_count += 1

            # Sleep until next evaluation
            await workflow.sleep(evaluation_interval)

        # Stop the session-scoped ledger workflow on graceful shutdown.
        # Never stop the legacy shared ledger.
        if self.ledger_workflow_id and self.ledger_workflow_id != MOCK_LEDGER_WORKFLOW_ID:
            try:
                ledger_handle = workflow.get_external_workflow_handle(self.ledger_workflow_id)
                await ledger_handle.signal("stop_workflow")
            except Exception as exc:
                workflow.logger.warning(f"Failed to stop ledger workflow {self.ledger_workflow_id}: {exc}")

        # Session stopped
        await workflow.execute_activity(
            emit_paper_trading_event_activity,
            args=[self.session_id, "session_stopped", {
                "cycle_count": self.cycle_count,
            }],
            schedule_to_close_timeout=timedelta(seconds=10),
        )

        return {
            "session_id": self.session_id,
            "status": "stopped",
            "cycle_count": self.cycle_count,
        }

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    async def _initialize_portfolio(
        self,
        initial_cash: float,
        initial_allocations: Dict[str, float],
    ) -> None:
        """Initialize the execution ledger with starting portfolio."""
        # Normalize allocation semantics so symbol allocations are funded from
        # initial_cash unless explicit cash is provided.
        allocs = {k: float(v) for k, v in (initial_allocations or {}).items()}
        non_cash_allocs = {
            k: v for k, v in allocs.items()
            if k.lower() != "cash" and v > 0
        }
        non_cash_total = sum(non_cash_allocs.values())
        explicit_cash = allocs.get("cash")
        if explicit_cash is None:
            if non_cash_total > float(initial_cash) + 1e-6:
                raise ValueError(
                    f"Initial allocations ({non_cash_total:.2f}) exceed initial_cash ({float(initial_cash):.2f})"
                )
            cash_for_ledger = max(0.0, float(initial_cash) - non_cash_total)
        else:
            total_budget = float(explicit_cash) + non_cash_total
            if total_budget > float(initial_cash) + 1e-6:
                raise ValueError(
                    f"cash + allocations ({total_budget:.2f}) exceed initial_cash ({float(initial_cash):.2f})"
                )
            cash_for_ledger = float(explicit_cash)

        # Fetch current prices for non-cash allocations
        prices = await workflow.execute_activity(
            fetch_current_prices_activity,
            args=[list(non_cash_allocs.keys())],
            schedule_to_close_timeout=timedelta(seconds=30),
            retry_policy=RetryPolicy(maximum_attempts=3),
        )

        # Signal the execution ledger to initialize
        ledger_handle = workflow.get_external_workflow_handle(self.ledger_workflow_id)
        await ledger_handle.signal("initialize_portfolio", {
            "cash": cash_for_ledger,
            "positions": {
                symbol: alloc / prices.get(symbol, 1)
                for symbol, alloc in non_cash_allocs.items()
                if prices.get(symbol, 0) > 0
            },
            "prices": prices,
        })

        workflow.logger.info(
            f"Initialized portfolio: cash={cash_for_ledger}, "
            f"symbol_allocations={non_cash_allocs}, initial_cash={initial_cash}"
        )

    async def _generate_plan(self) -> None:
        """Generate a new strategy plan."""
        # Get portfolio state from ledger (via activity — external handles can't query)
        portfolio_state = await workflow.execute_activity(
            query_ledger_portfolio_activity,
            args=[self.ledger_workflow_id],
            schedule_to_close_timeout=timedelta(seconds=10),
        )

        # Fetch OHLCV history and compute full indicator snapshots so the LLM
        # sees the same macro/micro picture as the backtester (RSI, SMA, ATR,
        # Bollinger, candlestick morphology, HTF daily anchors, etc.).
        indicator_snapshots = await workflow.execute_activity(
            fetch_indicator_snapshots_activity,
            args=[self.symbols, self.indicator_timeframe, 300],
            schedule_to_close_timeout=timedelta(minutes=2),
            retry_policy=RetryPolicy(maximum_attempts=3),
        )

        # Store snapshots for use in trigger evaluation between plan cycles
        self.last_indicators = indicator_snapshots
        self._append_structure_history(indicator_snapshots, origin="plan")

        # Build market context from full snapshots; fall back to ledger price
        # for any symbol that failed indicator computation.
        market_context = {}
        for symbol in self.symbols:
            snap = indicator_snapshots.get(symbol)
            if snap:
                market_context[symbol] = snap
            else:
                last_price = portfolio_state.get("last_prices", {}).get(symbol, 0)
                market_context[symbol] = {
                    "price": last_price,
                    "trend_state": "sideways",
                    "vol_state": "normal",
                }

        # ── Per-bar regime detection (Runbook 61) ─────────────────────────
        policy_triggers: List[PolicyLoopTriggerEvent] = []
        _primary_sym = self.symbols[0] if self.symbols else None
        if _primary_sym and indicator_snapshots.get(_primary_sym):
            try:
                _snap_dict = indicator_snapshots[_primary_sym]
                _snap_obj = IndicatorSnapshot.model_validate(_snap_dict)
                _asset_st = build_asset_state(_primary_sym, [_snap_obj])
                _fingerprint = build_regime_fingerprint(_snap_obj, _asset_st)
                _detector = RegimeTransitionDetector(symbol=_primary_sym)
                if self.regime_detector_state:
                    _detector.load_state(
                        RegimeTransitionDetectorState.model_validate(
                            self.regime_detector_state
                        )
                    )
                _bar_ts = _snap_obj.as_of
                if _bar_ts.tzinfo is None:
                    _bar_ts = _bar_ts.replace(tzinfo=timezone.utc)
                _transition_event = _detector.evaluate(_fingerprint, current_ts=_bar_ts)
                self.regime_detector_state = _detector.state.model_dump()
                if _transition_event.decision.transition_fired:
                    policy_triggers.append(PolicyLoopTriggerEvent(
                        trigger_type="regime_state_changed",
                        fired_at=workflow.now(),
                        source_detail=f"symbol={_primary_sym}",
                    ))
            except Exception as _det_exc:
                workflow.logger.warning(
                    "Regime detection failed (non-fatal): %s", _det_exc
                )

        _now_ts = workflow.now()
        if self._position_opened_since_last_eval:
            policy_triggers.append(PolicyLoopTriggerEvent(
                trigger_type="position_opened",
                fired_at=_now_ts,
            ))
        if self._position_closed_since_last_eval:
            policy_triggers.append(PolicyLoopTriggerEvent(
                trigger_type="position_closed",
                fired_at=_now_ts,
            ))

        # Gate: skip LLM call if policy state machine blocks evaluation.
        _gate = PolicyLoopGate()
        _state_record = PolicyStateMachineRecord.model_validate(
            self.policy_state_machine_record or {}
        )
        _last_eval_at = (
            datetime.fromisoformat(self.last_policy_eval_at)
            if self.last_policy_eval_at else None
        )
        _gate_allowed, _skip_event = _gate.evaluate(
            scope=self.session_id,
            state_record=_state_record,
            trigger_events=policy_triggers,
            last_eval_at=_last_eval_at,
            indicator_timeframe=self.indicator_timeframe,
        )
        if not _gate_allowed:
            if _skip_event:
                await workflow.execute_activity(
                    emit_paper_trading_event_activity,
                    args=[self.session_id, "policy_loop_skipped", _skip_event.model_dump()],
                    schedule_to_close_timeout=timedelta(seconds=10),
                    retry_policy=RetryPolicy(maximum_attempts=1),
                )
            workflow.logger.debug(
                "PolicyLoopGate: plan generation skipped for session %s", self.session_id
            )
            return
        # ─────────────────────────────────────────────────────────────────────

        # Generate plan (guarded by PolicyLoopGate — gate acquired above)
        try:
            # R63: thread in-session episode records so the activity can load them
            # into the memory store without a DB roundtrip.
            _plan_market_ctx = dict(market_context)
            if self.episode_memory_store_state:
                _plan_market_ctx["__episode_memory_store_state__"] = list(
                    self.episode_memory_store_state
                )
            plan_dict = await workflow.execute_activity(
                generate_strategy_plan_activity,
                args=[
                    self.symbols,
                    portfolio_state,
                    self.strategy_prompt,
                    _plan_market_ctx,
                    None,  # llm_model
                    self.session_id,
                    None,  # repair_instructions
                    self.indicator_timeframe,
                    self.direction_bias,
                    self.symbols[0] if len(self.symbols) == 1 else None,  # preferred_symbol
                    dict(self.policy_state_machine_record) if self.policy_state_machine_record else None,  # R66
                ],
                schedule_to_close_timeout=timedelta(minutes=5),
                retry_policy=RetryPolicy(maximum_attempts=2),
            )

            # R66: Judge validation gate — None signals stand-down (validation exhausted).
            if plan_dict is None:
                workflow.logger.warning(
                    "Judge validation gate: stand-down for session %s — skipping cycle",
                    self.session_id,
                )
                await workflow.execute_activity(
                    emit_paper_trading_event_activity,
                    args=[self.session_id, "plan_validation_rejected", {
                        "cycle": len(self.plan_history),
                        "reason": "validation_exhausted",
                    }],
                    schedule_to_close_timeout=timedelta(seconds=10),
                    retry_policy=RetryPolicy(maximum_attempts=1),
                )
                await workflow.execute_activity(
                    emit_paper_trading_event_activity,
                    args=[self.session_id, "plan_stand_down", {
                        "cycle": len(self.plan_history),
                        "reason": "validation_exhausted",
                    }],
                    schedule_to_close_timeout=timedelta(seconds=10),
                    retry_policy=RetryPolicy(maximum_attempts=1),
                )
                return

            # Compile-time validation: detect hazardous patterns (e.g. ATR tautologies
            # in emergency_exit rules) before the plan is stored.  If hard errors are
            # found, attempt one repair pass with the error context injected into the
            # prompt.  The runtime failsafe in trigger_engine handles anything that slips
            # through (e.g. plans persisted before this validator existed).
            _validation = validate_trigger_plan(plan_dict, base_tf_minutes=_timeframe_to_minutes(self.indicator_timeframe))
            if not _validation.is_valid:
                workflow.logger.warning(
                    f"Plan validation failed — attempting repair pass:\n{_validation.summary()}"
                )
                _repair_dict = await workflow.execute_activity(
                    generate_strategy_plan_activity,
                    args=[
                        self.symbols,
                        portfolio_state,
                        self.strategy_prompt,
                        market_context,
                        None,  # llm_model
                        self.session_id,
                        _validation.repair_prompt(),  # repair_instructions
                        self.indicator_timeframe,
                        self.direction_bias,
                        self.symbols[0] if len(self.symbols) == 1 else None,  # preferred_symbol
                        dict(self.policy_state_machine_record) if self.policy_state_machine_record else None,  # R66
                    ],
                    schedule_to_close_timeout=timedelta(minutes=5),
                    retry_policy=RetryPolicy(maximum_attempts=1),
                )
                # Repair pass skips judge validation (repair_instructions is set), so None
                # is not expected here; treat it defensively as a pass-through.
                if _repair_dict is None:
                    _repair_dict = plan_dict
                _recheck = validate_trigger_plan(_repair_dict, base_tf_minutes=_timeframe_to_minutes(self.indicator_timeframe))
                if _recheck.is_valid:
                    workflow.logger.info("Repair plan passed validation — using repaired plan.")
                    plan_dict = _repair_dict
                else:
                    workflow.logger.error(
                        f"Repair plan still has validation errors. "
                        f"Runtime failsafe in trigger_engine will suppress tautological "
                        f"emergency exits:\n{_recheck.summary()}"
                    )
                    # Proceed with original plan — runtime failsafe is the last line of defence

            # Pop retrieval + R49 snapshot metadata before storing — StrategyPlan uses extra="forbid"
            _retrieved_template_id = plan_dict.pop("_retrieved_template_id", None)
            _snapshot_id = plan_dict.pop("_snapshot_id", None)
            _snapshot_hash = plan_dict.pop("_snapshot_hash", None)
            _snapshot_version = plan_dict.pop("_snapshot_version", None)
            _snapshot_kind = plan_dict.pop("_snapshot_kind", None)
            _snapshot_as_of_ts = plan_dict.pop("_snapshot_as_of_ts", None)
            _snapshot_staleness_seconds = plan_dict.pop("_snapshot_staleness_seconds", None)
            _snapshot_missing_sections = plan_dict.pop("_snapshot_missing_sections", None)

            self.current_plan = plan_dict
            self.last_plan_time = workflow.now()

            # Store plan in history
            plan_record = {
                "plan_index": len(self.plan_history),
                "generated_at": self.last_plan_time.isoformat(),
                "trigger_count": len(plan_dict.get("triggers", [])),
                "max_trades_per_day": plan_dict.get("max_trades_per_day"),
                "market_regime": plan_dict.get("market_regime"),
                "symbols": plan_dict.get("allowed_symbols", self.symbols),
                "valid_until": plan_dict.get("valid_until"),
                "triggers": plan_dict.get("triggers", []),
            }
            self.plan_history.append(plan_record)

            # Emit event
            validation_errors = [
                {"trigger_id": e.trigger_id, "code": e.code, "message": e.message}
                for e in _validation.hard_errors
            ]
            await workflow.execute_activity(
                emit_paper_trading_event_activity,
                args=[self.session_id, "plan_generated", {
                    "trigger_count": len(plan_dict.get("triggers", [])),
                    "plan_index": len(self.plan_history) - 1,
                    "validation_errors": validation_errors,
                    "retrieved_template_id": _retrieved_template_id,
                    "template_id": plan_dict.get("template_id"),
                    # R49 snapshot provenance
                    "snapshot_id": _snapshot_id,
                    "snapshot_hash": _snapshot_hash,
                    "snapshot_version": _snapshot_version,
                    "snapshot_kind": _snapshot_kind,
                    "snapshot_as_of_ts": _snapshot_as_of_ts,
                    "snapshot_staleness_seconds": _snapshot_staleness_seconds,
                    "snapshot_missing_sections": _snapshot_missing_sections,
                }],
                schedule_to_close_timeout=timedelta(seconds=10),
            )

            workflow.logger.info(f"Generated strategy plan with {len(plan_dict.get('triggers', []))} triggers")

            # Update cadence tracking (Runbook 61)
            self.last_policy_eval_at = workflow.now().isoformat()
            self._position_opened_since_last_eval = False
            self._position_closed_since_last_eval = False
        finally:
            _gate.release(self.session_id)

    async def _refresh_live_indicators(self) -> None:
        """Refresh 1-minute indicator snapshots used by the structure UI."""
        try:
            live_snapshots = await workflow.execute_activity(
                fetch_indicator_snapshots_activity,
                args=[self.symbols, "1m", 240],
                schedule_to_close_timeout=timedelta(minutes=2),
                retry_policy=RetryPolicy(maximum_attempts=2),
            )
            if isinstance(live_snapshots, dict) and live_snapshots:
                self.last_indicators_live = live_snapshots
                self._append_structure_history(live_snapshots, origin="live")
        except Exception as exc:
            workflow.logger.warning(f"Failed to refresh live indicators: {exc}")

    @staticmethod
    def _candle_floor(dt: datetime, tf_minutes: int) -> str:
        """Return the ISO timestamp of the candle that started at or before *dt*."""
        total_minutes = dt.hour * 60 + dt.minute
        floor_minutes = (total_minutes // tf_minutes) * tf_minutes
        floored = dt.replace(
            hour=floor_minutes // 60,
            minute=floor_minutes % 60,
            second=0,
            microsecond=0,
        )
        return floored.isoformat()

    async def _evaluate_and_execute(self) -> None:
        """Evaluate triggers and execute orders using a two-tier model:

        Tier 1 — every 30-second tick (price-based, time-critical):
          * Fetch fresh prices and push to ledger for live P&L.
          * Check if any open position has crossed its stop or target level.
            Stop-losses and take-profits trigger immediately, regardless of
            whether a new candle has closed.

        Tier 2 — on new candle close only (indicator-based):
          * Evaluate the full trigger engine: entry signals, indicator-based
            exit rules, risk constraints, etc.
          * Only fires when a new candle of the strategy timeframe has started.
            For 1h strategies this means at most once per hour, which prevents
            the whipsaw pattern where an exit fires on the very next 30-second
            tick after entry (stale indicator values looked identical every tick).
        """
        if not self.current_plan:
            return

        # ── Candle-clock ──────────────────────────────────────────────────────
        TF_MINUTES: Dict[str, int] = {
            "1m": 1, "5m": 5, "15m": 15, "30m": 30,
            "1h": 60, "6h": 360, "1d": 1440,
        }
        plan_timeframes = {
            t.get("timeframe", "1h")
            for t in self.current_plan.get("triggers", [])
            if isinstance(t, dict)
        }
        if not plan_timeframes:
            plan_timeframes = {"1h"}

        now = workflow.now()
        current_candles = {
            tf: self._candle_floor(now, TF_MINUTES.get(tf, 60))
            for tf in plan_timeframes
        }
        new_candle_tfs = {
            tf for tf, floor in current_candles.items()
            if self.last_eval_candle_by_tf.get(tf) != floor
        }
        if new_candle_tfs:
            self.last_eval_candle_by_tf.update(
                {tf: current_candles[tf] for tf in new_candle_tfs}
            )
        # ─────────────────────────────────────────────────────────────────────

        # ── Tier 1: always runs every tick ───────────────────────────────────
        current_prices = await workflow.execute_activity(
            fetch_current_prices_activity,
            args=[self.symbols],
            schedule_to_close_timeout=timedelta(seconds=30),
            retry_policy=RetryPolicy(maximum_attempts=2),
        )

        # Push prices to ledger for unrealized P&L display
        live_prices = {s: float(current_prices.get(s) or 0) for s in self.symbols if current_prices.get(s)}
        if live_prices:
            ledger_handle = workflow.get_external_workflow_handle(self.ledger_workflow_id)
            await ledger_handle.signal("update_last_prices", live_prices)

        # Emit tick for UI
        await workflow.execute_activity(
            emit_paper_trading_event_activity,
            args=[self.session_id, "tick", {
                "prices": {s: float(current_prices.get(s) or 0) for s in self.symbols},
                "cycle": self.cycle_count,
            }],
            schedule_to_close_timeout=timedelta(seconds=10),
        )

        # Query portfolio once — used by both the stop/target sweep and (if
        # a new candle) the full indicator evaluation below.
        portfolio_state = await workflow.execute_activity(
            query_ledger_portfolio_activity,
            args=[self.ledger_workflow_id],
            schedule_to_close_timeout=timedelta(seconds=10),
        )

        # Stop/target price sweep — exits any position that has crossed its
        # absolute stop or target level since the last tick.
        await self._sweep_stop_target(current_prices, portfolio_state)

        # ── Tier 2: only on new candle close ─────────────────────────────────
        if not new_candle_tfs:
            return

        # Build market data: stale indicators + fresh close price
        market_data = {}
        for symbol in self.symbols:
            price = (current_prices.get(symbol)
                     or portfolio_state.get("last_prices", {}).get(symbol, 0))
            if not price:
                continue
            base = dict(self.last_indicators.get(symbol, {}))
            base.update({
                "timestamp": workflow.now().isoformat(),
                "price": price,
                "close": price,  # fresh price overrides stale snapshot close
                "open": price,
                "high": price,
                "low": price,
                "volume": 0,
            })
            market_data[symbol] = base

        if not market_data:
            return

        # R63: Per-bar AdaptiveTradeManagement tick for each open position.
        # Runs deterministically inside the workflow (pure Pydantic state update).
        try:
            from services.adaptive_trade_management import AdaptiveTradeManagementState
            _open_positions = portfolio_state.get("positions") or {}
            _position_meta = portfolio_state.get("position_meta") or {}
            for _sym, _qty in _open_positions.items():
                if not _qty or _qty <= 0:
                    continue
                _price = float(current_prices.get(_sym) or 0)
                if not _price:
                    continue
                _state_dict = self.adaptive_management_states.get(_sym, {})
                _meta = _position_meta.get(_sym) or {}
                _meta["symbol"] = _sym
                _mgmt = (
                    AdaptiveTradeManagementState.model_validate(_state_dict)
                    if _state_dict
                    else AdaptiveTradeManagementState.initial(_meta)
                )
                _mgmt = _mgmt.tick(current_price=_price)
                self.adaptive_management_states[_sym] = _mgmt.model_dump()
        except Exception as _amt_exc:
            workflow.logger.debug("AdaptiveTradeManagement tick failed (non-fatal): %s", _amt_exc)

        result = await workflow.execute_activity(
            evaluate_triggers_activity,
            args=[
                self.current_plan,
                market_data,
                portfolio_state,
                self.exit_binding_mode,
                self.conflicting_signal_policy,
                dict(self.position_originating_plans),  # R65: originating plan pinning
            ],
            schedule_to_close_timeout=timedelta(seconds=30),
        )

        orders = result.get("orders", []) if isinstance(result, dict) else list(result)
        trigger_events = result.get("events", []) if isinstance(result, dict) else []

        for ev in trigger_events:
            await workflow.execute_activity(
                emit_paper_trading_event_activity,
                args=[self.session_id, ev["type"], ev["payload"]],
                schedule_to_close_timeout=timedelta(seconds=10),
            )

        for order in orders:
            # Phase M3 (Runbook 60): block entry-rule flatten exits for symbols
            # that have an active exit contract.  Contract stop/target/time rules
            # govern the exit — the generic _flat path is suppressed.
            # Emergency exits and symbols without contracts are still allowed through.
            _sym = order.get("symbol", "")
            _reason = order.get("reason") or order.get("trigger_id") or ""
            if (
                order.get("intent") == "exit"
                and isinstance(_reason, str)
                and _reason.endswith("_flat")
                and order.get("trigger_category") != "emergency_exit"
                and _sym in self.exit_contracts
            ):
                await workflow.execute_activity(
                    emit_paper_trading_event_activity,
                    args=[
                        self.session_id,
                        "trade_blocked",
                        {
                            "symbol": _sym,
                            "trigger_id": _reason,
                            "reason": "m3_contract_backed_exit",
                            "exit_class": "strategy_contract",
                            "detail": (
                                "Entry-rule flatten suppressed (Phase M3): position has an "
                                "active exit contract. Contract stop/target/time rules govern "
                                "this exit path."
                            ),
                        },
                    ],
                    schedule_to_close_timeout=timedelta(seconds=10),
                )
                workflow.logger.debug(
                    "M3: suppressed entry-rule flatten for %s (%s) — exit contract active",
                    _sym, _reason,
                )
                continue
            await self._execute_order(order)

    async def _sweep_stop_target(
        self,
        current_prices: Dict[str, float],
        portfolio_state: Dict[str, Any],
    ) -> None:
        """Check open positions for stop/target level crosses and exit immediately.

        This runs on every 30-second tick so stop-losses and take-profits
        trigger as soon as price crosses the level, even mid-candle.  It reads
        the absolute stop/target prices that were stored in position_meta when
        the position was opened (Runbook 42 anchors).
        """
        positions = portfolio_state.get("positions") or {}
        position_meta = portfolio_state.get("position_meta") or {}

        for symbol, qty in positions.items():
            if not qty or qty <= 0:
                continue

            price = current_prices.get(symbol, 0)
            if not price:
                continue

            meta = position_meta.get(symbol) or {}
            stop_px: Optional[float] = meta.get("stop_price_abs")
            target_px: Optional[float] = meta.get("target_price_abs")
            direction: str = meta.get("entry_side", "long")

            hit: Optional[str] = None
            if direction == "long":
                if stop_px is not None and price <= stop_px:
                    hit = "stop"
                elif target_px is not None and price >= target_px:
                    hit = "target"
            else:  # short
                if stop_px is not None and price >= stop_px:
                    hit = "stop"
                elif target_px is not None and price <= target_px:
                    hit = "target"

            if hit is None:
                continue

            category = "stop_loss" if hit == "stop" else "take_profit"
            trigger_id = f"{symbol.lower().replace('-', '_')}_{hit}_hit"
            exit_side = "sell" if direction == "long" else "buy"

            workflow.logger.info(
                f"Price sweep: {symbol} {hit} hit @ {price:.4f} "
                f"(stop={stop_px}, target={target_px})"
            )

            # Emit trigger_fired event
            await workflow.execute_activity(
                emit_paper_trading_event_activity,
                args=[self.session_id, "trigger_fired", {
                    "symbol": symbol,
                    "side": exit_side,
                    "trigger_id": trigger_id,
                    "category": category,
                    "price": price,
                    "timeframe": "1h",
                }],
                schedule_to_close_timeout=timedelta(seconds=10),
            )

            # Execute the exit order
            await self._execute_order({
                "symbol": symbol,
                "side": exit_side,
                "quantity": float(qty),
                "price": price,
                "timeframe": "1h",
                "trigger_id": trigger_id,
                "trigger_category": category,
                "intent": "exit",
                "reason": trigger_id,
            })

            # A4: Build episode record after position close (non-fatal).
            try:
                _entry_price = float(
                    meta.get("signal_entry_price")
                    or portfolio_state.get("entry_prices", {}).get(symbol)
                    or price
                )
                _fill_ts = meta.get("opened_at") or workflow.now().isoformat()
                await workflow.execute_activity(
                    build_episode_activity,
                    args=[
                        meta.get("signal_id"),   # signal_id (may be None)
                        symbol,
                        direction,
                        _entry_price,
                        _fill_ts,
                        float(price),
                        workflow.now().isoformat(),
                        stop_px,
                        target_px,
                        hit,
                        self.indicator_timeframe,
                    ],
                    schedule_to_close_timeout=timedelta(seconds=15),
                    retry_policy=RetryPolicy(maximum_attempts=1),
                )
            except Exception:
                pass  # episode construction is non-critical

            # R63: Append lightweight episode record to in-session store for next plan gen.
            try:
                _ep_entry = float(
                    meta.get("signal_entry_price")
                    or portfolio_state.get("entry_prices", {}).get(symbol)
                    or price
                )
                _ep_risk = abs(_ep_entry - (stop_px or _ep_entry * (0.98 if direction == "long" else 1.02)))
                _ep_r = (
                    (float(price) - _ep_entry) / _ep_risk if direction == "long" else (_ep_entry - float(price)) / _ep_risk
                ) if _ep_risk > 0 else 0.0
                self.episode_memory_store_state = (self.episode_memory_store_state or [])[-99:]
                self.episode_memory_store_state.append({
                    "episode_id": f"paper_{symbol}_{workflow.now().isoformat()}",
                    "signal_id": meta.get("signal_id"),
                    "symbol": symbol,
                    "direction": direction,
                    "timeframe": self.indicator_timeframe,
                    "playbook_id": meta.get("playbook_id"),
                    "template_id": meta.get("template_id"),
                    "outcome_class": "win" if hit == "target" else "loss",
                    "r_achieved": round(_ep_r, 4),
                    "exit_ts": workflow.now().isoformat(),
                    "failure_modes": [],
                })
                # Also clear adaptive management state for this symbol on close
                self.adaptive_management_states.pop(symbol, None)
            except Exception as _ep_exc:
                workflow.logger.debug("episode_memory_store_state append failed (non-fatal): %s", _ep_exc)

            # Runbook 61: transition state machine to COOLDOWN on position close.
            self._position_closed_since_last_eval = True
            try:
                _sm = PolicyStateMachine()
                _sm_record = PolicyStateMachineRecord.model_validate(
                    self.policy_state_machine_record or {}
                )
                if _sm_record.current_state in ("POSITION_OPEN", "HOLD_LOCK"):
                    _sm_record = _sm.close_position(_sm_record)
                    self.policy_state_machine_record = _sm_record.model_dump()
            except Exception as _sm_exc:
                workflow.logger.warning(
                    "State machine close_position failed (non-fatal): %s", _sm_exc
                )

    def _resolve_order_stop_target(
        self, order: Dict[str, Any], fill_price: float
    ) -> tuple[float | None, float | None]:
        """Resolve stop and target prices for a filled entry order.

        Uses the trigger's anchor configuration and the most recent indicator
        snapshot (self.last_indicators) to compute absolute price levels.
        Returns (stop_price_abs, target_price_abs) — either may be None.
        """
        if not self.current_plan or order.get("intent") != "entry":
            return None, None

        trigger_id = order.get("trigger_id") or order.get("reason")
        symbol = order["symbol"]
        direction = "long" if order.get("side", "").lower() == "buy" else "short"

        # Find the trigger definition in the current plan
        triggers = (self.current_plan or {}).get("triggers", [])
        trigger_dict = next((t for t in triggers if t.get("id") == trigger_id), None)
        if not trigger_dict:
            return None, None

        # Get the indicator snapshot for this symbol
        snap_dict = self.last_indicators.get(symbol)
        if not snap_dict:
            return None, None

        try:
            from backtesting.llm_strategist_runner import (
                _resolve_stop_price_anchored,
                _resolve_target_price_anchored,
            )
            from schemas.llm_strategist import TriggerCondition
            from agents.analytics.indicator_snapshots import IndicatorSnapshot
            import datetime as _dt

            trigger = TriggerCondition.model_validate(trigger_dict)
            snap = IndicatorSnapshot.model_validate({
                **snap_dict,
                "symbol": symbol,
                "timeframe": trigger_dict.get("timeframe", "1h"),
                "as_of": _dt.datetime.now(_dt.timezone.utc),
            })
            stop_abs, _ = _resolve_stop_price_anchored(trigger, fill_price, snap, direction)
            target_abs, _ = _resolve_target_price_anchored(trigger, fill_price, stop_abs, snap, direction)
            return stop_abs, target_abs
        except Exception as e:
            workflow.logger.warning(f"Could not resolve stop/target for {trigger_id}: {e}")
            return None, None

    async def _execute_order(self, order: Dict[str, Any]) -> None:
        """Execute a single order."""
        ledger_handle = workflow.get_external_workflow_handle(self.ledger_workflow_id)
        # Record fill in ledger
        fill_price = float(order.get("price", 0.0) or 0.0)
        quantity = float(order.get("quantity", 0.0) or 0.0)
        cost = fill_price * quantity
        fee = cost * 0.001

        # Resolve stop/target for entry orders
        stop_price_abs, target_price_abs = self._resolve_order_stop_target(order, fill_price)
        trigger_id = order.get("trigger_id") or order.get("reason")
        trigger_dict = None
        if self.current_plan and trigger_id:
            trigger_dict = next(
                (t for t in (self.current_plan.get("triggers", []) or []) if t.get("id") == trigger_id),
                None,
            )
        target_anchor_type = (trigger_dict or {}).get("target_anchor_type")
        stop_anchor_type = (trigger_dict or {}).get("stop_anchor_type")

        # R64: structural stop/target candidate selection at entry (supplementary logging).
        # Only applied when the trigger has no explicit anchor type; never overrides explicit anchors.
        if order.get("intent") == "entry":
            _sym = order.get("symbol", "")
            _direction = "long" if order.get("side", "buy").lower() == "buy" else "short"
            _structure_snap = _get_latest_structure_snapshot(self.structure_history, _sym)
            if _structure_snap:
                try:
                    from services.structural_target_selector import (
                        select_stop_candidates as _select_stop_candidates,
                        select_target_candidates as _select_target_candidates,
                    )
                    if stop_anchor_type is None:
                        _stop_candidates = _select_stop_candidates(
                            _structure_snap,
                            direction=_direction,
                            max_distance_atr=3.0,
                        )
                        if _stop_candidates:
                            _nearest_stop = _stop_candidates[0]
                            workflow.logger.info(
                                "Structural stop candidate: %s @ %.4f (kind=%s)",
                                _nearest_stop.role_now,
                                _nearest_stop.price,
                                _nearest_stop.kind,
                            )
                    if target_anchor_type is None:
                        _target_candidates = _select_target_candidates(
                            _structure_snap,
                            direction=_direction,
                            max_distance_atr=10.0,
                        )
                        if _target_candidates:
                            _nearest_target = _target_candidates[0]
                            workflow.logger.info(
                                "Structural target candidate: %s @ %.4f (kind=%s)",
                                _nearest_target.role_now,
                                _nearest_target.price,
                                _nearest_target.kind,
                            )
                except Exception as _struct_exc:
                    workflow.logger.warning(
                        "Structural candidate selection failed (non-fatal): %s", _struct_exc
                    )

        # Gate: reject entries whose stop failed to resolve. An entry without a
        # known stop price cannot be risk-sized or swept. This is defense-in-depth
        # (schema validation ensures triggers define a stop; this catches anchor
        # resolution failures such as missing HTF snapshot fields).
        if order.get("intent") == "entry" and stop_price_abs is None:
            workflow.logger.warning(
                f"Entry order for {order.get('symbol')} ({trigger_id}) rejected: "
                f"stop price could not be resolved — position not opened"
            )
            await workflow.execute_activity(
                emit_paper_trading_event_activity,
                args=[
                    self.session_id,
                    "trade_blocked",
                    {
                        "symbol": order.get("symbol"),
                        "trigger_id": trigger_id,
                        "reason": "stop_price_unresolvable",
                        "detail": "Stop anchor resolved to None; entry rejected to prevent unprotected position.",
                    },
                ],
                schedule_to_close_timeout=timedelta(seconds=10),
            )
            return

        # Gate: if a target anchor is defined but target failed to resolve, reject.
        if order.get("intent") == "entry" and target_anchor_type and target_price_abs is None:
            workflow.logger.warning(
                f"Entry order for {order.get('symbol')} ({trigger_id}) rejected: "
                f"target price could not be resolved for anchor={target_anchor_type}"
            )
            await workflow.execute_activity(
                emit_paper_trading_event_activity,
                args=[
                    self.session_id,
                    "trade_blocked",
                    {
                        "symbol": order.get("symbol"),
                        "trigger_id": trigger_id,
                        "reason": "target_price_unresolvable",
                        "detail": f"Target anchor '{target_anchor_type}' resolved to None; entry rejected.",
                    },
                ],
                schedule_to_close_timeout=timedelta(seconds=10),
            )
            return

        # Gate: enforce minimum realized R:R from resolved stop/target at entry.
        if (
            order.get("intent") == "entry"
            and stop_price_abs is not None
            and target_price_abs is not None
            and self.min_rr_ratio > 0
        ):
            risk = abs(fill_price - stop_price_abs)
            reward = abs(target_price_abs - fill_price)
            rr = (reward / risk) if risk > 0 else 0.0
            if rr < self.min_rr_ratio:
                workflow.logger.warning(
                    f"Entry order for {order.get('symbol')} ({trigger_id}) rejected: "
                    f"resolved R:R {rr:.2f} < {self.min_rr_ratio:.2f}"
                )
                await workflow.execute_activity(
                    emit_paper_trading_event_activity,
                    args=[
                        self.session_id,
                        "trade_blocked",
                        {
                            "symbol": order.get("symbol"),
                            "trigger_id": trigger_id,
                            "reason": "insufficient_rr_resolved",
                            "detail": (
                                f"Resolved R:R {rr:.2f} below minimum {self.min_rr_ratio:.2f} "
                                f"(entry={fill_price:.4f}, stop={stop_price_abs:.4f}, target={target_price_abs:.4f})"
                            ),
                        },
                    ],
                    schedule_to_close_timeout=timedelta(seconds=10),
                )
                return

        fill_payload: Dict[str, Any] = {
            "symbol": order["symbol"],
            "side": order["side"].upper(),
            "qty": quantity,
            "fill_price": fill_price,
            "cost": cost,
            "fee": fee,  # 0.1% fee for reporting
            "timestamp": int(workflow.now().timestamp() * 1000),
            "trigger_id": trigger_id,
            "trigger_category": order.get("trigger_category"),
            "intent": order.get("intent"),
        }
        if stop_price_abs is not None:
            fill_payload["stop_price_abs"] = stop_price_abs
        if target_price_abs is not None:
            fill_payload["target_price_abs"] = target_price_abs
        # A2: thread signal_id + signal metadata through fill payload for position_meta.
        _signal_id = order.get("signal_id")
        _signal_ts = order.get("signal_ts")
        _signal_entry_price = order.get("signal_entry_price")
        if _signal_id:
            fill_payload["signal_id"] = _signal_id
        if _signal_ts:
            fill_payload["signal_ts"] = _signal_ts
        if _signal_entry_price is not None:
            fill_payload["signal_entry_price"] = float(_signal_entry_price)
        # Carry the LLM's time estimate to the fill record for playbook accuracy stats
        if order.get("intent") == "entry":
            trigger_id = order.get("trigger_id") or order.get("reason")
            triggers = (self.current_plan or {}).get("triggers", [])
            tdict = next((t for t in triggers if t.get("id") == trigger_id), None)
            if tdict:
                etb = tdict.get("estimated_bars_to_resolution")
                if etb is not None:
                    fill_payload["estimated_bars_to_resolution"] = int(etb)

        # Phase M2 (Runbook 60): Materialize PositionExitContract at entry.
        # The contract captures the precommitted exit plan (stop, targets, time expiry)
        # before capital is committed.  Building it here ensures provenance is tied
        # to the fill record.
        _exit_contract: Optional["PositionExitContract"] = None
        if order.get("intent") == "entry" and stop_price_abs is not None:
            _sym = order["symbol"]
            _dir: Literal["long", "short"] = (
                "long" if order.get("side", "buy").lower() == "buy" else "short"
            )
            _pos_id = f"{_sym}_{self.session_id}_{int(workflow.now().timestamp() * 1000)}"

            # Try primary path: build via TriggerCondition (preserves anchor/time metadata)
            try:
                if trigger_dict:
                    from schemas.llm_strategist import TriggerCondition as _TC  # noqa: PLC0415
                    _trigger_obj = _TC.model_validate(trigger_dict)
                    _exit_contract = build_exit_contract(
                        trigger=_trigger_obj,
                        position_id=_pos_id,
                        entry_price=fill_price,
                        initial_qty=float(quantity),
                        stop_price_abs=stop_price_abs,
                        target_price_abs=target_price_abs,
                        plan_id=(self.current_plan or {}).get("plan_id"),
                        playbook_id=trigger_dict.get("playbook_id"),
                        template_id=(self.current_plan or {}).get("_retrieved_template_id"),
                        created_at=workflow.now(),
                    )
            except Exception as _exc:
                workflow.logger.warning(
                    "Exit contract via TriggerCondition failed for %s (%s): %s — fallback",
                    _sym, trigger_id, _exc,
                )

            # Fallback: build directly from resolved price values
            if _exit_contract is None:
                try:
                    _legs = []
                    if target_price_abs is not None:
                        _legs.append(ExitLeg(
                            kind="full_exit",
                            trigger_mode="price_level",
                            fraction=1.0,
                            price_abs=target_price_abs,
                        ))
                    _exit_contract = PositionExitContract(
                        position_id=_pos_id,
                        symbol=_sym,
                        side=_dir,
                        created_at=workflow.now(),
                        source_trigger_id=trigger_id or f"trigger_{_pos_id}",
                        source_category=order.get("trigger_category"),
                        entry_price=fill_price,
                        initial_qty=float(quantity),
                        stop_price_abs=stop_price_abs,
                        target_legs=_legs,
                        remaining_qty=float(quantity),
                    )
                except Exception as _exc2:
                    workflow.logger.warning(
                        "Fallback exit contract creation failed for %s: %s — entry proceeds without contract",
                        _sym, _exc2,
                    )

            if _exit_contract is not None:
                self.exit_contracts[_sym] = _exit_contract.model_dump(mode="json")
                fill_payload["exit_contract_id"] = _exit_contract.contract_id

        await ledger_handle.signal("record_fill", fill_payload)

        # R65: Exit contract enforcement — track which plan opened each position.
        _fill_symbol = order.get("symbol", "")
        _fill_intent = order.get("intent")
        if _fill_intent == "entry":
            _origin_plan_id = (self.current_plan or {}).get("plan_id")
            if _origin_plan_id and _fill_symbol:
                self.position_originating_plans[_fill_symbol] = _origin_plan_id
        elif _fill_intent in ("exit", "flat", "conflict_exit", "conflict_reverse"):
            # Clear originating plan when position closes
            self.position_originating_plans.pop(_fill_symbol, None)

        # Runbook 61: transition state machine on entry fill (IDLE → ... → HOLD_LOCK).
        if order.get("intent") == "entry":
            self._position_opened_since_last_eval = True
            try:
                _sm = PolicyStateMachine()
                _sm_record = PolicyStateMachineRecord.model_validate(
                    self.policy_state_machine_record or {}
                )
                if _sm_record.current_state == "IDLE":
                    _sm_record, _ = _sm.arm_thesis(_sm_record)
                if _sm_record.current_state == "THESIS_ARMED":
                    _fill_pos_id = order.get("trigger_id") or f"pos_{int(workflow.now().timestamp() * 1000)}"
                    _sm_record, _ = _sm.activate_position(_sm_record, position_id=_fill_pos_id)
                if _sm_record.current_state == "POSITION_OPEN":
                    _sm_record = _sm.lock_hold(_sm_record)
                self.policy_state_machine_record = _sm_record.model_dump()
            except Exception as _sm_exc:
                workflow.logger.warning("State machine fill transition failed (non-fatal): %s", _sm_exc)

        # A2: Record fill drift telemetry to signal_ledger (non-fatal, fire-and-forget).
        if _signal_id and _signal_ts and _signal_entry_price is not None:
            try:
                _fill_ts_iso = workflow.now().isoformat()
                await workflow.execute_activity(
                    record_signal_fill_activity,
                    args=[_signal_id, fill_price, _fill_ts_iso, _signal_ts, _signal_entry_price],
                    schedule_to_close_timeout=timedelta(seconds=10),
                    retry_policy=RetryPolicy(maximum_attempts=1),
                )
            except Exception:
                pass  # fill drift telemetry is non-critical

        # Emit position_exit_contract_created event for auditable contract provenance
        if _exit_contract is not None:
            try:
                await workflow.execute_activity(
                    emit_paper_trading_event_activity,
                    args=[
                        self.session_id,
                        "position_exit_contract_created",
                        {
                            "contract_id": _exit_contract.contract_id,
                            "position_id": _exit_contract.position_id,
                            "symbol": _exit_contract.symbol,
                            "side": _exit_contract.side,
                            "entry_price": _exit_contract.entry_price,
                            "stop_price_abs": _exit_contract.stop_price_abs,
                            "target_legs_count": len(_exit_contract.target_legs),
                            "has_time_exit": _exit_contract.time_exit is not None,
                            "source_trigger_id": _exit_contract.source_trigger_id,
                            "source_category": _exit_contract.source_category,
                            "exit_class": "strategy_contract",
                        },
                    ],
                    schedule_to_close_timeout=timedelta(seconds=10),
                    retry_policy=RetryPolicy(maximum_attempts=1),
                )
            except Exception:
                pass  # event telemetry is non-critical; never crash the workflow

        # Emit event with stop/target for activity feed
        event_payload = dict(order)
        if stop_price_abs is not None:
            event_payload["stop_price"] = stop_price_abs
        if target_price_abs is not None:
            event_payload["target_price"] = target_price_abs
        await workflow.execute_activity(
            emit_paper_trading_event_activity,
            args=[self.session_id, "order_executed", event_payload],
            schedule_to_close_timeout=timedelta(seconds=10),
        )

        workflow.logger.info(f"Executed order: {order['side']} {order['quantity']} {order['symbol']} @ {order['price']}")

    async def _discover_new_symbols(self) -> None:
        """Discover and add new trading symbols."""
        discovered = await workflow.execute_activity(
            discover_symbols_activity,
            args=["coinbase", self.min_volume_24h, "USD"],
            schedule_to_close_timeout=timedelta(minutes=2),
            retry_policy=RetryPolicy(maximum_attempts=2),
        )

        new_symbols = [s for s in discovered if s not in self.symbols]
        if new_symbols:
            self.symbols.extend(new_symbols)
            workflow.logger.info(f"Discovered {len(new_symbols)} new symbols: {new_symbols}")

            await workflow.execute_activity(
                emit_paper_trading_event_activity,
                args=[self.session_id, "symbols_discovered", {
                    "new_symbols": new_symbols,
                    "total_symbols": len(self.symbols),
                }],
                schedule_to_close_timeout=timedelta(seconds=10),
            )

    async def _record_equity_snapshot(self) -> None:
        """Record a periodic equity snapshot for charting."""
        try:
            portfolio_state = await workflow.execute_activity(
                query_ledger_portfolio_activity,
                args=[self.ledger_workflow_id],
                schedule_to_close_timeout=timedelta(seconds=10),
            )

            snapshot = {
                "timestamp": workflow.now().isoformat(),
                "cash": portfolio_state.get("cash", 0),
                "total_equity": portfolio_state.get("total_equity", 0),
                "positions": portfolio_state.get("positions", {}),
                "unrealized_pnl": portfolio_state.get("unrealized_pnl", 0),
                "realized_pnl": portfolio_state.get("realized_pnl", 0),
            }

            # Keep only last 2000 snapshots to prevent unbounded growth
            if len(self.equity_history) >= 2000:
                self.equity_history = self.equity_history[-1500:]

            self.equity_history.append(snapshot)
            self.last_equity_snapshot = workflow.now()
        except Exception as e:
            workflow.logger.warning(f"Failed to record equity snapshot: {e}")

    def _snapshot_state(self) -> Dict[str, Any]:
        """Create state snapshot for continue-as-new."""
        return SessionState(
            session_id=self.session_id,
            ledger_workflow_id=self.ledger_workflow_id,
            symbols=self.symbols,
            strategy_prompt=self.strategy_prompt,
            plan_interval_hours=self.plan_interval_hours,
            indicator_timeframe=self.indicator_timeframe,
            direction_bias=self.direction_bias,
            replan_on_day_boundary=self.replan_on_day_boundary,
            current_plan=self.current_plan,
            last_plan_time=self.last_plan_time.isoformat() if self.last_plan_time else None,
            cycle_count=self.cycle_count,
            stopped=self.stopped,
            enable_symbol_discovery=self.enable_symbol_discovery,
            min_volume_24h=self.min_volume_24h,
            plan_history=self.plan_history,
            equity_history=self.equity_history[-500:],  # Keep last 500 snapshots across continue-as-new
            exit_binding_mode=self.exit_binding_mode,
            conflicting_signal_policy=self.conflicting_signal_policy,
            trigger_rule_edits=self.trigger_rule_edits,
            last_eval_candle_by_tf=dict(self.last_eval_candle_by_tf),
            research=self.research,
            active_experiments=list(self.active_experiments),
            structure_history=self._bounded_structure_history_snapshot(),
            exit_contracts=dict(self.exit_contracts),
            # Policy loop cadence (Runbook 61)
            policy_state_machine_record=dict(self.policy_state_machine_record) if self.policy_state_machine_record else None,
            regime_detector_state=dict(self.regime_detector_state) if self.regime_detector_state else None,
            last_policy_eval_at=self.last_policy_eval_at,
            # Adaptive management + episode memory (Runbook 63)
            adaptive_management_states=dict(self.adaptive_management_states),
            episode_memory_store_state=list(self.episode_memory_store_state)[-100:],
            # Exit contract enforcement (Runbook 65)
            position_originating_plans=dict(self.position_originating_plans),
        ).model_dump()

    def _restore_state(self, state: Dict[str, Any]) -> None:
        """Restore state from continue-as-new snapshot."""
        parsed = SessionState.model_validate(state)
        self.session_id = parsed.session_id
        self.ledger_workflow_id = parsed.ledger_workflow_id or MOCK_LEDGER_WORKFLOW_ID
        self.symbols = parsed.symbols
        self.strategy_prompt = parsed.strategy_prompt
        self.plan_interval_hours = parsed.plan_interval_hours
        self.indicator_timeframe = parsed.indicator_timeframe
        self.direction_bias = parsed.direction_bias
        self.replan_on_day_boundary = parsed.replan_on_day_boundary
        self.current_plan = parsed.current_plan
        self.last_plan_time = datetime.fromisoformat(parsed.last_plan_time) if parsed.last_plan_time else None
        self.cycle_count = parsed.cycle_count
        self.stopped = parsed.stopped
        self.enable_symbol_discovery = parsed.enable_symbol_discovery
        self.min_volume_24h = parsed.min_volume_24h
        self.plan_history = parsed.plan_history
        self.equity_history = parsed.equity_history
        self.exit_binding_mode = parsed.exit_binding_mode
        self.conflicting_signal_policy = parsed.conflicting_signal_policy
        self.trigger_rule_edits = parsed.trigger_rule_edits
        self.last_eval_candle_by_tf = dict(parsed.last_eval_candle_by_tf)
        self.research = parsed.research
        self.active_experiments = list(parsed.active_experiments)
        self.structure_history = {
            str(sym).upper(): list(snaps or [])
            for sym, snaps in (parsed.structure_history or {}).items()
        }
        self.exit_contracts = dict(parsed.exit_contracts or {})
        # Policy loop cadence (Runbook 61)
        self.policy_state_machine_record = dict(parsed.policy_state_machine_record or {})
        self.regime_detector_state = dict(parsed.regime_detector_state) if parsed.regime_detector_state else None
        self.last_policy_eval_at = parsed.last_policy_eval_at
        self._position_opened_since_last_eval = False  # reset on continue-as-new
        self._position_closed_since_last_eval = False  # reset on continue-as-new
        # Adaptive management + episode memory (Runbook 63)
        self.adaptive_management_states = dict(parsed.adaptive_management_states or {})
        self.episode_memory_store_state = list(parsed.episode_memory_store_state or [])
        # Exit contract enforcement (Runbook 65)
        self.position_originating_plans = dict(parsed.position_originating_plans or {})

    # -------------------------------------------------------------------------
    # Structure snapshot history helpers (UI time-travel)
    # -------------------------------------------------------------------------

    @staticmethod
    def _filter_indicator_dict(source: Dict[str, Any], symbol: Optional[str]) -> Dict[str, Any]:
        if symbol:
            snap = source.get(symbol)
            return {symbol: snap} if snap is not None else {}
        return dict(source)

    @staticmethod
    def _snapshot_as_of_str(snapshot: Any) -> Optional[str]:
        if not isinstance(snapshot, dict):
            return None
        as_of = snapshot.get("as_of")
        if as_of is None:
            return None
        if isinstance(as_of, datetime):
            return as_of.isoformat()
        return str(as_of)

    @staticmethod
    def _parse_ts_like(value: Any) -> Optional[datetime]:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        raw = str(value).strip()
        if not raw:
            return None
        # Support common ISO 'Z' suffix in API timestamps.
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            return None

    def _append_structure_history(self, snapshots: Dict[str, Any], *, origin: str) -> None:
        """Store a bounded, per-symbol history of indicator snapshots for UI lookup."""
        if not isinstance(snapshots, dict):
            return

        per_symbol_limit = 720
        for sym_raw, snap_raw in snapshots.items():
            sym = str(sym_raw).upper()
            if not isinstance(snap_raw, dict):
                continue
            snap = dict(snap_raw)
            snap["_snapshot_origin"] = origin

            history = self.structure_history.setdefault(sym, [])
            current_as_of = self._snapshot_as_of_str(snap)
            current_tf = str(snap.get("timeframe") or "")
            current_origin = str(snap.get("_snapshot_origin") or "")

            if history:
                last = history[-1]
                if (
                    self._snapshot_as_of_str(last) == current_as_of
                    and str(last.get("timeframe") or "") == current_tf
                    and str(last.get("_snapshot_origin") or "") == current_origin
                ):
                    history[-1] = snap
                else:
                    history.append(snap)
            else:
                history.append(snap)

            if len(history) > per_symbol_limit:
                self.structure_history[sym] = history[-per_symbol_limit:]

    def _bounded_structure_history_snapshot(self) -> Dict[str, List[Dict[str, Any]]]:
        out: Dict[str, List[Dict[str, Any]]] = {}
        for sym, history in self.structure_history.items():
            if not history:
                continue
            out[str(sym).upper()] = list(history[-720:])
        return out

    def _lookup_structure_at_or_before(
        self,
        requested_as_of: str,
        symbol: Optional[str] = None,
    ) -> tuple[Dict[str, Any], Dict[str, str]]:
        """Resolve snapshots at-or-before ``requested_as_of`` using bounded history."""
        target_ts = self._parse_ts_like(requested_as_of)
        if target_ts is None:
            latest_source = self.last_indicators_live or self.last_indicators
            indicators = self._filter_indicator_dict(latest_source, symbol)
            resolved = {
                k: ts for k, ts in (
                    (k, self._snapshot_as_of_str(v)) for k, v in indicators.items()
                ) if ts
            }
            return indicators, resolved

        latest_source = self.last_indicators_live or self.last_indicators
        symbols = [symbol] if symbol else sorted(set(self.structure_history.keys()) | set(latest_source.keys()))
        indicators: Dict[str, Any] = {}
        resolved: Dict[str, str] = {}

        for sym in symbols:
            if not sym:
                continue
            history = self.structure_history.get(sym, [])
            best_snapshot: Optional[Dict[str, Any]] = None
            best_ts: Optional[datetime] = None

            for snap in history:
                snap_ts = self._parse_ts_like(self._snapshot_as_of_str(snap))
                if snap_ts is None or snap_ts > target_ts:
                    continue
                if best_ts is None or snap_ts > best_ts:
                    best_snapshot = snap
                    best_ts = snap_ts

            # Fallback to latest snapshot only when there is no recorded history yet
            # (e.g., older sessions or immediately after startup).
            if best_snapshot is None and not history:
                latest = latest_source.get(sym)
                if isinstance(latest, dict):
                    best_snapshot = dict(latest)

            if best_snapshot is not None:
                indicators[sym] = best_snapshot
                ts_str = self._snapshot_as_of_str(best_snapshot)
                if ts_str:
                    resolved[sym] = ts_str

        return indicators, resolved
