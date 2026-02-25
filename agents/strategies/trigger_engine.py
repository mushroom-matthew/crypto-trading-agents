"""Trigger evaluation engine that turns LLM plans into deterministic orders."""

from __future__ import annotations

import hashlib
import logging
import math
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Iterable, List, Literal, Mapping

logger = logging.getLogger(__name__)

from schemas.llm_strategist import AssetState, IndicatorSnapshot, PortfolioState, StrategyPlan, TriggerCondition
from schemas.judge_feedback import JudgeConstraints
from schemas.learning_gate import LearningGateStatus
from trading_core.execution_engine import BlockReason

from .risk_engine import RiskEngine
from .rule_dsl import EvaluationTrace, MissingIndicatorError, RuleEvaluator, RuleSyntaxError
from .plan_validator import check_exit_rule_for_tautology

from .trade_risk import TradeRiskEvaluator

ExitBindingMode = Literal["none", "category"]
ConflictResolution = Literal["ignore", "exit", "reverse", "defer"]


@dataclass
class TriggerEvaluationSample:
    """A sampled trigger evaluation for debugging."""
    timestamp: datetime
    trigger_id: str
    symbol: str
    rule_type: Literal["entry", "exit"]
    trace: EvaluationTrace
    bar_close: float
    position: Literal["long", "short", "flat"]


@dataclass(frozen=True)
class Bar:
    symbol: str
    timeframe: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class Order:
    symbol: str
    side: Literal["buy", "sell"]
    quantity: float
    price: float
    timeframe: str
    reason: str
    timestamp: datetime
    stop_distance: float | None = None
    target_distance: float | None = None
    trail_distance: float | None = None
    trail_activation_r: float | None = None
    emergency: bool = False
    cooldown_recommendation_bars: int | None = None
    trigger_category: str | None = None
    intent: Literal["entry", "exit", "flat", "conflict_exit", "conflict_reverse"] | None = None
    learning_book: bool = False
    experiment_id: str | None = None
    experiment_variant: str | None = None
    exit_fraction: float | None = None  # Partial exit fraction (0 < f <= 1.0) for risk_reduce


class TriggerEngine:
    """Evaluates TriggerCondition strings on every bar walking forward."""

    # Confidence grade priority mapping (higher = more priority)
    CONFIDENCE_PRIORITY = {"A": 3, "B": 2, "C": 1, None: 0}

    def __init__(
        self,
        plan: StrategyPlan,
        risk_engine: RiskEngine,
        evaluator: RuleEvaluator | None = None,
        trade_risk: TradeRiskEvaluator | None = None,
        stop_distance_resolver: Callable[[TriggerCondition, IndicatorSnapshot, Bar], float | None] | None = None,
        min_rr_ratio: float = 1.2,
        confidence_override_threshold: Literal["A", "B", "C"] | None = "A",
        min_hold_bars: int = 4,
        trade_cooldown_bars: int = 2,
        debug_sample_rate: float = 0.0,
        debug_max_samples: int = 100,
        prioritize_by_confidence: bool = True,
        max_triggers_per_symbol_per_bar: int = 1,
        priority_skip_confidence_threshold: Literal["A", "B", "C"] | None = None,
        judge_constraints: JudgeConstraints | None = None,
        exit_binding_mode: ExitBindingMode = "none",
        conflicting_signal_policy: ConflictResolution = "reverse",
        risk_off_latch: bool = False,
    ) -> None:
        """Initialize the trigger engine.

        Args:
            plan: Strategy plan with triggers
            risk_engine: Risk engine for position sizing
            evaluator: Rule evaluator (optional)
            trade_risk: Trade risk evaluator (optional)
            min_rr_ratio: Minimum risk-to-reward ratio required for entry when
                target_anchor_type is set (default 1.2). Entries whose prospective R:R
                falls below this threshold are blocked with reason "insufficient_rr".
            confidence_override_threshold: Minimum confidence grade for entry to override exit.
                "A" = only A-grade entries override exits (most conservative)
                "B" = A or B grade entries override exits
                "C" = all graded entries override exits
                None = exit always has priority (no confidence override)
            min_hold_bars: Minimum bars to hold a position before allowing exit (default: 4).
                This prevents rapid flip-flops by requiring positions to be held for a minimum time.
            trade_cooldown_bars: Minimum bars between trades for the same symbol (default: 2).
                This prevents overtrading by enforcing a cooldown period after each trade.
            debug_sample_rate: Probability (0.0-1.0) of sampling trigger evaluations for debugging.
                Set to 0.0 to disable sampling (default). Set to 1.0 to sample all evaluations.
            debug_max_samples: Maximum number of samples to collect (default: 100).
            prioritize_by_confidence: If True, evaluate triggers in confidence order (A→B→C)
                and stop evaluating once a trigger fires for a symbol. Prevents competing signals.
            max_triggers_per_symbol_per_bar: Maximum triggers to fire per symbol per bar (default: 1).
                Prevents cascade of competing signals.
            priority_skip_confidence_threshold: Minimum confidence grade to bypass priority skips
                once max_triggers_per_symbol_per_bar is reached. None disables bypass.
            judge_constraints: Optional JudgeConstraints with disabled_trigger_ids and
                disabled_categories to block specific triggers from firing.
            exit_binding_mode: Exit binding policy ("none" or "category").
            conflicting_signal_policy: Resolver policy when opposing entries fire ("ignore", "exit", "reverse", "defer").
            risk_off_latch: When True, risk_off exits get Tier 1 priority (preempt entries).
                When False, risk_off competes like normal exits unless plan regime is "risk_off".
        """
        self.plan = plan
        self.risk_engine = risk_engine
        self.evaluator = evaluator or RuleEvaluator()
        self.trade_risk = trade_risk or TradeRiskEvaluator(risk_engine)
        self.stop_distance_resolver = stop_distance_resolver
        self.min_rr_ratio = max(0.0, min_rr_ratio)
        self.judge_constraints = judge_constraints
        self.confidence_override_threshold = confidence_override_threshold
        self.min_hold_bars = max(0, min_hold_bars)
        self.trade_cooldown_bars = max(0, trade_cooldown_bars)
        self.prioritize_by_confidence = prioritize_by_confidence
        self.max_triggers_per_symbol_per_bar = max(1, max_triggers_per_symbol_per_bar)
        self.priority_skip_confidence_threshold = priority_skip_confidence_threshold
        self.exit_binding_mode = exit_binding_mode
        self.conflicting_signal_policy = conflicting_signal_policy
        self.risk_off_latch = risk_off_latch
        self._unknown_identifier_warnings: set[tuple[str, str, tuple[str, ...]]] = set()
        # Store trigger confidence for deduplication
        self._trigger_confidence: dict[str, Literal["A", "B", "C"] | None] = {
            t.id: t.confidence_grade for t in plan.triggers
        }
        # Pre-sort triggers by confidence for prioritized evaluation
        self._sorted_triggers = self._sort_triggers_by_confidence(plan.triggers)
        # Track position entry bar index for minimum hold enforcement
        self._position_entry_bar: dict[str, int] = {}
        # Track last trade bar index for cooldown enforcement
        self._last_trade_bar: dict[str, int] = {}
        # Track last entry timestamp to prevent same-bar emergency exits
        self._last_entry_timestamp: dict[str, datetime] = {}
        # Current bar counter
        self._bar_counter: int = 0
        # Debug sampling configuration
        self.debug_sample_rate = max(0.0, min(1.0, debug_sample_rate))
        self.debug_max_samples = debug_max_samples
        self.evaluation_samples: List[TriggerEvaluationSample] = []
        # Hold rule suppression tracking (Runbook 23)
        self._hold_suppression_counts: dict[str, int] = {}  # trigger_id -> consecutive suppression count
        self.hold_suppression_warnings: List[str] = []  # warnings emitted for sustained suppression
        # Per-trigger fire rate tracking (Runbook 25)
        self._trigger_eval_counts: dict[str, int] = {}  # trigger_id -> total evaluations
        self._trigger_fire_counts: dict[str, int] = {}  # trigger_id -> total fires

    def _sort_triggers_by_confidence(self, triggers: Iterable[TriggerCondition]) -> List[TriggerCondition]:
        """Sort triggers by confidence grade (A first, then B, then C, then None)."""
        return sorted(
            triggers,
            key=lambda t: self.CONFIDENCE_PRIORITY.get(t.confidence_grade, 0),
            reverse=True,  # Highest priority first
        )

    def set_risk_off_latch(self, active: bool) -> None:
        """Set the risk_off latch state.

        When active, risk_off exits get Tier 1 priority (preempt entries).
        """
        self.risk_off_latch = active

    def _risk_off_has_priority(self) -> bool:
        """Check if risk_off exits should have Tier 1 priority.

        Returns True if:
        - risk_off_latch is active, OR
        - plan regime is "risk_off" (a regime we don't currently use, but future-proofed)

        When True, risk_off exits preempt entries in dedup.
        When False, risk_off competes like normal exits via confidence-based override.
        """
        if self.risk_off_latch:
            return True
        # Future: could also check if plan.regime == "risk_off" but that regime
        # doesn't exist in the current schema (bull, bear, range, high_vol, mixed)
        return False

    def _context(
        self,
        indicator: IndicatorSnapshot,
        asset_state: AssetState | None,
        market_structure: dict[str, float | str | None] | None = None,
        portfolio: PortfolioState | None = None,
        position_meta: Mapping[str, dict[str, Any]] | None = None,
        tick_snapshot: "Any | None" = None,
    ) -> dict[str, float | str | None]:
        """Build evaluation context, including cross-timeframe aliases."""

        def _alias_key(key: str) -> str | None:
            parts = key.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                return parts[0]
            return None

        context = indicator.model_dump()
        # Deterministic pseudo-random scalar for control strategies.
        rand_key = f"{indicator.symbol}|{indicator.timeframe}|{indicator.as_of.isoformat()}"
        rand_hash = hashlib.sha256(rand_key.encode("utf-8")).hexdigest()
        context["rand_u"] = (int(rand_hash[:16], 16) % 1_000_000) / 1_000_000.0
        for key, value in list(context.items()):
            alias = _alias_key(key)
            if alias and alias not in context:
                context[alias] = value
        if context.get("bollinger_middle") is None:
            upper = context.get("bollinger_upper")
            lower = context.get("bollinger_lower")
            if upper is not None and lower is not None:
                context["bollinger_middle"] = (upper + lower) / 2.0
        if asset_state:
            context["trend_state"] = asset_state.trend_state
            context["vol_state"] = asset_state.vol_state
            for snapshot in asset_state.indicators:
                prefix = f"tf_{snapshot.timeframe.replace('-', '_')}"
                snapshot_dict = snapshot.model_dump()
                for key, value in snapshot_dict.items():
                    if key in {"symbol", "timeframe", "as_of"}:
                        continue
                    if key == "bollinger_upper":
                        other = snapshot_dict.get("bollinger_lower")
                        if other is not None:
                            context[f"{prefix}_bollinger_middle"] = (value + other) / 2.0
                    context[f"{prefix}_{key}"] = value
                    alias = _alias_key(key)
                    if alias:
                        context[f"{prefix}_{alias}"] = value
                # Provide derived trend/vol states per timeframe for LLM flexibility.
                context[f"{prefix}_trend_state"] = self._trend_state_from_snapshot(snapshot)
                context[f"{prefix}_vol_state"] = self._vol_state_from_snapshot(snapshot)
        ms_keys = ("nearest_support", "nearest_resistance", "distance_to_support_pct", "distance_to_resistance_pct", "trend")
        if market_structure:
            # Surface market-structure telemetry fields directly for rule expressions.
            for key in ms_keys:
                if key in market_structure:
                    context[key] = market_structure.get(key)
            if "recent_tests" in market_structure:
                context["recent_tests"] = market_structure.get("recent_tests") or []
        # Ensure keys exist to avoid unknown identifier errors when no snapshot is available.
        for key in ms_keys:
            context.setdefault(key, None)
        context.setdefault("recent_tests", [])
        if portfolio:
            position = self._position_direction(indicator.symbol, portfolio)
            context["position"] = position
            # Boolean position indicators (preferred over string comparisons)
            # These avoid quote-escaping issues in trigger expressions
            context["is_flat"] = position == "flat"
            context["is_long"] = position == "long"
            context["is_short"] = position == "short"
            qty = portfolio.positions.get(indicator.symbol, 0.0)
            context["position_qty"] = qty
            context["position_value"] = qty * (indicator.close or 0.0)
            entry_price = None
            entry_side = None
            opened_at = None
            if position_meta:
                meta = position_meta.get(indicator.symbol) or {}
                entry_price = meta.get("entry_price")
                entry_side = meta.get("entry_side")
                opened_at = meta.get("opened_at")
                context["entry_price"] = entry_price
                context["avg_entry_price"] = entry_price
                context["entry_side"] = entry_side
                context["position_opened_at"] = opened_at
            opened_at_ts = opened_at
            if isinstance(opened_at_ts, str):
                try:
                    opened_at_ts = datetime.fromisoformat(opened_at_ts)
                except ValueError:
                    opened_at_ts = None
            if opened_at_ts and indicator.as_of:
                if opened_at_ts.tzinfo is None and indicator.as_of.tzinfo is not None:
                    opened_at_ts = opened_at_ts.replace(tzinfo=indicator.as_of.tzinfo)
                if indicator.as_of.tzinfo is None and opened_at_ts.tzinfo is not None:
                    current_ts = indicator.as_of.replace(tzinfo=opened_at_ts.tzinfo)
                else:
                    current_ts = indicator.as_of
                if current_ts is not None:
                    age_minutes = (current_ts - opened_at_ts).total_seconds() / 60.0
                    if age_minutes >= 0:
                        context["position_age_minutes"] = age_minutes
                        context["position_age_hours"] = age_minutes / 60.0
                        context["holding_minutes"] = age_minutes
                        context["holding_hours"] = age_minutes / 60.0
                        context["time_in_trade_min"] = age_minutes
                        context["time_in_trade_hours"] = age_minutes / 60.0
            if entry_price and indicator.close:
                if entry_side == "short" or qty < 0:
                    pnl_pct = (entry_price - indicator.close) / entry_price * 100.0
                    pnl_abs = (entry_price - indicator.close) * abs(qty)
                else:
                    pnl_pct = (indicator.close - entry_price) / entry_price * 100.0
                    pnl_abs = (indicator.close - entry_price) * abs(qty)
                context["unrealized_pnl_pct"] = pnl_pct
                context["unrealized_pnl_abs"] = pnl_abs
                context["unrealized_pnl"] = pnl_abs
                context["position_pnl_pct"] = pnl_pct
                context["position_pnl_abs"] = pnl_abs
            else:
                # Seed common PnL fields to avoid expression errors for unprompted LLM outputs.
                context.setdefault("unrealized_pnl_pct", None)
                context.setdefault("unrealized_pnl_abs", None)
                context.setdefault("unrealized_pnl", None)
                context.setdefault("position_pnl_pct", None)
                context.setdefault("position_pnl_abs", None)
            for key in (
                "position_age_minutes",
                "position_age_hours",
                "holding_minutes",
                "holding_hours",
                "time_in_trade_min",
                "time_in_trade_hours",
            ):
                context.setdefault(key, None)
        # Runbook 42: level-anchored stop/target identifiers
        # Read from position_meta (populated at fill time by _resolve_stop_price_anchored)
        active_meta = (position_meta or {}).get(indicator.symbol) or {} if position_meta else {}
        active_stop = active_meta.get("stop_price_abs")
        active_target = active_meta.get("target_price_abs")
        # entry_side is "long" or "short"; written at position open time
        pos_direction = active_meta.get("entry_side", "long")
        is_flat = context.get("is_flat", True)
        current_close = indicator.close or 0.0

        # Direction-aware stop_hit / target_hit (canonical — correct for longs AND shorts)
        if active_stop and not is_flat:
            if pos_direction == "short":
                context["stop_hit"] = bool(current_close > active_stop)
            else:
                context["stop_hit"] = bool(current_close < active_stop)
        else:
            context["stop_hit"] = False

        if active_target and not is_flat:
            if pos_direction == "short":
                context["target_hit"] = bool(current_close < active_target)
            else:
                context["target_hit"] = bool(current_close > active_target)
        else:
            context["target_hit"] = False

        # Backward-compat aliases (same value as stop_hit / target_hit)
        context["below_stop"] = context["stop_hit"]
        context["above_target"] = context["target_hit"]
        context["stop_price"] = active_stop or 0.0
        context["target_price"] = active_target or 0.0
        context["stop_distance_pct"] = (
            abs(current_close - active_stop) / current_close * 100.0
            if (active_stop and current_close)
            else 0.0
        )
        context["target_distance_pct"] = (
            abs(active_target - current_close) / current_close * 100.0
            if (active_target and current_close)
            else 0.0
        )

        # Runbook 45: R-tracking identifiers (written by _advance_trade_state into position_meta)
        r_current = active_meta.get("current_R", 0.0) or 0.0
        context["current_R"] = r_current
        context["mfe_r"] = active_meta.get("mfe_r", 0.0) or 0.0
        context["mae_r"] = active_meta.get("mae_r", 0.0) or 0.0
        context["trade_state"] = active_meta.get("trade_state", "EARLY") or "EARLY"
        context["position_fraction"] = active_meta.get("position_fraction", 1.0) if active_meta.get("position_fraction") is not None else 1.0
        context["r1_reached"] = r_current >= 1.0
        context["r2_reached"] = r_current >= 2.0
        context["r3_reached"] = r_current >= 3.0

        # R49: thread TickSnapshot provenance into evaluation context (optional; None when absent)
        if tick_snapshot is not None:
            prov = getattr(tick_snapshot, "provenance", None)
            context["snapshot_id"] = getattr(prov, "snapshot_id", None) if prov else None
            context["snapshot_hash"] = getattr(prov, "snapshot_hash", None) if prov else None
        else:
            context.setdefault("snapshot_id", None)
            context.setdefault("snapshot_hash", None)

        return context

    def _trend_state_from_snapshot(self, snapshot: IndicatorSnapshot) -> str:
        sma_short = snapshot.sma_short
        sma_medium = snapshot.sma_medium
        sma_long = snapshot.sma_long
        if sma_short and sma_medium and sma_long:
            if sma_short > sma_medium > sma_long:
                return "uptrend"
            if sma_short < sma_medium < sma_long:
                return "downtrend"
        ema_short = snapshot.ema_short
        ema_medium = snapshot.ema_medium
        if ema_short and ema_medium:
            if ema_short > ema_medium * 1.005:
                return "uptrend"
            if ema_short < ema_medium * 0.995:
                return "downtrend"
        return "sideways"

    def _vol_state_from_snapshot(self, snapshot: IndicatorSnapshot) -> str:
        atr = snapshot.atr_14 or 0.0
        realized = snapshot.realized_vol_short or 0.0
        price = snapshot.close or 1.0
        atr_ratio = atr / price if price else 0.0
        vol_metric = max(atr_ratio, realized)
        if vol_metric < 0.015:
            return "low"
        if vol_metric < 0.03:
            return "normal"
        if vol_metric < 0.07:
            return "high"
        return "extreme"

    def _position_direction(self, symbol: str, portfolio: PortfolioState) -> Literal["long", "short", "flat"]:
        qty = portfolio.positions.get(symbol, 0.0)
        if qty > 0:
            return "long"
        if qty < 0:
            return "short"
        return "flat"

    def _entry_category(self, position_meta: Mapping[str, dict[str, Any]] | None, symbol: str) -> str | None:
        if not position_meta:
            return None
        meta = position_meta.get(symbol) or {}
        return meta.get("entry_category") or meta.get("category")

    def _entry_trigger_id(self, position_meta: Mapping[str, dict[str, Any]] | None, symbol: str) -> str | None:
        if not position_meta:
            return None
        meta = position_meta.get(symbol) or {}
        return meta.get("entry_trigger_id") or meta.get("reason")

    def _exit_binding_allows(self, trigger: TriggerCondition, entry_category: str | None) -> bool:
        if self.exit_binding_mode == "none":
            return True
        if trigger.category == "emergency_exit":
            return True
        if trigger.direction == "exit" and getattr(trigger, "exit_binding_exempt", False):
            return True
        if not entry_category or not trigger.category:
            return True
        if self.exit_binding_mode == "category":
            return entry_category == trigger.category
        return True

    def _flatten_order(
        self,
        trigger: TriggerCondition,
        bar: Bar,
        portfolio: PortfolioState,
        reason: str,
        block_entries: List[dict] | None = None,
        intent: Literal["exit", "flat", "conflict_exit"] | None = None,
        fraction: float = 1.0,
    ) -> Order | None:
        """Create an exit order for a position.

        Args:
            trigger: The trigger condition that fired
            bar: Current price bar
            portfolio: Current portfolio state
            reason: Reason string for the order
            block_entries: Optional list to record blocked entries
            intent: Order intent (exit, flat, conflict_exit)
            fraction: Fraction of position to exit (0 < f <= 1.0). Default 1.0 = full exit.

        Returns:
            Order if position exists, None otherwise.
        """
        qty = portfolio.positions.get(trigger.symbol, 0.0)
        if abs(qty) <= 1e-9:
            return None

        # Calculate exit quantity based on fraction
        exit_qty = abs(qty) * fraction
        if exit_qty <= 1e-9:
            return None

        decision = self.trade_risk.evaluate(trigger, "flatten", bar.close, portfolio, None)
        if not decision.allowed and block_entries is not None and decision.reason:
            self._record_block(block_entries, trigger, decision.reason, "Flatten blocked unexpectedly", bar)
            return None
        side: Literal["buy", "sell"] = "sell" if qty > 0 else "buy"
        emergency = trigger.category == "emergency_exit"
        cooldown_bars = self._emergency_cooldown_bars() if emergency else None

        # Determine exit_fraction to pass to order (None for full exits)
        order_exit_fraction = fraction if fraction < 1.0 else None

        return Order(
            symbol=trigger.symbol,
            side=side,
            quantity=exit_qty,
            price=bar.close,
            timeframe=bar.timeframe,
            reason=reason,
            timestamp=bar.timestamp,
            emergency=emergency,
            cooldown_recommendation_bars=cooldown_bars,
            trigger_category=trigger.category,
            intent=intent or ("exit" if reason.endswith("_exit") else "flat"),
            learning_book=trigger.learning_book,
            experiment_id=trigger.experiment_id,
            exit_fraction=order_exit_fraction,
        )

    def _record_block(
        self,
        container: List[dict],
        trigger: TriggerCondition,
        reason: str,
        detail: str,
        bar: Bar,
        extra: dict[str, object] | None = None,
    ) -> None:
        payload = {
            "trigger_id": trigger.id,
            "symbol": trigger.symbol,
            "timeframe": bar.timeframe,
            "timestamp": bar.timestamp.isoformat(),
            "reason": reason,
            "detail": detail,
            "price": bar.close,
        }
        if extra:
            payload.update(extra)
        container.append(payload)

    def _unknown_identifiers(self, expr: str, context: Mapping[str, Any]) -> list[str]:
        identifiers = self.evaluator.extract_identifiers(expr)
        if not identifiers:
            return []
        allowed = set(context.keys()) | self.evaluator.allowed_names | {"True", "False", "true", "false", "None", "none"}
        unknown = [name for name in identifiers if name not in allowed and name.lower() not in allowed]
        return sorted(unknown)

    def _log_unknown_identifiers(
        self,
        trigger: TriggerCondition,
        rule_type: Literal["entry", "exit", "hold"],
        unknown: list[str],
        expr: str,
    ) -> None:
        if not unknown:
            return
        key = (trigger.id, rule_type, tuple(unknown))
        if key in self._unknown_identifier_warnings:
            return
        self._unknown_identifier_warnings.add(key)
        logger.warning(
            "Trigger %s %s_rule references unknown identifiers %s; expr=%s",
            trigger.id,
            rule_type,
            unknown,
            expr,
        )

    def _emergency_cooldown_bars(self) -> int:
        return max(1, self.trade_cooldown_bars, self.min_hold_bars)

    def _compute_target_distance(
        self,
        trigger: TriggerCondition,
        indicator: IndicatorSnapshot,
        entry_price: float,
        stop_distance: float,
        direction: str,
    ) -> float | None:
        """Return the projected distance from entry to target, or None if unresolvable.

        Mirrors the anchor logic in _resolve_target_price_anchored so the R:R
        check can run at order-generation time without importing backtesting code.
        """
        anchor = trigger.target_anchor_type
        if not anchor:
            return None

        def _to_float(value: Any) -> float | None:
            if value is None:
                return None
            try:
                parsed = float(value)
                return parsed if math.isfinite(parsed) else None
            except (TypeError, ValueError):
                return None

        def _apply_structure_cap(candidate: float | None) -> float | None:
            if candidate is None:
                return None

            mid_score = _to_float(getattr(indicator, "htf_price_vs_daily_mid", None))
            if mid_score is None or abs(mid_score) > 1.5:
                return candidate

            breakout_confirmed = _to_float(getattr(indicator, "breakout_confirmed", None)) or 0.0
            expansion_flag = _to_float(getattr(indicator, "expansion_flag", None)) or 0.0
            vol_burst = bool(getattr(indicator, "vol_burst", None))
            if breakout_confirmed > 0 or expansion_flag > 0 or vol_burst:
                return candidate

            if direction == "long":
                weekly_high = _to_float(getattr(indicator, "htf_5d_high", None))
                daily_high = _to_float(getattr(indicator, "htf_daily_high", None))
                ceiling = None
                if weekly_high:
                    if entry_price > weekly_high * 1.002:
                        return candidate
                    ceiling = weekly_high * 1.002
                elif daily_high:
                    if entry_price > daily_high * 1.002:
                        return candidate
                    ceiling = daily_high * 1.002
                if ceiling is None:
                    return candidate
                capped = min(candidate, ceiling)
                return capped if capped > entry_price else None

            if direction == "short":
                weekly_low = _to_float(getattr(indicator, "htf_5d_low", None))
                daily_low = _to_float(getattr(indicator, "htf_daily_low", None))
                floor = None
                if weekly_low:
                    if entry_price < weekly_low * 0.998:
                        return candidate
                    floor = weekly_low * 0.998
                elif daily_low:
                    if entry_price < daily_low * 0.998:
                        return candidate
                    floor = daily_low * 0.998
                if floor is None:
                    return candidate
                capped = max(candidate, floor)
                return capped if capped < entry_price else None

            return candidate

        target_price: float | None = None
        if anchor == "r_multiple_2":
            target_price = entry_price + 2.0 * stop_distance if direction == "long" else entry_price - 2.0 * stop_distance
        elif anchor == "r_multiple_3":
            target_price = entry_price + 3.0 * stop_distance if direction == "long" else entry_price - 3.0 * stop_distance
        elif anchor == "measured_move":
            upper = indicator.donchian_upper_short
            lower = indicator.donchian_lower_short
            if upper is not None and lower is not None and upper > lower:
                range_height = upper - lower
                target_price = entry_price + range_height if direction == "long" else entry_price - range_height
            else:
                return None
        elif anchor in {"htf_daily_high", "htf_daily_extreme"} and direction == "long":
            level = indicator.htf_daily_high
            target_price = level * 0.998 if level else None
        elif anchor in {"htf_5d_high", "htf_5d_extreme"} and direction == "long":
            level = indicator.htf_5d_high
            target_price = level * 0.998 if level else None
        elif anchor in {"htf_daily_low", "htf_daily_extreme"} and direction == "short":
            level = indicator.htf_daily_low
            target_price = level * 1.002 if level else None
        elif anchor in {"htf_5d_low", "htf_5d_extreme"} and direction == "short":
            level = indicator.htf_5d_low
            target_price = level * 1.002 if level else None
        else:
            return None

        target_price = _apply_structure_cap(target_price)
        if target_price is None:
            return None

        if direction == "long":
            return target_price - entry_price if target_price > entry_price else None
        if direction == "short":
            return entry_price - target_price if target_price < entry_price else None
        return None

    def _entry_order(
        self,
        trigger: TriggerCondition,
        indicator: IndicatorSnapshot,
        portfolio: PortfolioState,
        bar: Bar,
        block_entries: List[dict] | None = None,
        intent: Literal["entry", "conflict_reverse"] | None = None,
    ) -> Order | None:
        desired = trigger.direction
        current = self._position_direction(trigger.symbol, portfolio)
        if desired in {"flat", "exit", "flat_exit"}:
            if current == "flat":
                return None
            # Use exit_fraction for partial exits
            exit_fraction = trigger.exit_fraction if trigger.exit_fraction is not None else 1.0
            return self._flatten_order(
                trigger, bar, portfolio, f"{trigger.id}_flat", block_entries,
                fraction=exit_fraction,
            )
        if desired == current:
            return None
        stop_distance = None
        if self.stop_distance_resolver is not None:
            stop_distance = self.stop_distance_resolver(trigger, indicator, bar)
            if stop_distance is not None and stop_distance <= 0:
                stop_distance = None
        elif trigger.stop_loss_pct is not None:
            stop_distance = abs(bar.close * trigger.stop_loss_pct / 100.0)
        check = self.trade_risk.evaluate(trigger, "entry", bar.close, portfolio, indicator, stop_distance=stop_distance)
        if not check.allowed:
            if block_entries is not None and check.reason:
                detail = f"Risk constraint {check.reason} prevented sizing"
                self._record_block(block_entries, trigger, check.reason, detail, bar)
            return None
        side: Literal["buy", "sell"] = "buy" if desired == "long" else "sell"
        # R:R gate: when a target anchor is defined, the prospective reward/risk ratio
        # must meet the minimum threshold before capital is committed.
        if trigger.target_anchor_type and stop_distance and stop_distance > 0 and self.min_rr_ratio > 0:
            direction_str = "long" if side == "buy" else "short"
            target_dist = self._compute_target_distance(trigger, indicator, bar.close, stop_distance, direction_str)
            if target_dist is not None:
                rr = target_dist / stop_distance
                if rr < self.min_rr_ratio:
                    if block_entries is not None:
                        detail = (
                            f"R:R {rr:.2f} < minimum {self.min_rr_ratio:.2f} "
                            f"(stop={stop_distance:.4f}, target_dist={target_dist:.4f}, "
                            f"anchor={trigger.target_anchor_type})"
                        )
                        self._record_block(block_entries, trigger, "insufficient_rr", detail, bar)
                    return None
        return Order(
            symbol=trigger.symbol,
            side=side,
            quantity=check.quantity,
            price=bar.close,
            timeframe=bar.timeframe,
            reason=trigger.id,
            timestamp=bar.timestamp,
            stop_distance=stop_distance,
            trigger_category=trigger.category,
            intent=intent or "entry",
            learning_book=trigger.learning_book,
            experiment_id=trigger.experiment_id,
        )

    def on_bar(
        self,
        bar: Bar,
        indicator: IndicatorSnapshot,
        portfolio: PortfolioState,
        asset_state: AssetState | None = None,
        market_structure: dict[str, float | str | None] | None = None,
        position_meta: Mapping[str, dict[str, Any]] | None = None,
        learning_gate_status: LearningGateStatus | None = None,
    ) -> tuple[List[Order], List[dict]]:
        orders: List[Order] = []
        block_entries: List[dict] = []

        # Increment bar counter for cooldown/hold tracking
        self._bar_counter += 1

        # Check trade cooldown for this symbol
        if self._is_in_cooldown(bar.symbol):
            # Skip all triggers for this symbol during cooldown
            return orders, block_entries

        portfolio_base = portfolio
        portfolio_for_eval = portfolio_base
        context = self._context(
            indicator,
            asset_state,
            market_structure,
            portfolio_for_eval,
            position_meta,
        )

        # Track how many triggers have fired for this symbol this bar
        # (used for max_triggers_per_symbol_per_bar enforcement)
        triggers_fired_for_symbol = 0

        # Use sorted triggers if prioritize_by_confidence is enabled
        trigger_list = self._sorted_triggers if self.prioritize_by_confidence else self.plan.triggers

        for trigger in trigger_list:
            if trigger.symbol != bar.symbol or trigger.timeframe != bar.timeframe:
                continue

            # Per-trigger evaluation counting (Runbook 25)
            self._trigger_eval_counts[trigger.id] = self._trigger_eval_counts.get(trigger.id, 0) + 1

            # Enforce judge constraints: skip disabled triggers
            if self.judge_constraints:
                if trigger.id in self.judge_constraints.disabled_trigger_ids:
                    detail = f"Trigger {trigger.id} disabled by judge"
                    logger.info(
                        "Judge blocked trigger %s (reason: disabled_trigger_ids) at %s",
                        trigger.id, bar.timestamp.isoformat()
                    )
                    self._record_block(block_entries, trigger, BlockReason.SYMBOL_VETO.value, detail, bar)
                    continue
                if trigger.category and trigger.category in self.judge_constraints.disabled_categories:
                    detail = f"Category {trigger.category} disabled by judge"
                    logger.info(
                        "Judge blocked trigger %s (reason: disabled_categories, category=%s) at %s",
                        trigger.id, trigger.category, bar.timestamp.isoformat()
                    )
                    self._record_block(block_entries, trigger, BlockReason.CATEGORY.value, detail, bar)
                    continue

            # Block learning triggers when the learning gate is closed
            if trigger.learning_book and learning_gate_status is not None and not learning_gate_status.open:
                gate_reasons = ", ".join(learning_gate_status.reasons) if learning_gate_status.reasons else "gate_closed"
                detail = f"Learning gate closed: {gate_reasons}"
                self._record_block(block_entries, trigger, "learning_gate_closed", detail, bar)
                continue

            # Early exit if we've hit the max triggers for this symbol
            # EXCEPTIONS that bypass priority_skip:
            # - emergency_exit: safety interrupts are NEVER skipped
            # - risk_off (when latched): Tier 1 priority needs to participate in dedup
            if self.prioritize_by_confidence and triggers_fired_for_symbol >= self.max_triggers_per_symbol_per_bar:
                is_emergency = trigger.category == "emergency_exit"
                is_risk_off_with_priority = trigger.category == "risk_off" and self._risk_off_has_priority()
                if not is_emergency and not is_risk_off_with_priority and not self._priority_skip_bypass(trigger.confidence_grade):
                    # Log that we're skipping lower-confidence triggers
                    detail = f"Max triggers ({self.max_triggers_per_symbol_per_bar}) already fired for {bar.symbol}"
                    self._record_block(block_entries, trigger, "priority_skip", detail, bar)
                    continue

            current_position = self._position_direction(bar.symbol, portfolio_for_eval)
            context["position"] = current_position

            # Exit rule evaluation
            try:
                if trigger.category == "emergency_exit" and not (trigger.exit_rule or "").strip():
                    detail = "Emergency exit missing exit_rule"
                    self._record_block(
                        block_entries,
                        trigger,
                        "emergency_exit_missing_exit_rule",
                        detail,
                        bar,
                        extra={"cooldown_recommendation_bars": self._emergency_cooldown_bars()},
                    )
                    continue
                # Runtime failsafe: detect cross-timeframe ATR tautologies that would
                # cause the emergency exit to fire on every bar regardless of market
                # conditions.  This is belt-and-suspenders: the compile-time validator
                # in _generate_plan() should have already rejected such a plan, but this
                # prevents churn if a bad plan was persisted before the validator existed.
                if trigger.category == "emergency_exit" and trigger.exit_rule:
                    _tautologies = check_exit_rule_for_tautology(
                        trigger.exit_rule, trigger.timeframe
                    )
                    if _tautologies:
                        frags = "; ".join(t.fragment for t in _tautologies)
                        detail = (
                            f"Emergency exit suppressed: exit_rule contains ATR "
                            f"tautology (always-true cross-TF comparison). "
                            f"Flagged: {frags}"
                        )
                        self._record_block(
                            block_entries,
                            trigger,
                            "emergency_exit_tautology",
                            detail,
                            bar,
                        )
                        continue
                exit_fired = bool(trigger.exit_rule and self.evaluator.evaluate(trigger.exit_rule, context))
                # Debug sampling for exit rule
                self._maybe_sample_evaluation(
                    bar, trigger, "exit", trigger.exit_rule, context, current_position
                )
            except MissingIndicatorError as exc:
                detail = f"{exc}; exit_rule='{trigger.exit_rule}'"
                self._record_block(block_entries, trigger, BlockReason.MISSING_INDICATOR.value, detail, bar)
                continue
            except RuleSyntaxError as exc:
                unknown = self._unknown_identifiers(trigger.exit_rule, context)
                detail = f"{exc}; exit_rule='{trigger.exit_rule}'"
                extra = None
                if unknown:
                    detail = f"{detail}; unknown_identifiers={unknown}"
                    extra = {"unknown_identifiers": unknown}
                    self._log_unknown_identifiers(trigger, "exit", unknown, trigger.exit_rule)
                self._record_block(block_entries, trigger, BlockReason.EXPRESSION_ERROR.value, detail, bar, extra=extra)
                continue

            if exit_fired:
                if trigger.category == "emergency_exit":
                    last_entry_ts = self._last_entry_timestamp.get(trigger.symbol)
                    if last_entry_ts and last_entry_ts == bar.timestamp:
                        detail = "Emergency exit blocked on same bar as entry"
                        self._record_block(
                            block_entries,
                            trigger,
                            "emergency_exit_veto_same_bar",
                            detail,
                            bar,
                            extra={"cooldown_recommendation_bars": self._emergency_cooldown_bars()},
                        )
                        continue
                # Check if hold_rule suppresses this exit (unless emergency category)
                if trigger.hold_rule and trigger.category != "emergency_exit":
                    try:
                        hold_active = self.evaluator.evaluate(trigger.hold_rule, context)
                        if hold_active:
                            # Hold rule active - suppress exit to maintain position
                            count = self._hold_suppression_counts.get(trigger.id, 0) + 1
                            self._hold_suppression_counts[trigger.id] = count
                            detail = f"Hold rule active ('{trigger.hold_rule}'), suppressing exit (consecutive: {count})"
                            self._record_block(block_entries, trigger, "HOLD_RULE", detail, bar)
                            if count >= 12 and count % 12 == 0:
                                warn = (
                                    f"Trigger '{trigger.id}' hold rule has suppressed {count} consecutive exits — "
                                    f"rule may be degenerate: '{trigger.hold_rule}'"
                                )
                                self.hold_suppression_warnings.append(warn)
                                logger.warning(warn)
                            continue
                        else:
                            # Hold rule inactive — reset consecutive counter
                            self._hold_suppression_counts[trigger.id] = 0
                    except MissingIndicatorError:
                        # If hold rule can't evaluate due to missing indicator, allow exit
                        self._hold_suppression_counts[trigger.id] = 0
                    except RuleSyntaxError:
                        # If hold rule has syntax error, allow exit
                        self._hold_suppression_counts[trigger.id] = 0

                # Check minimum hold period before allowing exit
                if self._is_within_hold_period(trigger.symbol):
                    detail = f"Position held for less than {self.min_hold_bars} bars"
                    reason = "MIN_HOLD_PERIOD"
                    extra = None
                    if trigger.category == "emergency_exit":
                        reason = "emergency_exit_veto_min_hold"
                        extra = {"cooldown_recommendation_bars": self._emergency_cooldown_bars()}
                    self._record_block(block_entries, trigger, reason, detail, bar, extra=extra)
                    continue
                entry_category = self._entry_category(position_meta, trigger.symbol)
                if not self._exit_binding_allows(trigger, entry_category):
                    entry_trigger_id = self._entry_trigger_id(position_meta, trigger.symbol)
                    detail = f"Exit category {trigger.category} does not match entry category {entry_category}"
                    self._record_block(
                        block_entries,
                        trigger,
                        "exit_binding_mismatch",
                        detail,
                        bar,
                        extra={
                            "entry_category": entry_category,
                            "entry_trigger_id": entry_trigger_id,
                            "exit_category": trigger.category,
                            "exit_trigger_id": trigger.id,
                        },
                    )
                    continue
                # Use exit_fraction for partial exits (risk_reduce category)
                exit_fraction = trigger.exit_fraction if trigger.exit_fraction is not None else 1.0
                # Emergency exits should only flatten live exposure, not entries staged earlier in this bar.
                exit_portfolio = portfolio_base if trigger.category == "emergency_exit" else portfolio_for_eval
                exit_order = self._flatten_order(
                    trigger, bar, exit_portfolio, f"{trigger.id}_exit", block_entries,
                    fraction=exit_fraction,
                )
                if exit_order:
                    orders.append(exit_order)
                    orders = self._deduplicate_orders(orders, bar.symbol, block_entries)
                    triggers_fired_for_symbol = len(orders)
                    portfolio_for_eval = self._rebuild_portfolio_for_orders(portfolio_base, orders)
                    continue

            if trigger.category == "emergency_exit":
                # Emergency exits are exit-only; never evaluate entry rules.
                continue

            # Entry rule evaluation
            try:
                entry_fired = bool(trigger.entry_rule and self.evaluator.evaluate(trigger.entry_rule, context))
                # Debug sampling for entry rule
                self._maybe_sample_evaluation(
                    bar, trigger, "entry", trigger.entry_rule, context, current_position
                )
            except MissingIndicatorError as exc:
                detail = f"{exc}; entry_rule='{trigger.entry_rule}'"
                self._record_block(block_entries, trigger, BlockReason.MISSING_INDICATOR.value, detail, bar)
                continue
            except RuleSyntaxError as exc:
                unknown = self._unknown_identifiers(trigger.entry_rule, context)
                detail = f"{exc}; entry_rule='{trigger.entry_rule}'"
                extra = None
                if unknown:
                    detail = f"{detail}; unknown_identifiers={unknown}"
                    extra = {"unknown_identifiers": unknown}
                    self._log_unknown_identifiers(trigger, "entry", unknown, trigger.entry_rule)
                self._record_block(block_entries, trigger, BlockReason.EXPRESSION_ERROR.value, detail, bar, extra=extra)
                continue

            if entry_fired:
                desired = trigger.direction
                if desired in {"long", "short"} and current_position in {"long", "short"} and desired != current_position:
                    detail = f"Conflicting signal: {desired} while {current_position}"
                    self._record_block(
                        block_entries,
                        trigger,
                        "conflicting_signal_detected",
                        detail,
                        bar,
                        extra={
                            "current_position": current_position,
                            "desired_position": desired,
                            "confidence": trigger.confidence_grade,
                            "rationale": trigger.entry_rule,
                            "entry_rule": trigger.entry_rule,
                            "policy": self.conflicting_signal_policy,
                        },
                    )
                    if self.conflicting_signal_policy in {"ignore", "defer"}:
                        continue
                    if self.conflicting_signal_policy == "exit":
                        if self._is_within_hold_period(trigger.symbol):
                            self._record_block(
                                block_entries,
                                trigger,
                                "conflict_exit_min_hold",
                                f"Conflict exit blocked by min_hold ({self.min_hold_bars} bars)",
                                bar,
                            )
                            continue
                        exit_order = self._flatten_order(
                            trigger,
                            bar,
                            portfolio_for_eval,
                            f"{trigger.id}_exit",
                            block_entries,
                            intent="conflict_exit",
                        )
                        if exit_order:
                            orders.append(exit_order)
                            orders = self._deduplicate_orders(orders, bar.symbol, block_entries)
                            triggers_fired_for_symbol = len(orders)
                            portfolio_for_eval = self._rebuild_portfolio_for_orders(portfolio_base, orders)
                        continue
                    entry = self._entry_order(
                        trigger,
                        indicator,
                        portfolio_for_eval,
                        bar,
                        block_entries,
                        intent="conflict_reverse",
                    )
                else:
                    entry = self._entry_order(trigger, indicator, portfolio_for_eval, bar, block_entries)
                if entry:
                    orders.append(entry)
                    orders = self._deduplicate_orders(orders, bar.symbol, block_entries)
                    triggers_fired_for_symbol = len(orders)
                    portfolio_for_eval = self._rebuild_portfolio_for_orders(portfolio_base, orders)

        # Deduplicate orders per symbol with EXIT PRIORITY:
        # If both exit and entry orders exist for same symbol, only keep the exit.
        # This prevents whipsawing (entering then immediately exiting).
        orders = self._deduplicate_orders(orders, bar.symbol, block_entries)

        # Track per-trigger fire counts from final orders (Runbook 25)
        for order in orders:
            # Extract base trigger_id from order reason (strip _exit/_flat suffix)
            tid = order.reason
            for suffix in ("_exit", "_flat"):
                if tid.endswith(suffix):
                    tid = tid[: -len(suffix)]
                    break
            self._trigger_fire_counts[tid] = self._trigger_fire_counts.get(tid, 0) + 1

        return orders, block_entries

    def _is_in_cooldown(self, symbol: str) -> bool:
        """Check if symbol is in trade cooldown period."""
        if self.trade_cooldown_bars <= 0:
            return False
        last_trade = self._last_trade_bar.get(symbol)
        if last_trade is None:
            return False
        bars_since_trade = self._bar_counter - last_trade
        return bars_since_trade < self.trade_cooldown_bars

    def _is_within_hold_period(self, symbol: str) -> bool:
        """Check if position is within minimum hold period (can't exit yet)."""
        if self.min_hold_bars <= 0:
            return False
        entry_bar = self._position_entry_bar.get(symbol)
        if entry_bar is None:
            return False
        bars_held = self._bar_counter - entry_bar
        return bars_held < self.min_hold_bars

    def record_fill(self, symbol: str, is_entry: bool, timestamp: datetime) -> None:
        """Record a fill for cooldown and hold period tracking.

        Call this after an order is filled to update tracking state.

        Args:
            symbol: The symbol that was traded
            is_entry: True if this was an entry (new position), False if exit
            timestamp: The execution timestamp for the fill
        """
        self._last_trade_bar[symbol] = self._bar_counter
        if is_entry:
            self._position_entry_bar[symbol] = self._bar_counter
            self._last_entry_timestamp[symbol] = timestamp
        else:
            # Clear entry bar when position is closed
            self._position_entry_bar.pop(symbol, None)

    def _priority_skip_bypass(self, confidence: Literal["A", "B", "C"] | None) -> bool:
        if self.priority_skip_confidence_threshold is None:
            return False
        if confidence is None:
            return False
        threshold_priority = self.CONFIDENCE_PRIORITY.get(self.priority_skip_confidence_threshold, 0)
        confidence_priority = self.CONFIDENCE_PRIORITY.get(confidence, 0)
        return confidence_priority >= threshold_priority

    def _apply_order_to_portfolio(self, portfolio: PortfolioState, order: Order) -> PortfolioState:
        if order.quantity <= 0:
            return portfolio
        positions = dict(portfolio.positions)
        delta_qty = order.quantity if order.side == "buy" else -order.quantity
        positions[order.symbol] = positions.get(order.symbol, 0.0) + delta_qty
        if abs(positions.get(order.symbol, 0.0)) <= 1e-9:
            positions.pop(order.symbol, None)
        if order.side == "buy":
            cash = portfolio.cash - (order.price * order.quantity)
        else:
            cash = portfolio.cash + (order.price * order.quantity)
        return portfolio.model_copy(update={"positions": positions, "cash": cash})

    def _rebuild_portfolio_for_orders(self, portfolio: PortfolioState, orders: List[Order]) -> PortfolioState:
        updated = portfolio
        for order in orders:
            updated = self._apply_order_to_portfolio(updated, order)
        return updated

    def _maybe_sample_evaluation(
        self,
        bar: Bar,
        trigger: TriggerCondition,
        rule_type: Literal["entry", "exit"],
        rule: str | None,
        context: dict[str, Any],
        position: Literal["long", "short", "flat"],
    ) -> None:
        """Conditionally sample a trigger evaluation for debugging.

        Samples are collected probabilistically based on debug_sample_rate.
        Once debug_max_samples are collected, no more samples are added.
        """
        if not rule:
            return
        if self.debug_sample_rate <= 0.0:
            return
        if len(self.evaluation_samples) >= self.debug_max_samples:
            return
        if random.random() > self.debug_sample_rate:
            return

        # Get detailed trace from evaluator
        trace = self.evaluator.evaluate_with_trace(rule, context)

        sample = TriggerEvaluationSample(
            timestamp=bar.timestamp,
            trigger_id=trigger.id,
            symbol=trigger.symbol,
            rule_type=rule_type,
            trace=trace,
            bar_close=bar.close,
            position=position,
        )
        self.evaluation_samples.append(sample)

    def get_evaluation_samples_summary(self) -> List[dict[str, Any]]:
        """Get a summary of evaluation samples for inclusion in backtest results.

        Returns a list of dicts suitable for JSON serialization.
        """
        summaries = []
        for sample in self.evaluation_samples:
            summaries.append({
                "timestamp": sample.timestamp.isoformat(),
                "trigger_id": sample.trigger_id,
                "symbol": sample.symbol,
                "rule_type": sample.rule_type,
                "expression": sample.trace.expression,
                "result": sample.trace.result,
                "bar_close": sample.bar_close,
                "position": sample.position,
                "context_values": {
                    k: v for k, v in sample.trace.context_values.items()
                    if v != "<missing>" and v is not None
                },
                "sub_results": sample.trace.sub_results,
                "error": sample.trace.error,
            })
        return summaries

    def _deduplicate_orders(
        self,
        orders: List[Order],
        bar_symbol: str,
        block_entries: List[dict] | None = None,
    ) -> List[Order]:
        """Deduplicate orders per symbol using tiered exit priority.

        Precedence Tiers (strict order, no exceptions):
        - Tier 0: emergency_exit — always wins, preempts all entries and exits
        - Tier 1: risk_off (when latched) — preempts entries, beats normal exits
        - Tier 2: risk_reduce — competes via confidence-based override
        - Tier 3: normal exits — standard strategy exits

        Rules:
        1. Emergency exits are safety interrupts — they always win dedup,
           regardless of regime or competing entry confidence.
        2. risk_off exits with priority (latch active) preempt entries and beat normal exits.
        3. risk_reduce and normal exits compete; high-confidence entries can override them.
        4. Only one order per symbol per bar is allowed.
        5. If multiple competing orders exist, use highest confidence then first-in-list.

        The confidence_override_threshold determines when entries can override exits:
        - "A": Only A-grade entries override exits
        - "B": A or B grade entries override exits
        - "C": All graded entries override exits
        - None: Exit always has priority (no confidence override)
        """
        if len(orders) <= 1:
            return orders

        # Group by symbol
        by_symbol: dict[str, List[Order]] = {}
        for order in orders:
            by_symbol.setdefault(order.symbol, []).append(order)

        result: List[Order] = []
        for symbol, symbol_orders in by_symbol.items():
            if len(symbol_orders) == 1:
                result.append(symbol_orders[0])
                continue

            # Separate exits from entries
            exits = [o for o in symbol_orders if o.reason.endswith("_exit") or o.reason.endswith("_flat")]
            entries = [o for o in symbol_orders if o not in exits]
            emergency_exits = [o for o in exits if o.emergency]
            risk_off_exits = [o for o in exits if o.trigger_category == "risk_off" and not o.emergency]

            # Tier 0: Emergency exits always win
            if emergency_exits:
                result.append(emergency_exits[0])
                # Record preemption of competing entries for auditability
                if entries and block_entries is not None:
                    for entry_order in entries:
                        conf = self._trigger_confidence.get(entry_order.reason)
                        block_entries.append({
                            "trigger_id": entry_order.reason,
                            "symbol": entry_order.symbol,
                            "timeframe": entry_order.timeframe,
                            "timestamp": entry_order.timestamp.isoformat(),
                            "reason": "emergency_exit_preempts_entry",
                            "detail": (
                                f"Emergency exit preempted entry "
                                f"(trigger={entry_order.reason}, confidence={conf})"
                            ),
                            "price": entry_order.price,
                            "preempted_by": emergency_exits[0].reason,
                        })
                continue

            # Tier 1: risk_off with priority (latch active) preempts entries
            if risk_off_exits and self._risk_off_has_priority():
                result.append(risk_off_exits[0])
                # Record preemption of entries for auditability
                if entries and block_entries is not None:
                    for entry_order in entries:
                        conf = self._trigger_confidence.get(entry_order.reason)
                        block_entries.append({
                            "trigger_id": entry_order.reason,
                            "symbol": entry_order.symbol,
                            "timeframe": entry_order.timeframe,
                            "timestamp": entry_order.timestamp.isoformat(),
                            "reason": "risk_off_preempts_entry",
                            "detail": (
                                f"risk_off exit preempted entry (latch active) "
                                f"(trigger={entry_order.reason}, confidence={conf})"
                            ),
                            "price": entry_order.price,
                            "preempted_by": risk_off_exits[0].reason,
                        })
                continue

            # Get confidence levels for entries
            def get_confidence(order: Order) -> Literal["A", "B", "C"] | None:
                # Extract base trigger_id (remove _exit/_flat suffix)
                trigger_id = order.reason
                for suffix in ("_exit", "_flat"):
                    if trigger_id.endswith(suffix):
                        trigger_id = trigger_id[:-len(suffix)]
                        break
                return self._trigger_confidence.get(trigger_id)

            def meets_threshold(confidence: Literal["A", "B", "C"] | None) -> bool:
                """Check if confidence meets the override threshold."""
                if self.confidence_override_threshold is None:
                    return False  # No override allowed
                if confidence is None:
                    return False  # No confidence grade
                threshold_priority = self.CONFIDENCE_PRIORITY.get(self.confidence_override_threshold, 0)
                confidence_priority = self.CONFIDENCE_PRIORITY.get(confidence, 0)
                return confidence_priority >= threshold_priority

            # Find highest-confidence entry that meets threshold
            high_conf_entries = [
                (o, self.CONFIDENCE_PRIORITY.get(get_confidence(o), 0))
                for o in entries
                if meets_threshold(get_confidence(o))
            ]

            # Tier 2/3: Confidence-based competition
            # risk_off (non-latched) and risk_reduce compete like normal exits
            if high_conf_entries:
                # Sort by confidence priority (descending), take highest
                high_conf_entries.sort(key=lambda x: x[1], reverse=True)
                best_entry = high_conf_entries[0][0]
                # High-confidence entry overrides exits (risk_reduce, risk_off non-latched, normal)
                result.append(best_entry)
                # Record override for auditability
                if exits and block_entries is not None:
                    for exit_order in exits:
                        exit_category = exit_order.trigger_category or "normal"
                        block_entries.append({
                            "trigger_id": exit_order.reason,
                            "symbol": exit_order.symbol,
                            "timeframe": exit_order.timeframe,
                            "timestamp": exit_order.timestamp.isoformat(),
                            "reason": f"entry_overrides_{exit_category}",
                            "detail": (
                                f"High-confidence entry overrode {exit_category} exit "
                                f"(entry={best_entry.reason}, confidence={get_confidence(best_entry)})"
                            ),
                            "price": exit_order.price,
                            "overridden_by": best_entry.reason,
                        })
            elif exits:
                # Default: exit priority (risk_off beats risk_reduce beats normal)
                # Sort exits by category priority: risk_off > risk_reduce > normal
                def exit_priority(o: Order) -> int:
                    if o.trigger_category == "risk_off":
                        return 2
                    if o.trigger_category == "risk_reduce":
                        return 1
                    return 0
                exits.sort(key=exit_priority, reverse=True)
                result.append(exits[0])
            elif entries:
                # No exits, no high-conf entries: use first entry
                result.append(entries[0])

        return result
