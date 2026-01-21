"""Trigger evaluation engine that turns LLM plans into deterministic orders."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, List, Literal

from schemas.llm_strategist import AssetState, IndicatorSnapshot, PortfolioState, StrategyPlan, TriggerCondition
from trading_core.execution_engine import BlockReason

from .risk_engine import RiskEngine
from .rule_dsl import EvaluationTrace, MissingIndicatorError, RuleEvaluator, RuleSyntaxError


from .trade_risk import TradeRiskEvaluator


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
        confidence_override_threshold: Literal["A", "B", "C"] | None = "A",
        min_hold_bars: int = 4,
        trade_cooldown_bars: int = 2,
        debug_sample_rate: float = 0.0,
        debug_max_samples: int = 100,
        prioritize_by_confidence: bool = True,
        max_triggers_per_symbol_per_bar: int = 1,
    ) -> None:
        """Initialize the trigger engine.

        Args:
            plan: Strategy plan with triggers
            risk_engine: Risk engine for position sizing
            evaluator: Rule evaluator (optional)
            trade_risk: Trade risk evaluator (optional)
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
        """
        self.plan = plan
        self.risk_engine = risk_engine
        self.evaluator = evaluator or RuleEvaluator()
        self.trade_risk = trade_risk or TradeRiskEvaluator(risk_engine)
        self.confidence_override_threshold = confidence_override_threshold
        self.min_hold_bars = max(0, min_hold_bars)
        self.trade_cooldown_bars = max(0, trade_cooldown_bars)
        self.prioritize_by_confidence = prioritize_by_confidence
        self.max_triggers_per_symbol_per_bar = max(1, max_triggers_per_symbol_per_bar)
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

    def _sort_triggers_by_confidence(self, triggers: Iterable[TriggerCondition]) -> List[TriggerCondition]:
        """Sort triggers by confidence grade (A first, then B, then C, then None)."""
        return sorted(
            triggers,
            key=lambda t: self.CONFIDENCE_PRIORITY.get(t.confidence_grade, 0),
            reverse=True,  # Highest priority first
        )

    def _context(
        self,
        indicator: IndicatorSnapshot,
        asset_state: AssetState | None,
        market_structure: dict[str, float | str | None] | None = None,
        portfolio: PortfolioState | None = None,
    ) -> dict[str, float | str | None]:
        """Build evaluation context, including cross-timeframe aliases."""

        def _alias_key(key: str) -> str | None:
            parts = key.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                return parts[0]
            return None

        context = indicator.model_dump()
        for key, value in list(context.items()):
            alias = _alias_key(key)
            if alias and alias not in context:
                context[alias] = value
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
            context["position"] = self._position_direction(indicator.symbol, portfolio)
        return context

    def _position_direction(self, symbol: str, portfolio: PortfolioState) -> Literal["long", "short", "flat"]:
        qty = portfolio.positions.get(symbol, 0.0)
        if qty > 0:
            return "long"
        if qty < 0:
            return "short"
        return "flat"

    def _flatten_order(
        self,
        trigger: TriggerCondition,
        bar: Bar,
        portfolio: PortfolioState,
        reason: str,
        block_entries: List[dict] | None = None,
    ) -> Order | None:
        qty = portfolio.positions.get(trigger.symbol, 0.0)
        if abs(qty) <= 1e-9:
            return None
        decision = self.trade_risk.evaluate(trigger, "flatten", bar.close, portfolio, None)
        if not decision.allowed and block_entries is not None and decision.reason:
            self._record_block(block_entries, trigger, decision.reason, "Flatten blocked unexpectedly", bar)
            return None
        side: Literal["buy", "sell"] = "sell" if qty > 0 else "buy"
        return Order(symbol=trigger.symbol, side=side, quantity=abs(qty), price=bar.close, timeframe=bar.timeframe, reason=reason, timestamp=bar.timestamp)

    def _record_block(
        self,
        container: List[dict],
        trigger: TriggerCondition,
        reason: str,
        detail: str,
        bar: Bar,
    ) -> None:
        container.append(
            {
                "trigger_id": trigger.id,
                "symbol": trigger.symbol,
                "timeframe": bar.timeframe,
                "timestamp": bar.timestamp.isoformat(),
                "reason": reason,
                "detail": detail,
                "price": bar.close,
            }
        )

    def _entry_order(
        self,
        trigger: TriggerCondition,
        indicator: IndicatorSnapshot,
        portfolio: PortfolioState,
        bar: Bar,
        block_entries: List[dict] | None = None,
    ) -> Order | None:
        desired = trigger.direction
        current = self._position_direction(trigger.symbol, portfolio)
        if desired in {"flat", "exit", "flat_exit"}:
            if current == "flat":
                return None
            return self._flatten_order(trigger, bar, portfolio, f"{trigger.id}_flat", block_entries)
        if desired == current:
            return None
        stop_distance = None
        if trigger.stop_loss_pct is not None:
            stop_distance = abs(bar.close * trigger.stop_loss_pct / 100.0)
        check = self.trade_risk.evaluate(trigger, "entry", bar.close, portfolio, indicator, stop_distance=stop_distance)
        if not check.allowed:
            if block_entries is not None and check.reason:
                detail = f"Risk constraint {check.reason} prevented sizing"
                self._record_block(block_entries, trigger, check.reason, detail, bar)
            return None
        side: Literal["buy", "sell"] = "buy" if desired == "long" else "sell"
        return Order(
            symbol=trigger.symbol,
            side=side,
            quantity=check.quantity,
            price=bar.close,
            timeframe=bar.timeframe,
            reason=trigger.id,
            timestamp=bar.timestamp,
            stop_distance=stop_distance,
        )

    def on_bar(
        self,
        bar: Bar,
        indicator: IndicatorSnapshot,
        portfolio: PortfolioState,
        asset_state: AssetState | None = None,
        market_structure: dict[str, float | str | None] | None = None,
    ) -> tuple[List[Order], List[dict]]:
        orders: List[Order] = []
        block_entries: List[dict] = []

        # Increment bar counter for cooldown/hold tracking
        self._bar_counter += 1

        # Check trade cooldown for this symbol
        if self._is_in_cooldown(bar.symbol):
            # Skip all triggers for this symbol during cooldown
            return orders, block_entries

        context = self._context(indicator, asset_state, market_structure, portfolio)
        current_position = self._position_direction(bar.symbol, portfolio)

        # Track how many triggers have fired for this symbol this bar
        # (used for max_triggers_per_symbol_per_bar enforcement)
        triggers_fired_for_symbol = 0

        # Use sorted triggers if prioritize_by_confidence is enabled
        trigger_list = self._sorted_triggers if self.prioritize_by_confidence else self.plan.triggers

        for trigger in trigger_list:
            if trigger.symbol != bar.symbol or trigger.timeframe != bar.timeframe:
                continue

            # Early exit if we've hit the max triggers for this symbol
            if self.prioritize_by_confidence and triggers_fired_for_symbol >= self.max_triggers_per_symbol_per_bar:
                # Log that we're skipping lower-confidence triggers
                detail = f"Max triggers ({self.max_triggers_per_symbol_per_bar}) already fired for {bar.symbol}"
                self._record_block(block_entries, trigger, "SIGNAL_PRIORITY", detail, bar)
                continue

            # Exit rule evaluation
            try:
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
                detail = f"{exc}; exit_rule='{trigger.exit_rule}'"
                self._record_block(block_entries, trigger, BlockReason.EXPRESSION_ERROR.value, detail, bar)
                continue

            if exit_fired:
                if trigger.category == "emergency_exit":
                    last_entry_ts = self._last_entry_timestamp.get(trigger.symbol)
                    if last_entry_ts and last_entry_ts == bar.timestamp:
                        detail = "Emergency exit blocked on same bar as entry"
                        self._record_block(block_entries, trigger, "SAME_BAR_ENTRY", detail, bar)
                        continue
                # Check if hold_rule suppresses this exit (unless emergency category)
                if trigger.hold_rule and trigger.category != "emergency_exit":
                    try:
                        hold_active = self.evaluator.evaluate(trigger.hold_rule, context)
                        if hold_active:
                            # Hold rule active - suppress exit to maintain position
                            detail = f"Hold rule active ('{trigger.hold_rule}'), suppressing exit"
                            self._record_block(block_entries, trigger, "HOLD_RULE", detail, bar)
                            continue
                    except MissingIndicatorError:
                        # If hold rule can't evaluate due to missing indicator, allow exit
                        pass
                    except RuleSyntaxError:
                        # If hold rule has syntax error, allow exit
                        pass

                # Check minimum hold period before allowing exit
                if self._is_within_hold_period(trigger.symbol):
                    detail = f"Position held for less than {self.min_hold_bars} bars"
                    self._record_block(block_entries, trigger, "MIN_HOLD_PERIOD", detail, bar)
                    continue
                exit_order = self._flatten_order(trigger, bar, portfolio, f"{trigger.id}_exit", block_entries)
                if exit_order:
                    orders.append(exit_order)
                    triggers_fired_for_symbol += 1
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
                detail = f"{exc}; entry_rule='{trigger.entry_rule}'"
                self._record_block(block_entries, trigger, BlockReason.EXPRESSION_ERROR.value, detail, bar)
                continue

            if entry_fired:
                entry = self._entry_order(trigger, indicator, portfolio, bar, block_entries)
                if entry:
                    orders.append(entry)
                    triggers_fired_for_symbol += 1

        # Deduplicate orders per symbol with EXIT PRIORITY:
        # If both exit and entry orders exist for same symbol, only keep the exit.
        # This prevents whipsawing (entering then immediately exiting).
        orders = self._deduplicate_orders(orders, bar.symbol)

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

    def _deduplicate_orders(self, orders: List[Order], bar_symbol: str) -> List[Order]:
        """Deduplicate orders per symbol using confidence-aware exit priority.

        Rules:
        1. Default: Exit orders (reason ending in _exit or _flat) take priority
        2. Exception: High-confidence entries can override exits if confidence >= threshold
        3. Only one order per symbol per bar is allowed
        4. If multiple competing orders exist, use highest confidence then first-in-list

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

            if high_conf_entries:
                # Sort by confidence priority (descending), take highest
                high_conf_entries.sort(key=lambda x: x[1], reverse=True)
                best_entry = high_conf_entries[0][0]
                # High-confidence entry overrides exits
                result.append(best_entry)
            elif exits:
                # Default: exit priority
                result.append(exits[0])
            elif entries:
                # No exits, no high-conf entries: use first entry
                result.append(entries[0])

        return result
