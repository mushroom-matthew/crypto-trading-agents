"""Hypothesis executor.

Runbook 78: TradeHypothesis Schema + HypothesisExecutor.

Manages the lifecycle of open TradeHypothesis objects for one or more symbols.
Replaces the exit trigger evaluation path for hypothesis-mode plans.

Lifecycle per hypothesis:
  PENDING  — entry_rule not yet fired; waiting for entry condition
  OPEN     — position is open; executor watches stop/target; evaluates invalidation
  COMPLETED — position closed (win/loss/invalidation/expiry)

Thread-safety: designed for single-threaded use inside PaperTradingWorkflow
(Temporal workflow environment). State is fully serializable via to_dict()/from_dict().
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, TYPE_CHECKING

from schemas.hypothesis import TradeHypothesis

if TYPE_CHECKING:
    from schemas.llm_strategist import IndicatorSnapshot

logger = logging.getLogger(__name__)


class HypothesisStatus(str, Enum):
    PENDING = "pending"
    OPEN = "open"
    COMPLETED = "completed"


@dataclass
class OpenHypothesisState:
    """Runtime state for a hypothesis that has entered (position is open)."""

    hypothesis_id: str
    symbol: str
    direction: Literal["long", "short"]
    entry_price: float
    entry_bar_index: int
    stop_price: float
    target_price: float
    trailing_stop_price: Optional[float]
    bars_held: int = 0
    invalidated: bool = False
    invalidation_reason: Optional[str] = None
    hypothesis_snapshot: Dict[str, Any] = field(default_factory=dict)

    # Trailing stop activation threshold (in R)
    trailing_activation_r: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "symbol": self.symbol,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "entry_bar_index": self.entry_bar_index,
            "stop_price": self.stop_price,
            "target_price": self.target_price,
            "trailing_stop_price": self.trailing_stop_price,
            "bars_held": self.bars_held,
            "invalidated": self.invalidated,
            "invalidation_reason": self.invalidation_reason,
            "hypothesis_snapshot": self.hypothesis_snapshot,
            "trailing_activation_r": self.trailing_activation_r,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OpenHypothesisState":
        return cls(
            hypothesis_id=d["hypothesis_id"],
            symbol=d["symbol"],
            direction=d["direction"],
            entry_price=d["entry_price"],
            entry_bar_index=d["entry_bar_index"],
            stop_price=d["stop_price"],
            target_price=d["target_price"],
            trailing_stop_price=d.get("trailing_stop_price"),
            bars_held=d.get("bars_held", 0),
            invalidated=d.get("invalidated", False),
            invalidation_reason=d.get("invalidation_reason"),
            hypothesis_snapshot=d.get("hypothesis_snapshot", {}),
            trailing_activation_r=d.get("trailing_activation_r"),
        )


@dataclass
class HypothesisExitSignal:
    """Returned by on_tick / on_bar when a hypothesis should be closed."""

    hypothesis_id: str
    symbol: str
    direction: Literal["long", "short"]
    exit_reason: Literal["stop_hit", "target_hit", "invalidated", "expired"]
    exit_price: float
    stop_price: float
    target_price: float
    entry_price: float
    bars_held: int
    hypothesis_snapshot: Dict[str, Any]


class HypothesisExecutor:
    """Manages open hypotheses for one or more symbols in a session.

    One hypothesis can be OPEN at a time per symbol. A second hypothesis
    for the same symbol will remain PENDING until the first is COMPLETED.

    Usage (from PaperTradingWorkflow._evaluate_and_execute):

      # On each tick (fast path — stop/target sweep):
      exit_signals = executor.on_tick(symbol, current_price)

      # On each bar (entry + invalidation evaluation):
      entry_signals, block_records, expiry_signals = executor.on_bar(
          bar, indicator, evaluator, portfolio
      )

      # After fill:
      executor.on_fill(symbol, fill_price, hypothesis_id)
    """

    def __init__(self) -> None:
        # symbol → list of pending hypotheses (ordered by confidence/priority)
        self._pending: Dict[str, List[TradeHypothesis]] = {}
        # symbol → open hypothesis state (max one per symbol)
        self._open: Dict[str, OpenHypothesisState] = {}
        # symbol → list of completed hypothesis dicts (for attribution)
        self._completed: Dict[str, List[Dict[str, Any]]] = {}
        # Global bar counter (incremented on each on_bar call)
        self._bar_count: int = 0
        # R80: WorldState risk multiplier — updated by paper_trading before each on_bar call
        self._world_state_risk_multiplier: float = 1.0

    @property
    def world_state_risk_multiplier(self) -> float:
        """Current risk multiplier from WorldState judge guidance.

        paper_trading.py reads this before computing position quantity.
        Set to <1.0 when judge guidance requests reduced sizing.
        Always returns a positive value; defaults to 1.0 (no adjustment).
        """
        return self._world_state_risk_multiplier

    @world_state_risk_multiplier.setter
    def world_state_risk_multiplier(self, value: float) -> None:
        """Set the risk multiplier. Clamps to [0.1, 2.0] to prevent extremes."""
        self._world_state_risk_multiplier = max(0.1, min(2.0, float(value)))

    # ── Plan loading ──────────────────────────────────────────────────────

    def load_hypotheses(self, hypotheses: List[TradeHypothesis]) -> None:
        """Load a set of compiled hypotheses as PENDING.

        Clears any existing PENDING hypotheses for symbols that appear
        in the new list. OPEN hypotheses are preserved (ongoing positions).
        """
        # Group by symbol
        by_symbol: Dict[str, List[TradeHypothesis]] = {}
        for h in hypotheses:
            by_symbol.setdefault(h.symbol, []).append(h)

        for symbol, hyps in by_symbol.items():
            if symbol in self._open:
                # Don't replace pending queue if a position is open
                logger.debug(
                    "hypothesis_executor: %s has open position; queuing %d new hypotheses",
                    symbol, len(hyps),
                )
                # Append to pending after existing ones
                existing = self._pending.get(symbol, [])
                self._pending[symbol] = existing + hyps
            else:
                # Sort by confidence_grade: A > B > C
                grade_order = {"A": 0, "B": 1, "C": 2}
                self._pending[symbol] = sorted(
                    hyps, key=lambda h: grade_order.get(h.confidence_grade, 1)
                )
                logger.debug(
                    "hypothesis_executor: loaded %d hypotheses for %s",
                    len(hyps), symbol,
                )

    # ── Tick processing (fast path) ───────────────────────────────────────

    def on_tick(
        self, symbol: str, current_price: float
    ) -> List[HypothesisExitSignal]:
        """Fast path: check if open hypothesis stop/target has been crossed.

        Called on every price tick, not just bar close.
        Returns exit signals for any hypothesis where stop or target is breached.
        """
        open_state = self._open.get(symbol)
        if open_state is None:
            return []

        signals: List[HypothesisExitSignal] = []

        if open_state.direction == "long":
            # Check stop (price fell to or below stop)
            if current_price <= open_state.stop_price:
                signals.append(HypothesisExitSignal(
                    hypothesis_id=open_state.hypothesis_id,
                    symbol=symbol,
                    direction="long",
                    exit_reason="stop_hit",
                    exit_price=current_price,
                    stop_price=open_state.stop_price,
                    target_price=open_state.target_price,
                    entry_price=open_state.entry_price,
                    bars_held=open_state.bars_held,
                    hypothesis_snapshot=open_state.hypothesis_snapshot,
                ))
            # Check target (price rose to or above target)
            elif current_price >= open_state.target_price:
                signals.append(HypothesisExitSignal(
                    hypothesis_id=open_state.hypothesis_id,
                    symbol=symbol,
                    direction="long",
                    exit_reason="target_hit",
                    exit_price=current_price,
                    stop_price=open_state.stop_price,
                    target_price=open_state.target_price,
                    entry_price=open_state.entry_price,
                    bars_held=open_state.bars_held,
                    hypothesis_snapshot=open_state.hypothesis_snapshot,
                ))
        else:  # short
            # Check stop (price rose to or above stop)
            if current_price >= open_state.stop_price:
                signals.append(HypothesisExitSignal(
                    hypothesis_id=open_state.hypothesis_id,
                    symbol=symbol,
                    direction="short",
                    exit_reason="stop_hit",
                    exit_price=current_price,
                    stop_price=open_state.stop_price,
                    target_price=open_state.target_price,
                    entry_price=open_state.entry_price,
                    bars_held=open_state.bars_held,
                    hypothesis_snapshot=open_state.hypothesis_snapshot,
                ))
            # Check target (price fell to or below target)
            elif current_price <= open_state.target_price:
                signals.append(HypothesisExitSignal(
                    hypothesis_id=open_state.hypothesis_id,
                    symbol=symbol,
                    direction="short",
                    exit_reason="target_hit",
                    exit_price=current_price,
                    stop_price=open_state.stop_price,
                    target_price=open_state.target_price,
                    entry_price=open_state.entry_price,
                    bars_held=open_state.bars_held,
                    hypothesis_snapshot=open_state.hypothesis_snapshot,
                ))

        # Update trailing stop if applicable
        if not signals and open_state.trailing_stop_price is not None:
            self._update_trailing_stop(open_state, current_price)

        return signals

    def _update_trailing_stop(
        self, state: OpenHypothesisState, current_price: float
    ) -> None:
        """Ratchet the trailing stop upward (long) or downward (short)."""
        if state.direction == "long":
            # Move stop up to maintain fixed distance from current price
            new_trail = current_price - (state.entry_price - state.stop_price)
            if new_trail > state.trailing_stop_price:  # type: ignore[operator]
                state.trailing_stop_price = new_trail
                state.stop_price = max(state.stop_price, new_trail)
        else:  # short
            new_trail = current_price + (state.stop_price - state.entry_price)
            if new_trail < state.trailing_stop_price:  # type: ignore[operator]
                state.trailing_stop_price = new_trail
                state.stop_price = min(state.stop_price, new_trail)

    # ── Bar processing ────────────────────────────────────────────────────

    def on_bar(
        self,
        bar: Any,  # Bar object from trigger_engine
        indicator: "IndicatorSnapshot",
        evaluator: Any,  # RuleEvaluator from trigger_engine
        portfolio: Any,  # portfolio state dict
    ) -> tuple[
        List[Dict[str, Any]],   # entry signals (hypothesis_id, symbol, direction)
        List[Dict[str, Any]],   # block records (same format as TriggerEngine._record_block)
        List[HypothesisExitSignal],  # invalidation/expiry signals
    ]:
        """Evaluate entries for PENDING hypotheses and invalidation for OPEN ones.

        Per-bar processing:
        1. Increment bar counter.
        2. For each OPEN hypothesis:
           a. Increment bars_held.
           b. Evaluate invalidation_rule if set.
           c. Check invalidation_horizon_bars expiry.
        3. For each PENDING hypothesis (first in queue per symbol):
           a. Check invalidation_horizon_bars for pending expiry.
           b. Evaluate entry_rule (same evaluator as TriggerEngine).
           c. If entry_rule fires → emit entry signal.
        """
        self._bar_count += 1

        entry_signals: List[Dict[str, Any]] = []
        block_records: List[Dict[str, Any]] = []
        exit_signals: List[HypothesisExitSignal] = []

        current_price = bar.close if hasattr(bar, "close") else indicator.close

        # Build evaluation context from indicator snapshot
        context = self._build_eval_context(indicator, current_price)

        # ── Process OPEN hypotheses ───────────────────────────────────────
        for symbol, state in list(self._open.items()):
            state.bars_held += 1
            hyp_snap = state.hypothesis_snapshot

            # Check invalidation_rule
            invalidation_rule = hyp_snap.get("invalidation_rule")
            if invalidation_rule:
                try:
                    fired = evaluator.evaluate(invalidation_rule, context)
                    if fired:
                        exit_signals.append(HypothesisExitSignal(
                            hypothesis_id=state.hypothesis_id,
                            symbol=symbol,
                            direction=state.direction,
                            exit_reason="invalidated",
                            exit_price=current_price,
                            stop_price=state.stop_price,
                            target_price=state.target_price,
                            entry_price=state.entry_price,
                            bars_held=state.bars_held,
                            hypothesis_snapshot=hyp_snap,
                        ))
                        logger.info(
                            "hypothesis_executor: %s invalidated after %d bars (rule fired)",
                            state.hypothesis_id, state.bars_held,
                        )
                        continue
                except Exception as e:
                    logger.debug(
                        "hypothesis_executor: invalidation_rule eval error for %s: %s",
                        state.hypothesis_id, e,
                    )

            # Check expiry horizon
            horizon = hyp_snap.get("invalidation_horizon_bars", 12)
            if state.bars_held >= horizon:
                exit_signals.append(HypothesisExitSignal(
                    hypothesis_id=state.hypothesis_id,
                    symbol=symbol,
                    direction=state.direction,
                    exit_reason="expired",
                    exit_price=current_price,
                    stop_price=state.stop_price,
                    target_price=state.target_price,
                    entry_price=state.entry_price,
                    bars_held=state.bars_held,
                    hypothesis_snapshot=hyp_snap,
                ))
                logger.info(
                    "hypothesis_executor: %s expired after %d bars (horizon=%d)",
                    state.hypothesis_id, state.bars_held, horizon,
                )

        # ── Process PENDING hypotheses ────────────────────────────────────
        for symbol, pending_list in list(self._pending.items()):
            if not pending_list:
                continue

            # Skip if there's an open position for this symbol
            if symbol in self._open:
                continue

            # Evaluate only the first (highest priority) pending hypothesis
            hyp = pending_list[0]

            # Check pending expiry
            # Pending hypotheses use invalidation_horizon_bars from initial plan time
            # We don't track per-pending-bar-count currently; use bar_count delta
            # For simplicity, pending hypotheses don't expire here — they expire
            # when the plan is replaced by a new generation. This matches existing
            # TriggerCondition behavior.

            # Evaluate entry_rule
            try:
                fired = evaluator.evaluate(hyp.entry_rule, context)
            except Exception as e:
                block_records.append({
                    "hypothesis_id": hyp.id,
                    "symbol": symbol,
                    "reason": "entry_rule_eval_error",
                    "detail": str(e),
                    "price": current_price,
                    "block_class": "infra",
                })
                continue

            if fired:
                entry_signals.append({
                    "hypothesis_id": hyp.id,
                    "symbol": symbol,
                    "direction": hyp.direction,
                    "stop_price": hyp.stop_price,
                    "target_price": hyp.target_price,
                    "trailing_stop_activation_r": hyp.trailing_stop_activation_r,
                    "confidence_grade": hyp.confidence_grade,
                    "hypothesis_snapshot": hyp.model_dump(),
                })
                logger.info(
                    "hypothesis_executor: entry fired for %s (%s %s)",
                    hyp.id, hyp.direction, symbol,
                )

        return entry_signals, block_records, exit_signals

    def _build_eval_context(
        self,
        indicator: "IndicatorSnapshot",
        current_price: float,
    ) -> Dict[str, Any]:
        """Build a minimal evaluation context from the indicator snapshot.

        Mirrors the context built by TriggerEngine._context() for compatibility
        with the same DSL evaluator.
        """
        ctx: Dict[str, Any] = {
            "close": current_price,
            "price": current_price,
        }
        # Inject all non-None indicator fields as context keys
        for fname in indicator.model_fields:
            val = getattr(indicator, fname, None)
            if val is not None:
                ctx[fname] = val
        return ctx

    # ── Fill and exit handling ────────────────────────────────────────────

    def on_fill(
        self,
        symbol: str,
        fill_price: float,
        hypothesis_id: str,
    ) -> None:
        """Record entry fill: move hypothesis from PENDING to OPEN.

        Called by PaperTradingWorkflow after an entry order fills.
        """
        pending_list = self._pending.get(symbol, [])
        matched = None
        remaining = []
        for h in pending_list:
            if h.id == hypothesis_id and matched is None:
                matched = h
            else:
                remaining.append(h)

        if matched is None:
            logger.warning(
                "hypothesis_executor.on_fill: %s not found in pending for %s",
                hypothesis_id, symbol,
            )
            return

        self._pending[symbol] = remaining

        # Compute trailing stop initial price if activation_r is set
        trailing_stop_price = None
        if matched.trailing_stop_activation_r is not None:
            stop_dist = abs(fill_price - matched.stop_price)
            activation_dist = matched.trailing_stop_activation_r * stop_dist
            if matched.direction == "long":
                trailing_stop_price = fill_price + activation_dist
            else:
                trailing_stop_price = fill_price - activation_dist

        self._open[symbol] = OpenHypothesisState(
            hypothesis_id=hypothesis_id,
            symbol=symbol,
            direction=matched.direction,
            entry_price=fill_price,
            entry_bar_index=self._bar_count,
            stop_price=matched.stop_price,
            target_price=matched.target_price,
            trailing_stop_price=trailing_stop_price,
            trailing_activation_r=matched.trailing_stop_activation_r,
            hypothesis_snapshot=matched.model_dump(),
        )
        logger.info(
            "hypothesis_executor.on_fill: %s OPEN at %.4f "
            "(stop=%.4f target=%.4f trail_activation=%s)",
            hypothesis_id, fill_price,
            matched.stop_price, matched.target_price,
            matched.trailing_stop_activation_r,
        )

    def on_exit(
        self,
        symbol: str,
        exit_price: float,
        exit_reason: str,
    ) -> Optional[Dict[str, Any]]:
        """Record exit: move hypothesis from OPEN to COMPLETED.

        Returns the completed hypothesis attribution record, or None if
        no hypothesis was open for this symbol.
        """
        state = self._open.pop(symbol, None)
        if state is None:
            return None

        pnl_pct = (
            (exit_price - state.entry_price) / state.entry_price * 100
            if state.direction == "long"
            else (state.entry_price - exit_price) / state.entry_price * 100
        )

        record = {
            "hypothesis_id": state.hypothesis_id,
            "symbol": symbol,
            "direction": state.direction,
            "entry_price": state.entry_price,
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "bars_held": state.bars_held,
            "stop_price": state.stop_price,
            "target_price": state.target_price,
            "pnl_pct": round(pnl_pct, 4),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "hypothesis_snapshot": state.hypothesis_snapshot,
        }

        self._completed.setdefault(symbol, []).append(record)
        logger.info(
            "hypothesis_executor.on_exit: %s COMPLETED at %.4f (%s, pnl=%.2f%%, bars=%d)",
            state.hypothesis_id, exit_price, exit_reason, pnl_pct, state.bars_held,
        )
        return record

    # ── State queries ──────────────────────────────────────────────────────

    def get_open_state(self, symbol: str) -> Optional[OpenHypothesisState]:
        return self._open.get(symbol)

    def get_pending(self, symbol: str) -> List[TradeHypothesis]:
        return list(self._pending.get(symbol, []))

    def has_open_position(self, symbol: str) -> bool:
        return symbol in self._open

    def get_completed(self, symbol: str) -> List[Dict[str, Any]]:
        return list(self._completed.get(symbol, []))

    def all_symbols(self) -> List[str]:
        symbols = set(self._pending.keys()) | set(self._open.keys())
        return sorted(symbols)

    # ── Serialization (for SessionState / _snapshot_state) ────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bar_count": self._bar_count,
            "pending": {
                sym: [h.model_dump() for h in hyps]
                for sym, hyps in self._pending.items()
            },
            "open": {
                sym: state.to_dict() for sym, state in self._open.items()
            },
            "completed": dict(self._completed),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HypothesisExecutor":
        executor = cls()
        executor._bar_count = d.get("bar_count", 0)
        executor._completed = {k: list(v) for k, v in d.get("completed", {}).items()}
        for sym, hyps_raw in d.get("pending", {}).items():
            executor._pending[sym] = [
                TradeHypothesis.model_validate(h) for h in hyps_raw
            ]
        for sym, state_raw in d.get("open", {}).items():
            executor._open[sym] = OpenHypothesisState.from_dict(state_raw)
        return executor
