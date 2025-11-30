"""Utilities for summarizing recent judge evaluations into strategist memory."""

from __future__ import annotations

from collections import deque
from statistics import mean
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def _empty_memory() -> Dict[str, Any]:
    return {
        "recent_evaluations": [],
        "risk_events": {"bad_exits": 0, "late_entries": 0, "overtrading": 0},
        "trigger_performance": {},
        "trade_behavior": {
            "avg_trades_per_day": 0.0,
            "avg_return_pct": 0.0,
            "avg_duration_hours": None,
            "win_rate": 0.0,
        },
        "equity_trend": {"direction": "flat", "slope": 0.0},
        "constraints_snapshot": {},
    }


def _issue_flags(summary: Dict[str, Any]) -> Dict[str, bool]:
    notes = (summary.get("judge_feedback", {}).get("notes") or "").lower()
    return_pct = float(summary.get("return_pct", 0.0))
    trade_count = int(summary.get("trade_count", 0))
    flags = {
        "bad_exits": return_pct < -0.5 or "drawdown" in notes,
        "late_entries": "late" in notes or "miss" in notes,
        "overtrading": trade_count > 12 or "over-trading" in notes,
    }
    return flags


def _iter_triggers(summary: Dict[str, Any]) -> Iterable[Tuple[str, int]]:
    for entry in summary.get("top_triggers", []):
        if isinstance(entry, dict):
            trig_id = entry.get("id") or entry.get("trigger")
            count = int(entry.get("count", 1))
            if trig_id:
                yield trig_id, count
        elif isinstance(entry, (list, tuple)) and len(entry) == 2:
            trig_id, count = entry
            if isinstance(trig_id, str):
                yield trig_id, int(count)
        elif isinstance(entry, str):
            yield entry, 1


def build_strategy_memory(history: Sequence[Dict[str, Any]], limit: int = 8) -> Dict[str, Any]:
    """Summarize recent judge feedback for the strategist prompt."""

    if not history:
        return _empty_memory()

    tail: List[Dict[str, Any]] = list(deque(history, maxlen=limit))
    recent_evals: List[Dict[str, Any]] = []
    issue_counts = {"bad_exits": 0, "late_entries": 0, "overtrading": 0}
    trigger_perf: Dict[str, Dict[str, float]] = {}
    returns: List[float] = []
    trade_counts: List[int] = []
    end_equities: List[float] = []
    win_days = 0
    last_constraints: Dict[str, Any] | None = None

    for entry in tail:
        judge = entry.get("judge_feedback", {})
        recent_evals.append(
            {
                "date": entry.get("date"),
                "score": judge.get("score"),
                "return_pct": entry.get("return_pct"),
                "trade_count": entry.get("trade_count"),
                "notes": judge.get("notes"),
            }
        )
        flags = _issue_flags(entry)
        for name, flagged in flags.items():
            if flagged:
                issue_counts[name] += 1
        day_return = float(entry.get("return_pct", 0.0))
        returns.append(day_return)
        if day_return > 0:
            win_days += 1
        trade_counts.append(int(entry.get("trade_count", 0)))
        if isinstance(entry.get("end_equity"), (int, float)):
            end_equities.append(float(entry["end_equity"]))
        for trig_id, count in _iter_triggers(entry):
            stats = trigger_perf.setdefault(trig_id, {"fires": 0.0, "weighted_return": 0.0})
            stats["fires"] += float(count)
            stats["weighted_return"] += float(count) * day_return
        constraints = judge.get("strategist_constraints")
        if constraints:
            last_constraints = constraints

    trigger_summary = {
        trig: {
            "fires": stats["fires"],
            "avg_return_pct": stats["weighted_return"] / stats["fires"] if stats["fires"] else 0.0,
        }
        for trig, stats in trigger_perf.items()
    }

    avg_trades = mean(trade_counts) if trade_counts else 0.0
    avg_return = mean(returns) if returns else 0.0
    duration_samples = [24.0 / max(count, 1) for count in trade_counts if count]
    avg_duration = mean(duration_samples) if duration_samples else None
    slope = 0.0
    if len(end_equities) >= 2:
        slope = (end_equities[-1] - end_equities[0]) / (len(end_equities) - 1)
    direction = "up" if slope > 0 else "down" if slope < 0 else "flat"
    win_rate = win_days / len(tail) if tail else 0.0

    memory = {
        "recent_evaluations": recent_evals,
        "risk_events": issue_counts,
        "trigger_performance": trigger_summary,
        "trade_behavior": {
            "avg_trades_per_day": avg_trades,
            "avg_return_pct": avg_return,
            "avg_duration_hours": avg_duration,
            "win_rate": win_rate,
        },
        "equity_trend": {"direction": direction, "slope": slope},
        "constraints_snapshot": last_constraints or {},
    }
    return memory
