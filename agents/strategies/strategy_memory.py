"""Utilities for summarizing recent judge evaluations into strategist memory."""

from __future__ import annotations

from collections import deque, defaultdict
from statistics import mean
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def _empty_memory() -> Dict[str, Any]:
    return {
        "recent_evaluations": [],
        "risk_events": {"bad_exits": 0, "late_entries": 0, "overtrading": 0},
        "trigger_performance": {},
        "category_performance": {},
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
    category_perf: Dict[str, Dict[str, float]] = {}
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
        catalog = (entry.get("plan_limits") or {}).get("trigger_catalog") or {}
        trigger_stats = entry.get("trigger_stats") or {}
        symbol_pnl = entry.get("symbol_pnl") or {}
        symbol_exec_totals: Dict[str, float] = defaultdict(float)
        for trig_id, stats in trigger_stats.items():
            symbol = catalog.get(trig_id, {}).get("symbol")
            if symbol:
                symbol_exec_totals[symbol] += float(stats.get("executed", 0))
        for trig_id, stats in trigger_stats.items():
            meta = catalog.get(trig_id) or {}
            symbol = meta.get("symbol")
            category = meta.get("category") or "other"
            executed = float(stats.get("executed", 0))
            blocked = float(stats.get("blocked", 0))
            perf = trigger_perf.setdefault(
                trig_id,
                {
                    "symbol": symbol,
                    "category": category,
                    "fires": 0.0,
                    "blocked": 0.0,
                    "wins": 0.0,
                    "gross_pct": 0.0,
                    "net_pct": 0.0,
                },
            )
            perf["fires"] += executed
            perf["blocked"] += blocked
            gross_share = 0.0
            net_share = 0.0
            if symbol and executed > 0 and symbol_exec_totals.get(symbol):
                symbol_stats = symbol_pnl.get(symbol, {})
                share = executed / symbol_exec_totals[symbol]
                gross_share = symbol_stats.get("gross_pct", 0.0) * share
                net_share = symbol_stats.get("net_pct", 0.0) * share
                perf["gross_pct"] += gross_share
                perf["net_pct"] += net_share
                if net_share > 0:
                    perf["wins"] += executed
            cat_stats = category_perf.setdefault(
                category,
                {"executed": 0.0, "gross_pct": 0.0, "net_pct": 0.0, "wins": 0.0},
            )
            cat_stats["executed"] += executed
            cat_stats["gross_pct"] += gross_share
            cat_stats["net_pct"] += net_share
            if net_share > 0:
                cat_stats["wins"] += executed
        for trig_id, count in _iter_triggers(entry):
            stats = trigger_perf.setdefault(
                trig_id,
                {
                    "symbol": catalog.get(trig_id, {}).get("symbol"),
                    "category": catalog.get(trig_id, {}).get("category") or "other",
                    "fires": 0.0,
                    "blocked": 0.0,
                    "wins": 0.0,
                    "gross_pct": 0.0,
                    "net_pct": 0.0,
                    "weighted_return": 0.0,
                },
            )
            stats.setdefault("weighted_return", 0.0)
            stats["weighted_return"] += float(count) * day_return
            stats["fires"] += float(count)
        constraints = judge.get("strategist_constraints")
        if constraints:
            last_constraints = constraints

    trigger_summary = {}
    for trig, stats in trigger_perf.items():
        fires = stats.get("fires", 0.0)
        weighted_return = stats.get("weighted_return", 0.0)
        wins = stats.get("wins", 0.0)
        trigger_summary[trig] = {
            "symbol": stats.get("symbol"),
            "category": stats.get("category"),
            "fires": fires,
            "blocked": stats.get("blocked", 0.0),
            "avg_return_pct": (weighted_return / fires) if fires else 0.0,
            "win_rate": (wins / fires) if fires else 0.0,
            "gross_pct": stats.get("gross_pct", 0.0),
            "net_pct": stats.get("net_pct", 0.0),
        }

    category_summary = {}
    for category, stats in category_perf.items():
        executed = stats.get("executed", 0.0)
        category_summary[category] = {
            "executed": executed,
            "gross_pct": stats.get("gross_pct", 0.0),
            "net_pct": stats.get("net_pct", 0.0),
            "win_rate": (stats.get("wins", 0.0) / executed) if executed else 0.0,
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
        "category_performance": category_summary,
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
