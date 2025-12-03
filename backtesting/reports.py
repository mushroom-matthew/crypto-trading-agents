"""Utilities for aggregating backtest daily reports into run-level summaries."""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping, Sequence


def _safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    return mean(values) if values else 0.0


def _safe_median(values: Iterable[float]) -> float:
    values = list(values)
    return median(values) if values else 0.0


def _pearson(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    mx = mean(x)
    my = mean(y)
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    varx = sum((a - mx) ** 2 for a in x)
    vary = sum((b - my) ** 2 for b in y)
    denom = math.sqrt(varx * vary)
    return (cov / denom) if denom else 0.0


def build_run_summary(daily_reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate daily reports into a compact run-level summary."""

    risk_usage: list[float] = []
    returns: list[float] = []
    trades: list[int] = []
    blocked_daily_cap: list[int] = []
    blocked_plan: list[int] = []
    blocked_direction: list[int] = []
    execution_rates: list[float] = []
    brake_counts = {"daily_cap": 0, "plan_limit": 0, "both": 0, "neither": 0}

    for report in daily_reports:
        limit_stats = report.get("limit_stats") or {}
        risk_entry = report.get("risk_budget") or {}
        used_pct = risk_entry.get("used_pct")
        if used_pct is None:
            used_pct = limit_stats.get("risk_budget_used_pct", 0.0)
        risk_usage.append(float(used_pct or 0.0))
        returns.append(float(report.get("equity_return_pct", 0.0)))
        trades.append(int(report.get("trade_count", 0)))
        blocked_daily_cap.append(int(limit_stats.get("blocked_by_daily_cap", 0)))
        blocked_plan.append(int(limit_stats.get("blocked_by_plan_limits", 0)))
        blocked_direction.append(int(limit_stats.get("blocked_by_direction", 0)))
        attempted = report.get("attempted_triggers") or limit_stats.get("attempted_triggers") or 0
        executed = report.get("executed_trades") or limit_stats.get("executed_trades") or 0
        if attempted:
            execution_rates.append(executed / attempted)
        daily_cap = blocked_daily_cap[-1] > 0
        plan_cap = blocked_plan[-1] > 0
        if daily_cap and plan_cap:
            brake_counts["both"] += 1
        elif daily_cap:
            brake_counts["daily_cap"] += 1
        elif plan_cap:
            brake_counts["plan_limit"] += 1
        else:
            brake_counts["neither"] += 1

    risk_under_10 = sum(1 for val in risk_usage if val < 10.0)
    corr = _pearson(risk_usage, returns) if len(risk_usage) >= 2 else 0.0
    days = len(daily_reports)
    summary = {
        "days": days,
        "risk_budget_used_pct_mean": _safe_mean(risk_usage),
        "risk_budget_used_pct_median": _safe_median(risk_usage),
        "risk_budget_under_10_pct_days": (risk_under_10 / days * 100.0) if days else 0.0,
        "trade_count_mean": _safe_mean(trades),
        "blocked_by_daily_cap_mean": _safe_mean(blocked_daily_cap),
        "blocked_by_plan_limits_mean": _safe_mean(blocked_plan),
        "blocked_by_direction_mean": _safe_mean(blocked_direction),
        "execution_rate_mean": _safe_mean(execution_rates),
        "risk_usage_vs_return_corr": corr,
        "brake_distribution": brake_counts,
    }
    return summary


def write_run_summary(daily_reports: Sequence[Mapping[str, Any]], target_path: Path) -> dict[str, Any]:
    """Build and write a run summary JSON next to daily reports."""

    summary = build_run_summary(daily_reports)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(summary, indent=2))
    return summary
