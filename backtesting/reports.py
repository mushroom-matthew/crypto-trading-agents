"""Utilities for aggregating backtest daily reports into run-level summaries."""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping, Sequence

GOOD_RPR_THRESHOLD = 0.2
BAD_RPR_THRESHOLD = -0.2
MIN_TRADES_FOR_RPR = 5
DELTA_PENALTY = 0.1
GOOD_MULTIPLIER = 1.2
NEUTRAL_MULTIPLIER = 1.0
BAD_MULTIPLIER = 0.5


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


def _multiplier_for_label(label: str) -> float:
    if label == "good":
        return GOOD_MULTIPLIER
    if label == "bad":
        return BAD_MULTIPLIER
    return NEUTRAL_MULTIPLIER


def _label_for_rpr(llm_rpr: float, trades: int, baseline_rpr: float) -> str:
    if trades < MIN_TRADES_FOR_RPR:
        return "neutral"
    if llm_rpr >= GOOD_RPR_THRESHOLD and (baseline_rpr - llm_rpr) < DELTA_PENALTY:
        return "good"
    if llm_rpr <= BAD_RPR_THRESHOLD or (baseline_rpr - llm_rpr) >= DELTA_PENALTY:
        return "bad"
    return "neutral"


def _comparison_entry(llm_payload: Mapping[str, Any] | None, baseline_payload: Mapping[str, Any] | None) -> dict[str, Any]:
    llm_rpr = float((llm_payload or {}).get("rpr_actual", 0.0))
    llm_trades = int((llm_payload or {}).get("trades", 0))
    baseline_rpr = float((baseline_payload or {}).get("rpr_actual", 0.0))
    baseline_trades = int((baseline_payload or {}).get("trades", 0))
    delta_rpr = llm_rpr - baseline_rpr
    label = _label_for_rpr(llm_rpr, llm_trades, baseline_rpr)
    return {
        "llm": {"rpr_actual": llm_rpr, "trades": llm_trades},
        "baseline": {"rpr_actual": baseline_rpr, "trades": baseline_trades},
        "delta": {"rpr_actual": delta_rpr},
        "label": label,
        "multiplier": _multiplier_for_label(label),
    }


def _build_rpr_comparison(current_summary: Mapping[str, Any], baseline_summary: Mapping[str, Any]) -> dict[str, Any]:
    archetype_entries: dict[str, Any] = {}
    archetype_hour_entries: dict[str, Any] = {}
    current_archetypes = current_summary.get("archetype_quality") or {}
    baseline_archetypes = baseline_summary.get("archetype_quality") or {}
    baseline_hours = baseline_summary.get("archetype_hour_quality") or {}
    if not baseline_archetypes and not baseline_hours:
        return {
            "archetypes": {},
            "archetype_hours": {},
            "thresholds": {
                "good_rpr_threshold": GOOD_RPR_THRESHOLD,
                "bad_rpr_threshold": BAD_RPR_THRESHOLD,
                "min_trades": MIN_TRADES_FOR_RPR,
                "delta_penalty": DELTA_PENALTY,
                "multipliers": {
                    "good": GOOD_MULTIPLIER,
                    "neutral": NEUTRAL_MULTIPLIER,
                    "bad": BAD_MULTIPLIER,
                },
            },
            "baseline_missing": True,
        }
    for key in set(current_archetypes.keys()) | set(baseline_archetypes.keys()):
        archetype_entries[key] = _comparison_entry(current_archetypes.get(key), baseline_archetypes.get(key))
    current_hour = current_summary.get("archetype_hour_quality") or {}
    for key in set(current_hour.keys()) | set(baseline_hours.keys()):
        archetype_hour_entries[key] = _comparison_entry(current_hour.get(key), baseline_hours.get(key))
    return {
        "archetypes": archetype_entries,
        "archetype_hours": archetype_hour_entries,
        "thresholds": {
            "good_rpr_threshold": GOOD_RPR_THRESHOLD,
            "bad_rpr_threshold": BAD_RPR_THRESHOLD,
            "min_trades": MIN_TRADES_FOR_RPR,
            "delta_penalty": DELTA_PENALTY,
            "multipliers": {
                "good": GOOD_MULTIPLIER,
                "neutral": NEUTRAL_MULTIPLIER,
                "bad": BAD_MULTIPLIER,
            },
        },
    }


def build_run_summary(daily_reports: Sequence[Mapping[str, Any]], baseline_summary: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Aggregate daily reports into a compact run-level summary."""

    risk_usage_pct: list[float] = []
    risk_utilization_pct: list[float] = []
    returns: list[float] = []
    trades: list[int] = []
    blocked_daily_cap: list[int] = []
    blocked_plan: list[int] = []
    blocked_direction: list[int] = []
    execution_rates: list[float] = []
    brake_counts = {"daily_cap": 0, "plan_limit": 0, "both": 0, "neither": 0}
    timeframe_counts: dict[str, int] = {}
    timeframe_executions: dict[str, int] = {}
    hour_counts: dict[int, int] = {}
    hour_executions: dict[int, int] = {}
    flatten_pcts: list[float] = []
    fee_pcts: list[float] = []
    flatten_days = 0
    trigger_quality: dict[str, dict[str, float | int]] = {}
    timeframe_quality: dict[str, dict[str, float | int]] = {}
    hour_quality: dict[str, dict[str, float | int]] = {}
    archetype_quality: dict[str, dict[str, float | int]] = {}
    archetype_hour_quality: dict[str, dict[str, float | int]] = {}
    factor_exposures: dict[str, Any] = {}
    block_totals: dict[str, int] = {}
    risk_profile_snapshot: dict[str, Any] | None = None
    llm_fallback_days = 0
    llm_plan_fail_days = 0
    rpr_source_days = 0
    rpr_filtered_days = 0

    for report in daily_reports:
        data_quality = report.get("llm_data_quality") or "ok"
        if data_quality == "fallback":
            llm_fallback_days += 1
        if report.get("llm_failed_parse"):
            llm_plan_fail_days += 1
        limit_stats = report.get("limit_stats") or {}
        base_blocks = limit_stats.get("risk_block_breakdown") or {}
        block_map = base_blocks | {
            "daily_loss": report.get("daily_loss_blocks", 0),
            "daily_cap": report.get("daily_cap_blocks", 0),
            "risk_budget": report.get("risk_budget_blocks", 0),
            "session_cap": report.get("session_cap_blocks", 0),
            "archetype_load": report.get("archetype_load_blocks", 0),
            "trigger_load": report.get("trigger_load_blocks", 0),
            "symbol_cap": report.get("symbol_cap_blocks", 0),
            # Aliases to satisfy reporting consumers.
            "max_daily_loss_pct": base_blocks.get("max_daily_loss_pct", 0),
            "max_daily_risk_budget_pct": base_blocks.get("risk_budget", 0),
            "max_daily_cap": base_blocks.get("daily_cap", 0),
            "max_symbol_exposure_pct": base_blocks.get("max_symbol_exposure_pct", 0),
        }
        for key, val in block_map.items():
            try:
                block_totals[key] = block_totals.get(key, 0) + int(val or 0)
            except (TypeError, ValueError):
                continue
        risk_entry = report.get("risk_budget") or {}
        if not risk_entry and report.get("risk_budget_pct"):
            budget_pct = float(report.get("risk_budget_pct") or 0.0)
            budget_abs = float(report.get("start_equity", 0.0)) * (budget_pct / 100.0)
            used_abs = sum(float(evt.get("risk_used", 0.0)) for evt in (report.get("risk_usage_events") or []))
            used_pct = (used_abs / budget_abs * 100.0) if budget_abs > 0 else 0.0
            risk_entry = {
                "budget_pct": budget_pct,
                "budget_abs": budget_abs,
                "used_abs": used_abs,
                "used_pct": used_pct,
                "utilization_pct": used_pct,
            }
        used_pct = risk_entry.get("used_pct")
        if used_pct is None:
            used_pct = limit_stats.get("risk_budget_used_pct", 0.0)
        risk_usage_pct.append(float(used_pct or 0.0))
        utilization = risk_entry.get("utilization_pct", used_pct)
        risk_utilization_pct.append(float(utilization or 0.0))
        returns.append(float(report.get("equity_return_pct", 0.0)))
        trades.append(int(report.get("trade_count", 0)))
        blocked_daily_cap.append(int(limit_stats.get("blocked_by_daily_cap", 0)))
        blocked_plan.append(int(limit_stats.get("blocked_by_plan_limits", 0)))
        blocked_direction.append(int(limit_stats.get("blocked_by_direction", 0)))
        attempted = report.get("attempted_triggers") or limit_stats.get("attempted_triggers") or 0
        executed = report.get("executed_trades") or limit_stats.get("executed_trades") or 0
        if attempted:
            execution_rates.append(executed / attempted)
        pnl_breakdown = report.get("pnl_breakdown") or {}
        flattening_pct = float(pnl_breakdown.get("flattening_pct", 0.0))
        fees_pct = float(pnl_breakdown.get("fees_pct", 0.0))
        flatten_pcts.append(flattening_pct)
        fee_pcts.append(fees_pct)
        if abs(flattening_pct) > 1e-9 or report.get("flatten_positions_daily") or report.get("flatten_session_hour") is not None:
            flatten_days += 1
        include_for_rpr = data_quality != "fallback"
        if include_for_rpr:
            rpr_source_days += 1
        else:
            rpr_filtered_days += 1
            continue
        for key, payload in (report.get("trigger_quality") or {}).items():
            agg = trigger_quality.setdefault(
                key,
                {
                    "pnl": 0.0,
                    "risk_used_abs": 0.0,
                    "trades": 0,
                    "wins": 0,
                    "losses": 0,
                    "latency_sum": 0.0,
                    "latency_count": 0,
                    "mae_sum": 0.0,
                    "mfe_sum": 0.0,
                    "decay_sum": 0.0,
                    "abs_mae_sum": 0.0,
                    "abs_mfe_sum": 0.0,
                    "win_abs_sum": 0.0,
                    "loss_abs_sum": 0.0,
                    "load_sum": 0.0,
                    "load_count": 0,
                    "actual_risk_abs": 0.0,
                },
            )
            agg["pnl"] += float(payload.get("pnl", 0.0))
            agg["risk_used_abs"] += float(payload.get("risk_used_abs", 0.0))
            agg["actual_risk_abs"] += float(payload.get("actual_risk_abs", 0.0))
            agg["trades"] += int(payload.get("trades", 0))
            agg["wins"] += int(payload.get("wins", 0))
            agg["losses"] += int(payload.get("losses", 0))
            latency = payload.get("latency_seconds")
            if latency is not None:
                agg["latency_sum"] += float(latency)
                agg["latency_count"] += 1
            agg["mae_sum"] += float(payload.get("mae_pct", 0.0)) * float(payload.get("trades", 0))
            agg["mfe_sum"] += float(payload.get("mfe_pct", 0.0)) * float(payload.get("trades", 0))
            agg["decay_sum"] += float(payload.get("response_decay_pct", 0.0)) * float(payload.get("trades", 0))
            agg["abs_mae_sum"] += float(payload.get("mae_pct", 0.0)).__abs__() * float(payload.get("trades", 0))
            agg["abs_mfe_sum"] += float(payload.get("mfe_pct", 0.0)).__abs__() * float(payload.get("trades", 0))
            agg["win_abs_sum"] += float(payload.get("pnl", 0.0)) if payload.get("pnl", 0.0) > 0 else 0.0
            agg["loss_abs_sum"] += abs(float(payload.get("pnl", 0.0))) if payload.get("pnl", 0.0) < 0 else 0.0
            load_count = int(payload.get("load_count", 0))
            agg["load_sum"] += float(payload.get("avg_load", 0.0)) * load_count
            agg["load_count"] += load_count
        if not include_for_rpr:
            continue
        for tf, payload in (report.get("timeframe_quality") or {}).items():
            agg = timeframe_quality.setdefault(
                tf,
                {
                    "pnl": 0.0,
                    "risk_used_abs": 0.0,
                    "trades": 0,
                    "wins": 0,
                    "losses": 0,
                    "latency_sum": 0.0,
                    "latency_count": 0,
                    "mae_sum": 0.0,
                    "mfe_sum": 0.0,
                    "decay_sum": 0.0,
                    "abs_mae_sum": 0.0,
                    "abs_mfe_sum": 0.0,
                    "win_abs_sum": 0.0,
                    "loss_abs_sum": 0.0,
                    "load_sum": 0.0,
                    "load_count": 0,
                    "actual_risk_abs": 0.0,
                },
            )
            agg["pnl"] += float(payload.get("pnl", 0.0))
            agg["risk_used_abs"] += float(payload.get("risk_used_abs", 0.0))
            agg["actual_risk_abs"] += float(payload.get("actual_risk_abs", 0.0))
            agg["trades"] += int(payload.get("trades", 0))
            agg["wins"] += int(payload.get("wins", 0))
            agg["losses"] += int(payload.get("losses", 0))
            latency = payload.get("latency_seconds")
            if latency is not None:
                agg["latency_sum"] += float(latency)
                agg["latency_count"] += 1
            agg["mae_sum"] += float(payload.get("mae_pct", 0.0)) * float(payload.get("trades", 0))
            agg["mfe_sum"] += float(payload.get("mfe_pct", 0.0)) * float(payload.get("trades", 0))
            agg["decay_sum"] += float(payload.get("response_decay_pct", 0.0)) * float(payload.get("trades", 0))
            agg["abs_mae_sum"] += float(payload.get("mae_pct", 0.0)).__abs__() * float(payload.get("trades", 0))
            agg["abs_mfe_sum"] += float(payload.get("mfe_pct", 0.0)).__abs__() * float(payload.get("trades", 0))
            agg["win_abs_sum"] += float(payload.get("pnl", 0.0)) if payload.get("pnl", 0.0) > 0 else 0.0
            agg["loss_abs_sum"] += abs(float(payload.get("pnl", 0.0))) if payload.get("pnl", 0.0) < 0 else 0.0
            load_count = int(payload.get("load_count", 0))
            agg["load_sum"] += float(payload.get("avg_load", 0.0)) * load_count
            agg["load_count"] += load_count
        for hour, payload in (report.get("hour_quality") or {}).items():
            agg = hour_quality.setdefault(
                hour,
                {
                    "pnl": 0.0,
                    "risk_used_abs": 0.0,
                    "trades": 0,
                    "wins": 0,
                    "losses": 0,
                    "latency_sum": 0.0,
                    "latency_count": 0,
                    "mae_sum": 0.0,
                    "mfe_sum": 0.0,
                    "decay_sum": 0.0,
                    "abs_mae_sum": 0.0,
                    "abs_mfe_sum": 0.0,
                    "win_abs_sum": 0.0,
                    "loss_abs_sum": 0.0,
                    "load_sum": 0.0,
                    "load_count": 0,
                    "actual_risk_abs": 0.0,
                },
            )
            agg["pnl"] += float(payload.get("pnl", 0.0))
            agg["risk_used_abs"] += float(payload.get("risk_used_abs", 0.0))
            agg["actual_risk_abs"] += float(payload.get("actual_risk_abs", 0.0))
            agg["trades"] += int(payload.get("trades", 0))
            agg["wins"] += int(payload.get("wins", 0))
            agg["losses"] += int(payload.get("losses", 0))
            latency = payload.get("latency_seconds")
            if latency is not None:
                agg["latency_sum"] += float(latency)
                agg["latency_count"] += 1
            agg["mae_sum"] += float(payload.get("mae_pct", 0.0)) * float(payload.get("trades", 0))
            agg["mfe_sum"] += float(payload.get("mfe_pct", 0.0)) * float(payload.get("trades", 0))
            agg["decay_sum"] += float(payload.get("response_decay_pct", 0.0)) * float(payload.get("trades", 0))
            agg["abs_mae_sum"] += float(payload.get("mae_pct", 0.0)).__abs__() * float(payload.get("trades", 0))
            agg["abs_mfe_sum"] += float(payload.get("mfe_pct", 0.0)).__abs__() * float(payload.get("trades", 0))
            agg["win_abs_sum"] += float(payload.get("pnl", 0.0)) if payload.get("pnl", 0.0) > 0 else 0.0
            agg["loss_abs_sum"] += abs(float(payload.get("pnl", 0.0))) if payload.get("pnl", 0.0) < 0 else 0.0
        for archetype, payload in (report.get("archetype_quality") or {}).items():
            agg = archetype_quality.setdefault(
                archetype,
                {
                    "pnl": 0.0,
                    "risk_used_abs": 0.0,
                    "trades": 0,
                    "wins": 0,
                    "losses": 0,
                    "latency_sum": 0.0,
                    "latency_count": 0,
                    "mae_sum": 0.0,
                    "mfe_sum": 0.0,
                    "decay_sum": 0.0,
                    "abs_mae_sum": 0.0,
                    "abs_mfe_sum": 0.0,
                    "win_abs_sum": 0.0,
                    "loss_abs_sum": 0.0,
                    "load_sum": 0.0,
                    "load_count": 0,
                    "actual_risk_abs": 0.0,
                },
            )
            agg["pnl"] += float(payload.get("pnl", 0.0))
            agg["risk_used_abs"] += float(payload.get("risk_used_abs", 0.0))
            agg["actual_risk_abs"] += float(payload.get("actual_risk_abs", 0.0))
            agg["trades"] += int(payload.get("trades", 0))
            agg["wins"] += int(payload.get("wins", 0))
            agg["losses"] += int(payload.get("losses", 0))
            latency = payload.get("latency_seconds")
            if latency is not None:
                agg["latency_sum"] += float(latency)
                agg["latency_count"] += 1
            agg["mae_sum"] += float(payload.get("mae_pct", 0.0)) * float(payload.get("trades", 0))
            agg["mfe_sum"] += float(payload.get("mfe_pct", 0.0)) * float(payload.get("trades", 0))
            agg["decay_sum"] += float(payload.get("response_decay_pct", 0.0)) * float(payload.get("trades", 0))
            agg["abs_mae_sum"] += float(payload.get("mae_pct", 0.0)).__abs__() * float(payload.get("trades", 0))
            agg["abs_mfe_sum"] += float(payload.get("mfe_pct", 0.0)).__abs__() * float(payload.get("trades", 0))
            agg["win_abs_sum"] += float(payload.get("pnl", 0.0)) if payload.get("pnl", 0.0) > 0 else 0.0
            agg["loss_abs_sum"] += abs(float(payload.get("pnl", 0.0))) if payload.get("pnl", 0.0) < 0 else 0.0
        for arch_hour, payload in (report.get("archetype_hour_quality") or {}).items():
            agg = archetype_hour_quality.setdefault(
                arch_hour,
                {
                    "pnl": 0.0,
                    "risk_used_abs": 0.0,
                    "trades": 0,
                    "wins": 0,
                    "losses": 0,
                    "latency_sum": 0.0,
                    "latency_count": 0,
                    "mae_sum": 0.0,
                    "mfe_sum": 0.0,
                    "decay_sum": 0.0,
                    "abs_mae_sum": 0.0,
                    "abs_mfe_sum": 0.0,
                    "win_abs_sum": 0.0,
                    "loss_abs_sum": 0.0,
                    "load_sum": 0.0,
                    "load_count": 0,
                    "actual_risk_abs": 0.0,
                },
            )
            agg["pnl"] += float(payload.get("pnl", 0.0))
            agg["risk_used_abs"] += float(payload.get("risk_used_abs", 0.0))
            agg["actual_risk_abs"] += float(payload.get("actual_risk_abs", 0.0))
            agg["trades"] += int(payload.get("trades", 0))
            agg["wins"] += int(payload.get("wins", 0))
            agg["losses"] += int(payload.get("losses", 0))
            latency = payload.get("latency_seconds")
            if latency is not None:
                agg["latency_sum"] += float(latency)
                agg["latency_count"] += 1
            agg["mae_sum"] += float(payload.get("mae_pct", 0.0)) * float(payload.get("trades", 0))
            agg["mfe_sum"] += float(payload.get("mfe_pct", 0.0)) * float(payload.get("trades", 0))
            agg["decay_sum"] += float(payload.get("response_decay_pct", 0.0)) * float(payload.get("trades", 0))
            agg["abs_mae_sum"] += float(payload.get("mae_pct", 0.0)).__abs__() * float(payload.get("trades", 0))
            agg["abs_mfe_sum"] += float(payload.get("mfe_pct", 0.0)).__abs__() * float(payload.get("trades", 0))
            agg["win_abs_sum"] += float(payload.get("pnl", 0.0)) if payload.get("pnl", 0.0) > 0 else 0.0
            agg["loss_abs_sum"] += abs(float(payload.get("pnl", 0.0))) if payload.get("pnl", 0.0) < 0 else 0.0
        for entry in (limit_stats.get("blocked_details") or []) + (limit_stats.get("executed_details") or []):
            tf = entry.get("timeframe")
            if tf:
                timeframe_counts[tf] = timeframe_counts.get(tf, 0) + 1
            outcome = entry.get("outcome") or ""
            reason = entry.get("reason")
            executed_outcome = outcome == "executed" or not reason
            if executed_outcome and tf:
                timeframe_executions[tf] = timeframe_executions.get(tf, 0) + 1
            ts = entry.get("timestamp")
            if ts and isinstance(ts, str) and len(ts) >= 13:
                try:
                    hour = int(ts[11:13])
                    hour_counts[hour] = hour_counts.get(hour, 0) + 1
                    if executed_outcome:
                        hour_executions[hour] = hour_executions.get(hour, 0) + 1
                except ValueError:
                    continue
        if report.get("factor_exposures"):
            factor_exposures = report["factor_exposures"]
        if report.get("risk_profile"):
            risk_profile_snapshot = report.get("risk_profile")
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

    def _finalize_quality(agg_map: dict[str, dict[str, float | int]]) -> dict[str, dict[str, float]]:
        finalized: dict[str, dict[str, float]] = {}
        for key, payload in agg_map.items():
            trades = int(payload.get("trades", 0))
            pnl = float(payload.get("pnl", 0.0))
            risk_used = float(payload.get("risk_used_abs", 0.0))
            wins = int(payload.get("wins", 0))
            losses = int(payload.get("losses", 0))
            latency_sum = float(payload.get("latency_sum", 0.0))
            latency_count = int(payload.get("latency_count", 0))
            latency_avg = (latency_sum / latency_count) if latency_count else None
            risk_per_trade = (risk_used / trades) if trades and risk_used else 0.0
            mean_r = (pnl / trades / risk_per_trade) if risk_per_trade else (pnl / trades if trades else 0.0)
            abs_mae_mean = (float(payload.get("abs_mae_sum", 0.0)) / trades) if trades else 0.0
            abs_mfe_mean = (float(payload.get("abs_mfe_sum", 0.0)) / trades) if trades else 0.0
            win_abs = float(payload.get("win_abs_sum", 0.0))
            loss_abs = float(payload.get("loss_abs_sum", 0.0))
            load_sum = float(payload.get("load_sum", 0.0))
            load_count = int(payload.get("load_count", 0))
            allocated_risk = float(payload.get("risk_used_abs", 0.0))
            actual_risk = float(payload.get("actual_risk_abs", 0.0))
            finalized[key] = {
                "trades": trades,
                "pnl": pnl,
                "risk_used_abs": risk_used,
                "rpr": (pnl / risk_used) if risk_used else 0.0,
                "rpr_allocated": (pnl / allocated_risk) if allocated_risk else 0.0,
                "rpr_actual": (pnl / actual_risk) if actual_risk else 0.0,
                "win_rate": (wins / trades) if trades else 0.0,
                "mean_r": mean_r,
                "latency_seconds": latency_avg,
                "mae_pct": (float(payload.get("mae_sum", 0.0)) / trades) if trades else 0.0,
                "mfe_pct": (float(payload.get("mfe_sum", 0.0)) / trades) if trades else 0.0,
                "response_decay_pct": (float(payload.get("decay_sum", 0.0)) / trades) if trades else 0.0,
                "relative_efficiency_mae": (mean_r / abs_mae_mean) if abs_mae_mean else 0.0,
                "relative_efficiency_mfe": (mean_r / abs_mfe_mean) if abs_mfe_mean else 0.0,
                "asymmetry": ((win_abs - loss_abs) / (win_abs + loss_abs)) if (win_abs + loss_abs) else 0.0,
                "avg_load": (load_sum / load_count) if load_count else 0.0,
                "load_sum": load_sum,
                "load_count": load_count,
            }
        return finalized

    risk_under_25 = sum(1 for val in risk_utilization_pct if val < 25.0)
    risk_25_to_75 = sum(1 for val in risk_utilization_pct if 25.0 <= val <= 75.0)
    risk_over_75 = sum(1 for val in risk_utilization_pct if val > 75.0)
    corr = _pearson(risk_utilization_pct, returns) if len(risk_utilization_pct) >= 2 else 0.0
    days = len(daily_reports)
    timeframe_exec_rate = {
        tf: (timeframe_executions.get(tf, 0) / timeframe_counts[tf]) if timeframe_counts.get(tf) else 0.0 for tf in timeframe_counts
    }
    hour_exec_rate = {
        hr: (hour_executions.get(hr, 0) / hour_counts[hr]) if hour_counts.get(hr) else 0.0 for hr in hour_counts
    }
    summary = {
        "days": days,
        "risk_budget_used_pct_mean": _safe_mean(risk_usage_pct),
        "risk_budget_used_pct_median": _safe_median(risk_usage_pct),
        "risk_budget_utilization_pct_mean": _safe_mean(risk_utilization_pct),
        "risk_budget_utilization_pct_median": _safe_median(risk_utilization_pct),
        "risk_budget_under_25_pct_days": (risk_under_25 / days * 100.0) if days else 0.0,
        "risk_budget_25_to_75_pct_days": (risk_25_to_75 / days * 100.0) if days else 0.0,
        "risk_budget_over_75_pct_days": (risk_over_75 / days * 100.0) if days else 0.0,
        "trade_count_mean": _safe_mean(trades),
        "blocked_by_daily_cap_mean": _safe_mean(blocked_daily_cap),
        "blocked_by_plan_limits_mean": _safe_mean(blocked_plan),
        "blocked_by_direction_mean": _safe_mean(blocked_direction),
        "execution_rate_mean": _safe_mean(execution_rates),
        "risk_usage_vs_return_corr": corr,
        "brake_distribution": brake_counts,
        "timeframe_execution_rate": timeframe_exec_rate,
        "hour_execution_rate": hour_exec_rate,
        "flattening_pct_mean": _safe_mean(flatten_pcts),
        "fees_pct_mean": _safe_mean(fee_pcts),
        "flatten_trade_days_pct": (flatten_days / days * 100.0) if days else 0.0,
        "trigger_quality": _finalize_quality(trigger_quality),
        "timeframe_quality": _finalize_quality(timeframe_quality),
        "hour_quality": _finalize_quality(hour_quality),
        "archetype_quality": _finalize_quality(archetype_quality),
        "archetype_hour_quality": _finalize_quality(archetype_hour_quality),
        "factor_exposures": factor_exposures,
        "block_totals": block_totals,
        "llm_fallback_days": llm_fallback_days,
        "llm_plan_fail_days": llm_plan_fail_days,
        "llm_rpr_source_days": rpr_source_days,
        "llm_rpr_filtered_days": rpr_filtered_days,
    }
    if risk_profile_snapshot:
        summary["risk_profile"] = risk_profile_snapshot

    if baseline_summary:
        summary["rpr_comparison"] = _build_rpr_comparison(
            summary,
            baseline_summary,
        )
    else:
        summary["rpr_comparison"] = {}
    return summary


def write_run_summary(
    daily_reports: Sequence[Mapping[str, Any]],
    target_path: Path,
    baseline_summary_path: Path | None = None,
    baseline_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and write a run summary JSON next to daily reports."""

    if baseline_summary is None and baseline_summary_path and baseline_summary_path.exists():
        try:
            baseline_summary = json.loads(baseline_summary_path.read_text())
        except json.JSONDecodeError:
            baseline_summary = None
    summary = build_run_summary(daily_reports, baseline_summary=baseline_summary)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(summary, indent=2))
    return summary
