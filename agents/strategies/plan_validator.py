"""Compile-time validation for LLM-generated trigger plans.

Detects structurally hazardous patterns — most critically, cross-timeframe ATR
tautologies in emergency_exit exit_rules that cause the trigger to fire on every
bar, creating a constant buy-exit churn loop.

The key insight: ATR naturally scales as sqrt(T_high / T_low) across timeframes.
A comparison like ``htf_daily_atr > 3 * atr_14`` (daily vs 1h, sqrt(24)≈4.9x)
is nearly always True because the daily ATR is naturally ~5x the hourly ATR.

Usage::

    result = validate_trigger_plan(plan_dict, base_tf_minutes=60)
    if not result.is_valid:
        repair_text = result.repair_prompt()
        # Replan with repair_text injected into the LLM prompt
    if result.is_valid:
        pass  # safe to proceed
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

# ── Timeframe string → minutes ───────────────────────────────────────────────────

_TF_TO_MINUTES: dict[str, int] = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "4h": 240, "6h": 360, "12h": 720,
    "1d": 1440, "1D": 1440, "1w": 10080,
}


def _parse_tf_minutes(timeframe: str) -> int:
    """Convert a timeframe string to minutes. Falls back to 60 (1h) for unknowns."""
    return _TF_TO_MINUTES.get(timeframe.strip(), 60)


# ── ATR identifier → approximate timeframe mapping ──────────────────────────────
# "base" means: resolve to the trigger's own timeframe at call-time.

_ATR_PATTERNS: list[tuple[re.Pattern[str], int | str]] = [
    (re.compile(r"\bhtf_daily_atr\b"), 1440),
    (re.compile(r"\bdaily_atr\b"), 1440),
    (re.compile(r"\batr_1d\b"), 1440),
    (re.compile(r"\btf_1[dD]_atr\b"), 1440),
    (re.compile(r"\btf_4[hH]_atr\b"), 240),
    (re.compile(r"\btf_240m_atr\b"), 240),
    (re.compile(r"\btf_1[hH]_atr\b"), 60),
    (re.compile(r"\btf_60m_atr\b"), 60),
    (re.compile(r"\btf_30m_atr\b"), 30),
    (re.compile(r"\btf_15m_atr\b"), 15),
    (re.compile(r"\btf_5m_atr\b"), 5),
    (re.compile(r"\btf_3m_atr\b"), 3),
    (re.compile(r"\btf_1m_atr\b"), 1),
    # Bare atr_N identifiers (e.g. atr_14, atr_21) belong to the trigger's own TF
    (re.compile(r"\batr_\d+\b"), "base"),
    # Bare 'atr' alias
    (re.compile(r"\batr\b"), "base"),
]


def _atr_minutes(identifier: str, base_minutes: int) -> int | None:
    """Return the timeframe in minutes for a known ATR identifier, else None."""
    ident = identifier.strip()
    for pattern, tf in _ATR_PATTERNS:
        if pattern.fullmatch(ident):
            return base_minutes if tf == "base" else int(tf)  # type: ignore[arg-type]
    return None


# ── Minimum safe multiplier k ────────────────────────────────────────────────────

# ATR scales as sqrt(T_high / T_low) in a random walk.  BTC has fat tails and
# intraday clustering, but the sqrt ratio is a reliable floor.  We require the
# multiplier k to exceed the natural ratio by MARGIN so that the condition
# represents genuinely extreme volatility, not the average daily/hourly spread.
_TAUTOLOGY_MARGIN = 1.3


def _k_min(high_minutes: int, low_minutes: int) -> float:
    """Minimum multiplier k for HIGH_TF_ATR > k * LOW_TF_ATR to be non-tautological."""
    if low_minutes <= 0 or high_minutes <= low_minutes:
        return 1.0
    return math.sqrt(high_minutes / low_minutes) * _TAUTOLOGY_MARGIN


# ── Regex scanners for comparison patterns ───────────────────────────────────────

# Pattern 1: IDENT OP k * IDENT  — e.g. htf_daily_atr > 3 * atr_14
_CMP_K = re.compile(r"(\w+)\s*([><])\s*(\d+(?:\.\d+)?)\s*\*\s*(\w+)")

# Pattern 2: IDENT OP IDENT / k  — e.g. atr_14 < htf_daily_atr / 3
_DIV_K = re.compile(r"(\w+)\s*([><])\s*(\w+)\s*/\s*(\d+(?:\.\d+)?)")

# Pattern 3: IDENT OP IDENT (implicit k=1) — e.g. htf_daily_atr > atr_14
# The negative lookahead prevents matching the LHS of pattern 1/2 (e.g. "atr_14 <
# htf_daily_atr" inside "atr_14 < htf_daily_atr / 3") by rejecting when the
# matched RHS identifier is immediately followed by * or /.
_DIRECT = re.compile(r"(\w+)\s*([><])\s*(\w+)(?!\s*[*/])")


@dataclass
class TautologyMatch:
    """Describes a detected cross-timeframe ATR tautology in a rule expression."""

    high_ident: str      # The higher-timeframe ATR identifier
    high_minutes: int    # Its approximate timeframe in minutes
    low_ident: str       # The lower-timeframe ATR identifier
    low_minutes: int     # Its approximate timeframe in minutes
    k: float             # Actual multiplier used in the rule
    k_min: float         # Minimum k required to avoid tautology
    fragment: str        # The matched substring of the rule expression


def _scan_atr_tautologies(rule: str, base_minutes: int) -> list[TautologyMatch]:
    """Scan a rule expression for cross-timeframe ATR comparisons with k < k_min."""
    results: list[TautologyMatch] = []
    seen: set[str] = set()

    def _check(lhs_id: str, op: str, k: float, rhs_id: str, fragment: str) -> None:
        lhs_m = _atr_minutes(lhs_id, base_minutes)
        rhs_m = _atr_minutes(rhs_id, base_minutes)
        if lhs_m is None or rhs_m is None or lhs_m == rhs_m:
            return
        # Normalise: high_minutes > low_minutes, op must point from high to low
        if op == ">" and lhs_m > rhs_m:
            high_id, high_m, low_id, low_m = lhs_id, lhs_m, rhs_id, rhs_m
        elif op == "<" and rhs_m > lhs_m:
            high_id, high_m, low_id, low_m = rhs_id, rhs_m, lhs_id, lhs_m
        else:
            return  # Direction not comparing high > low → not a tautology pattern

        km = _k_min(high_m, low_m)
        if k < km and fragment not in seen:
            seen.add(fragment)
            results.append(TautologyMatch(
                high_ident=high_id, high_minutes=high_m,
                low_ident=low_id, low_minutes=low_m,
                k=k, k_min=km, fragment=fragment,
            ))

    # Pattern 1: IDENT > k * IDENT
    for m in _CMP_K.finditer(rule):
        _check(m.group(1), m.group(2), float(m.group(3)), m.group(4), m.group(0))

    # Pattern 2: IDENT OP IDENT / k  → flip: rhs > k * lhs
    for m in _DIV_K.finditer(rule):
        flipped = "<" if m.group(2) == ">" else ">"
        _check(m.group(3), flipped, float(m.group(4)), m.group(1), m.group(0))

    # Pattern 3: IDENT OP IDENT (k=1 implicit)
    for m in _DIRECT.finditer(rule):
        _check(m.group(1), m.group(2), 1.0, m.group(3), m.group(0))

    return results


# ── Public result types ──────────────────────────────────────────────────────────

@dataclass
class TriggerError:
    """A validation error or warning for a single trigger."""

    trigger_id: str
    category: str
    code: str       # e.g. "EMERGENCY_EXIT_TAUTOLOGY", "MISSING_EXIT_RULE"
    message: str    # Human-readable explanation
    raw_rule: str   # The rule expression that triggered the error
    tautology: TautologyMatch | None = None


@dataclass
class PlanValidationResult:
    """Outcome of validate_trigger_plan()."""

    hard_errors: list[TriggerError] = field(default_factory=list)
    warnings: list[TriggerError] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.hard_errors) == 0

    def summary(self) -> str:
        lines: list[str] = []
        for e in self.hard_errors:
            lines.append(f"[ERROR][{e.trigger_id}] {e.code}: {e.message}")
        for w in self.warnings:
            lines.append(f"[WARN][{w.trigger_id}] {w.code}: {w.message}")
        return "\n".join(lines) if lines else "OK"

    def repair_prompt(self) -> str:
        """Return a repair block to inject into the next LLM plan-generation prompt."""
        if not self.hard_errors:
            return ""
        lines = [
            "## PREVIOUS PLAN REJECTED — YOU MUST FIX ALL ERRORS BELOW",
            "Do NOT reuse the flagged expression patterns in any trigger.",
            "",
        ]
        for e in self.hard_errors:
            lines.append(f"TRIGGER: {e.trigger_id}  (category={e.category})")
            lines.append(f"RULE:    {e.raw_rule}")
            lines.append(f"ERROR:   {e.message}")
            if e.tautology:
                t = e.tautology
                nat = math.sqrt(t.high_minutes / t.low_minutes)
                safe_k = math.ceil(t.k_min * 10) / 10
                lines.append(
                    f"  Natural ATR scaling {t.high_ident}/{t.low_ident} = "
                    f"sqrt({t.high_minutes}/{t.low_minutes}) ≈ {nat:.2f}x. "
                    f"Your k={t.k:.1f} < required k≥{t.k_min:.1f}. "
                    f"FIX OPTION A: use 'vol_state == \"extreme\"' instead. "
                    f"FIX OPTION B: raise k to ≥{safe_k:.1f}."
                )
            lines.append("")
        return "\n".join(lines)


# ── Public API ───────────────────────────────────────────────────────────────────

def validate_trigger_plan(
    plan_dict: dict[str, Any],
    base_tf_minutes: int = 60,
) -> PlanValidationResult:
    """Validate a raw plan dict from generate_strategy_plan_activity.

    Currently validates:
    - emergency_exit triggers must have a non-empty exit_rule
    - emergency_exit exit_rules must not contain cross-timeframe ATR tautologies

    Args:
        plan_dict: Raw plan dictionary with a "triggers" list.
        base_tf_minutes: Base bar timeframe in minutes (default 60 = 1h).
            Used to resolve bare ATR identifiers like ``atr_14``.

    Returns:
        PlanValidationResult with hard_errors (plan rejected) and warnings.
    """
    result = PlanValidationResult()

    for trigger in plan_dict.get("triggers", []):
        tid = trigger.get("id", "<unknown>")
        cat = trigger.get("category", "")

        if cat != "emergency_exit":
            continue

        timeframe = trigger.get("timeframe", "1h") or "1h"
        tf_minutes = _parse_tf_minutes(timeframe)

        exit_rule = (trigger.get("exit_rule") or "").strip()
        if not exit_rule:
            result.hard_errors.append(TriggerError(
                trigger_id=tid, category=cat,
                code="MISSING_EXIT_RULE",
                message="emergency_exit trigger must have a non-empty exit_rule.",
                raw_rule="",
            ))
            continue

        for taut in _scan_atr_tautologies(exit_rule, tf_minutes):
            nat = math.sqrt(taut.high_minutes / taut.low_minutes)
            result.hard_errors.append(TriggerError(
                trigger_id=tid, category=cat,
                code="EMERGENCY_EXIT_TAUTOLOGY",
                message=(
                    f"'{taut.high_ident}' ({taut.high_minutes}min) vs "
                    f"'{taut.low_ident}' ({taut.low_minutes}min): "
                    f"natural ATR ratio ≈{nat:.2f}x; "
                    f"k={taut.k:.1f} < required k≥{taut.k_min:.1f}. "
                    f"This condition is true on virtually every bar."
                ),
                raw_rule=exit_rule,
                tautology=taut,
            ))

    return result


def check_exit_rule_for_tautology(
    exit_rule: str,
    trigger_timeframe: str = "1h",
) -> list[TautologyMatch]:
    """Lightweight runtime check for the trigger engine failsafe.

    Returns the list of detected tautologies. An empty list means the rule is
    structurally sound (no detected cross-TF ATR tautologies).

    This function is designed to be called inside trigger_engine.on_bar() for
    every emergency_exit trigger evaluation, so it must be fast.  The regex
    scan is O(len(exit_rule)) and negligible compared to bar processing.
    """
    base_minutes = _parse_tf_minutes(trigger_timeframe)
    return _scan_atr_tautologies(exit_rule, base_minutes)
