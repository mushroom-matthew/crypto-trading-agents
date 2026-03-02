#!/usr/bin/env python3
"""Audit which Phase 8+ services are wired into the live execution paths.

Run from repo root:
    uv run python scripts/check_wiring.py

Exit code 0 = all targets wired. Exit code 1 = gaps found.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

# Each entry: (label, regex_to_search, {short_name: relative_path})
TARGETS: list[tuple[str, str, dict[str, str]]] = [
    (
        "PolicyLoopGate",
        r"\bPolicyLoopGate\b",
        {
            "paper": "tools/paper_trading.py",
            "backtest": "backtesting/llm_strategist_runner.py",
        },
    ),
    (
        "RegimeTransitionDetector",
        r"\bRegimeTransitionDetector\b",
        {
            "paper": "tools/paper_trading.py",
            "backtest": "backtesting/llm_strategist_runner.py",
        },
    ),
    (
        "PlaybookRegistry.list_eligible",
        r"\blist_eligible\b",
        {
            "plan_provider": "agents/strategies/plan_provider.py",
            "llm_client": "agents/strategies/llm_client.py",
        },
    ),
    (
        "MemoryRetrievalService",
        r"\bMemoryRetrievalService\b",
        {
            "paper": "tools/paper_trading.py",
            "backtest": "backtesting/llm_strategist_runner.py",
        },
    ),
    (
        "build_episode_record",
        r"\bbuild_episode_record\b",
        {
            "paper": "tools/paper_trading.py",
            "backtest": "backtesting/llm_strategist_runner.py",
        },
    ),
    (
        "EpisodeMemoryStore.persist_episode",
        r"\bpersist_episode\b",
        {
            "paper": "tools/paper_trading.py",
            "backtest": "backtesting/llm_strategist_runner.py",
        },
    ),
    (
        "JudgePlanValidationService",
        r"\bJudgePlanValidationService\b",
        {
            "paper": "tools/paper_trading.py",
            "backtest": "backtesting/llm_strategist_runner.py",
        },
    ),
    (
        "build_tick_snapshot",
        r"\bbuild_tick_snapshot\b",
        {
            "paper": "tools/paper_trading.py",
            "trigger_engine": "agents/strategies/trigger_engine.py",
        },
    ),
    (
        "StructuralTargetSelector",
        r"\bStructuralTargetSelector\b|select_stop_candidates|select_target_candidates",
        {
            "paper": "tools/paper_trading.py",
            "trigger_engine": "agents/strategies/trigger_engine.py",
        },
    ),
    (
        "SetupEventGenerator",
        r"\bSetupEventGenerator\b",
        {
            "paper": "tools/paper_trading.py",
            "backtest": "backtesting/llm_strategist_runner.py",
        },
    ),
    (
        "AdaptiveTradeManagement",
        r"\bAdaptiveTradeManagement\b",
        {
            "paper": "tools/paper_trading.py",
            "backtest": "backtesting/llm_strategist_runner.py",
        },
    ),
    (
        "PositionExitContract enforcement",
        r"\bPositionExitContract\b",
        {
            "paper": "tools/paper_trading.py",
            "trigger_engine": "agents/strategies/trigger_engine.py",
        },
    ),
    (
        "PolicyStateMachineRecord in SessionState",
        r"\bpolicy_state\b|\bpolicy_machine\b|\bstate_machine_record\b",
        {
            "session_state": "tools/paper_trading.py",
        },
    ),
    (
        "RegimeDetectorState in SessionState",
        r"\bregime_detector_state\b|\bregime_transition_state\b",
        {
            "session_state": "tools/paper_trading.py",
        },
    ),
    (
        "originating_plan_id in position_meta",
        r"\boriginating_plan_id\b",
        {
            "paper": "tools/paper_trading.py",
        },
    ),
]

LABEL_W = 42
COL_W = 16


def check(pattern: str, path: str) -> bool:
    full = ROOT / path
    if not full.exists():
        return False
    return bool(re.search(pattern, full.read_text()))


def main() -> int:
    gaps: list[str] = []

    # header
    all_cols: list[str] = []
    for _, _, paths in TARGETS:
        for k in paths:
            if k not in all_cols:
                all_cols.append(k)
    col_labels = all_cols[:5]  # show at most 5 columns

    header = f"{'Component':<{LABEL_W}}" + "".join(f"  {c:>{COL_W}}" for c in col_labels)
    print(header)
    print("-" * len(header))

    for label, pattern, paths in TARGETS:
        row = f"{label:<{LABEL_W}}"
        for col in col_labels:
            if col in paths:
                wired = check(pattern, paths[col])
                status = "✅ wired" if wired else "❌ MISSING"
                if not wired:
                    gaps.append(f"{label} → {paths[col]}")
                row += f"  {status:>{COL_W}}"
            else:
                row += f"  {'—':>{COL_W}}"
        print(row)

    print()
    if gaps:
        print(f"⚠️  {len(gaps)} wiring gap(s) found:")
        for g in gaps:
            print(f"   • {g}")
        print()
        print("Implement runbooks R61-R67 to close these gaps.")
        return 1
    else:
        print("✅ All targets wired.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
