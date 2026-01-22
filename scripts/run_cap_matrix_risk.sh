#!/usr/bin/env bash
set -euo pipefail

# Helper to run a risk-geometry matrix over a fixed date slice.
# Default slice: 2021-02-10 -> 2021-02-16
# Caps fixed at trades=30, triggers=40, STRICT_FIXED_CAPS=true.
# Sweeps risk budgets and per-trade risk with/without session multipliers.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
START_DATE="${1:-2021-02-10}"
END_DATE="${2:-2021-02-16}"
PAIR="${PAIR:-BTC-USD}"
OUTPUT_ROOT="${OUTPUT_ROOT:-.runs/cap_matrix_risk}"
mkdir -p "${OUTPUT_ROOT}"

run_cfg() {
  local label="$1"
  local budget_pct="$2"
  local per_trade_pct="$3"
  local multipliers_flag="$4" # e.g., "--session-trade-multipliers \"0-4:1.5,4-24:0.75\"" or empty
  local run_id="risk-${label}"
  echo ">>> Running ${label} (budget=${budget_pct}%, per-trade=${per_trade_pct}%)"
  STRICT_FIXED_CAPS="true" \
  STRATEGIST_PLAN_DEFAULT_MAX_TRADES="30" \
  STRATEGIST_PLAN_DEFAULT_MAX_TRIGGERS_PER_SYMBOL="40" \
  ARCHETYPE_MULTIPLIERS_PATH="" \
  uv run python -m backtesting.cli \
    --llm-strategist enabled \
    --pairs "${PAIR}" \
    --start "${START_DATE}" \
    --end "${END_DATE}" \
    --llm-run-id "${run_id}" \
    --llm-cache-dir "${OUTPUT_ROOT}" \
    --log-level INFO \
    --max-daily-risk-budget-pct "${budget_pct}" \
    --max-position-risk-pct "${per_trade_pct}" \
    ${multipliers_flag}
  echo "Summary at ${OUTPUT_ROOT}/${run_id}/run_summary.json"
}

# Define risk regimes (adjust as needed)
# R1: budget 10%, per-trade 0.30%
run_cfg "r1_nomult" "10.0" "0.30" ""
run_cfg "r1_mult"   "10.0" "0.30" "--session-trade-multipliers \"0-4:1.5,4-24:0.75\""

# R2: budget 10%, per-trade 0.40%
run_cfg "r2_nomult" "10.0" "0.40" ""
run_cfg "r2_mult"   "10.0" "0.40" "--session-trade-multipliers \"0-4:1.5,4-24:0.75\""

# R3: budget 12%, per-trade 0.40%
run_cfg "r3_nomult" "12.0" "0.40" ""
run_cfg "r3_mult"   "12.0" "0.40" "--session-trade-multipliers \"0-4:1.5,4-24:0.75\""

# R4: budget 15%, per-trade 0.50% (stretch case)
run_cfg "r4_nomult" "15.0" "0.50" ""
run_cfg "r4_mult"   "15.0" "0.50" "--session-trade-multipliers \"0-4:1.5,4-24:0.75\""

echo "Runs complete. Inspect summaries under ${OUTPUT_ROOT}/risk-*/run_summary.json."
