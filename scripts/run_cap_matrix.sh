#!/usr/bin/env bash
set -euo pipefail

# Helper to run the four-cap validation matrix on a fixed date slice.
# Usage: scripts/run_cap_matrix.sh [start_date] [end_date]
# Defaults: 2021-02-10 -> 2021-02-16

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
START_DATE="${1:-2021-02-10}"
END_DATE="${2:-2021-02-16}"
PAIR="${PAIR:-BTC-USD}"
OUTPUT_ROOT="${OUTPUT_ROOT:-.runs/cap_matrix}"
mkdir -p "${OUTPUT_ROOT}"

run_cfg() {
  local label="$1"
  local strict_caps="$2"
  local trade_cap="$3"
  local trigger_cap="$4"
  local multipliers_flag="$5" # e.g., "--session-trade-multipliers \"0-4:1.5,4-24:0.75\"" or empty
  local run_id="cap-${label}"
  echo ">>> Running ${label} (strict_fixed_caps=${strict_caps}, trades=${trade_cap}, triggers=${trigger_cap})"
  STRICT_FIXED_CAPS="${strict_caps}" \
  STRATEGIST_PLAN_DEFAULT_MAX_TRADES="${trade_cap}" \
  STRATEGIST_PLAN_DEFAULT_MAX_TRIGGERS_PER_SYMBOL="${trigger_cap}" \
  ARCHETYPE_MULTIPLIERS_PATH="" \
  uv run python -m backtesting.cli \
    --llm-strategist enabled \
    --pairs "${PAIR}" \
    --start "${START_DATE}" \
    --end "${END_DATE}" \
    --llm-run-id "${run_id}" \
    --llm-cache-dir "${OUTPUT_ROOT}" \
    --log-level INFO \
    --max-daily-risk-budget-pct 10 \
    --max-position-risk-pct 1.0 \
    ${multipliers_flag}
  echo "Summary at ${OUTPUT_ROOT}/${run_id}/run_summary.json"
}

# A) Legacy, no multipliers (modest caps)
run_cfg "legacy_nomult" "false" "15" "25" ""

# B) Legacy, multipliers (set as needed)
# Example session multipliers; adjust or clear as desired.
run_cfg "legacy_mult" "false" "15" "25" "--session-trade-multipliers \"0-4:1.5,4-24:0.75\""

# C) Fixed, no multipliers (generous caps)
run_cfg "fixed_nomult" "true" "30" "40" ""

# D) Fixed, multipliers
run_cfg "fixed_mult" "true" "30" "40" "--session-trade-multipliers \"0-4:1.5,4-24:0.75\""

echo "Runs complete. Inspect summaries under ${OUTPUT_ROOT}/cap-*/run_summary.json and daily reports in the same folders."
