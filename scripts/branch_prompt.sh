#!/usr/bin/env bash
set -euo pipefail

REPO_PATH="${REPO_PATH:-/home/getzinmw/crypto-trading-agents}"

usage() {
  cat <<USAGE
Usage:
  $0 RUNBOOK_PATH BRANCH_NAME PRIORITY AGENT_NAME
  $0 --preset NAME --agent AGENT_NAME [--priority P0] [--runbook PATH] [--branch NAME]
  $0 --list

Examples:
  $0 docs/branching/comp-audit-risk-core.md comp-audit-risk-core P0 agent-1
  $0 --preset comp-audit-risk-core --agent agent-1
  $0 --preset comp-audit-risk-core --agent agent-1 --priority P1

USAGE
}

PRESET_ORDER=(
  "comp-audit-risk-core"
  "comp-audit-trigger-cadence"
  "comp-audit-indicators-prompts"
  "comp-audit-metrics-parity"
  "comp-audit-ui-trade-stats"
  "aws-deploy"
  "multi-wallet"
  "policy-pivot-phase0"
  "judge-unification"
  "strategist-tool-loop"
  "scalper-mode"
  "ui-unification"
)

declare -A PRESETS
PRESETS["comp-audit-risk-core"]="docs/branching/comp-audit-risk-core.md|comp-audit-risk-core|P0"
PRESETS["comp-audit-trigger-cadence"]="docs/branching/comp-audit-trigger-cadence.md|comp-audit-trigger-cadence|P0"
PRESETS["comp-audit-indicators-prompts"]="docs/branching/comp-audit-indicators-prompts.md|comp-audit-indicators-prompts|P1"
PRESETS["comp-audit-metrics-parity"]="docs/branching/comp-audit-metrics-parity.md|comp-audit-metrics-parity|P1"
PRESETS["comp-audit-ui-trade-stats"]="docs/branching/comp-audit-ui-trade-stats.md|comp-audit-ui-trade-stats|P1"
PRESETS["aws-deploy"]="docs/branching/aws-deploy.md|aws-deploy|P1"
PRESETS["multi-wallet"]="docs/branching/multi-wallet.md|multi-wallet|P1"
PRESETS["policy-pivot-phase0"]="docs/branching/later/policy-pivot-phase0.md|policy-pivot-phase0|P2"
PRESETS["judge-unification"]="docs/branching/later/judge-unification.md|judge-unification|P2"
PRESETS["strategist-tool-loop"]="docs/branching/later/strategist-tool-loop.md|strategist-tool-loop|P2"
PRESETS["scalper-mode"]="docs/branching/later/scalper-mode.md|scalper-mode|P2"
PRESETS["ui-unification"]="docs/branching/later/ui-unification.md|ui-unification|P3"

list_presets() {
  printf "Available presets:\n"
  for name in "${PRESET_ORDER[@]}"; do
    entry="${PRESETS[$name]}"
    IFS='|' read -r path branch priority <<<"$entry"
    printf "- %s -> %s (%s, %s)\n" "$name" "$path" "$branch" "$priority"
  done
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

RUNBOOK_PATH=""
BRANCH_NAME=""
PRIORITY=""
AGENT_NAME=""

case "$1" in
  --list|-l)
    list_presets
    exit 0
    ;;
  --preset|-p)
    if [[ $# -lt 3 ]]; then
      usage
      exit 1
    fi
    PRESET_NAME="$2"
    shift 2

    if [[ -z "${PRESETS[$PRESET_NAME]+x}" ]]; then
      echo "Unknown preset: $PRESET_NAME"
      list_presets
      exit 1
    fi

    IFS='|' read -r RUNBOOK_PATH BRANCH_NAME PRIORITY <<<"${PRESETS[$PRESET_NAME]}"

    while [[ $# -gt 0 ]]; do
      case "$1" in
        --agent|-a)
          AGENT_NAME="$2"
          shift 2
          ;;
        --priority)
          PRIORITY="$2"
          shift 2
          ;;
        --runbook)
          RUNBOOK_PATH="$2"
          shift 2
          ;;
        --branch)
          BRANCH_NAME="$2"
          shift 2
          ;;
        *)
          echo "Unknown option: $1"
          usage
          exit 1
          ;;
      esac
    done

    if [[ -z "$AGENT_NAME" ]]; then
      echo "--agent is required when using --preset"
      usage
      exit 1
    fi
    ;;
  *)
    if [[ $# -lt 4 ]]; then
      usage
      exit 1
    fi
    RUNBOOK_PATH="$1"
    BRANCH_NAME="$2"
    PRIORITY="$3"
    AGENT_NAME="$4"
    ;;
esac

PROMPT=$(cat <<EOF2
You are an implementation agent on repo: ${REPO_PATH}.
Follow instructions in CLAUDE.md and docs/CODEX_HOWTO.md.

Assignment
- Runbook: ${RUNBOOK_PATH}
- Branch: ${BRANCH_NAME}
- Agent: ${AGENT_NAME}
- Priority: ${PRIORITY}

Required behavior
- Read the runbook first; follow its scope, key files, acceptance criteria, and Git Workflow exactly.
- Do not widen scope without approval.
- Do not commit until:
  1) tests listed in the runbook pass or the user provides explicit test output, AND
  2) Human Verification Evidence is filled in (when required).
- If a test can’t be run, ask the human for pasted output and record it in the runbook.
- Keep changes isolated to the runbook’s files; avoid conflicts with other branches.

Deliverables
- Implement the runbook scope.
- Run the runbook tests and paste output into the runbook’s Test Evidence section.
- Request human verification when required and paste the human feedback into the runbook’s Human Verification Evidence section.
- Commit using the message specified in the runbook.

Report format
- What changed (files)
- Test results (or reason not run)
- Evidence added (test + human verification)
- Next steps / blockers
EOF2
)

copy_to_clipboard() {
  if command -v pbcopy >/dev/null 2>&1; then
    printf "%s" "$PROMPT" | pbcopy
    return 0
  fi
  if command -v xclip >/dev/null 2>&1; then
    printf "%s" "$PROMPT" | xclip -selection clipboard
    return 0
  fi
  if command -v xsel >/dev/null 2>&1; then
    printf "%s" "$PROMPT" | xsel --clipboard --input
    return 0
  fi
  if command -v clip >/dev/null 2>&1; then
    printf "%s" "$PROMPT" | clip
    return 0
  fi
  return 1
}

if copy_to_clipboard; then
  echo "Prompt copied to clipboard."
else
  echo "No clipboard tool found. Prompt printed below."
fi

printf "\n%s\n" "$PROMPT"
