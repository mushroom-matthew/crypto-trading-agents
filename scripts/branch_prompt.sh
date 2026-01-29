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
  $0 docs/branching/01-strategist-simplification.md strategist-simplification P01 agent-1
  $0 --preset strategist-simplification --agent agent-1
  $0 --preset strategist-simplification --agent agent-1 --priority P01

USAGE
}

PRESET_ORDER=(
  # Priority queue (numbered)
  "strategist-simplification"
  "comp-audit-ui-trade-stats"
  "emergency-exit-same-bar-dedup"
  "emergency-exit-hold-cooldown"
  "emergency-exit-bypass-override"
  "emergency-exit-edge-cases"
  "aws-deploy"
  "multi-wallet"
  "runbook-architecture-wiring"
  "runbook-learning-book"
  "runbook-experiment-specs"
  "runbook-no-learn-zones-and-killswitches"
  # Backlog (_)
  "emergency-exit-judge-loop-design"
  "synthetic-data-testing"
  "comp-audit-risk-followups"
  "policy-pivot-phase0"
  "judge-unification"
  "strategist-tool-loop"
  "scalper-mode"
  "ui-unification"
  "ui-config-cleanup"
  # Completed (X)
  "comp-audit-risk-core"
  "comp-audit-trigger-cadence"
  "comp-audit-indicators-prompts"
  "comp-audit-metrics-parity"
  "judge-feedback-enforcement"
)

declare -A PRESETS
PRESETS["strategist-simplification"]="docs/branching/01-strategist-simplification.md|strategist-simplification|P01"
PRESETS["comp-audit-ui-trade-stats"]="docs/branching/02-comp-audit-ui-trade-stats.md|comp-audit-ui-trade-stats|P02"
PRESETS["emergency-exit-same-bar-dedup"]="docs/branching/03-emergency-exit-runbook-same-bar-dedup.md|emergency-exit-runbook-same-bar-dedup|P03"
PRESETS["emergency-exit-hold-cooldown"]="docs/branching/04-emergency-exit-runbook-hold-cooldown.md|emergency-exit-runbook-hold-cooldown|P04"
PRESETS["emergency-exit-bypass-override"]="docs/branching/05-emergency-exit-runbook-bypass-override.md|emergency-exit-runbook-bypass-override|P05"
PRESETS["emergency-exit-edge-cases"]="docs/branching/06-emergency-exit-runbook-edge-cases.md|emergency-exit-runbook-edge-cases|P06"
PRESETS["aws-deploy"]="docs/branching/07-aws-deploy.md|aws-deploy|P07"
PRESETS["multi-wallet"]="docs/branching/08-multi-wallet.md|multi-wallet|P08"
PRESETS["runbook-architecture-wiring"]="docs/branching/09-runbook-architecture-wiring.md|runbook-architecture-wiring|P09"
PRESETS["runbook-learning-book"]="docs/branching/10-runbook-learning-book.md|runbook-learning-book|P10"
PRESETS["runbook-experiment-specs"]="docs/branching/11-runbook-experiment-specs.md|runbook-experiment-specs|P11"
PRESETS["runbook-no-learn-zones-and-killswitches"]="docs/branching/12-runbook-no-learn-zones-and-killswitches.md|runbook-no-learn-zones-and-killswitches|P12"
PRESETS["emergency-exit-judge-loop-design"]="docs/branching/_emergency-exit-runbook-judge-loop-design.md|emergency-exit-runbook-judge-loop-design|B"
PRESETS["synthetic-data-testing"]="docs/branching/_synthetic-data-testing.md|synthetic-data-testing|B"
PRESETS["comp-audit-risk-followups"]="docs/branching/later/_comp-audit-risk-followups.md|comp-audit-risk-followups|B"
PRESETS["policy-pivot-phase0"]="docs/branching/later/_policy-pivot-phase0.md|policy-pivot-phase0|B"
PRESETS["judge-unification"]="docs/branching/later/_judge-unification.md|judge-unification|B"
PRESETS["strategist-tool-loop"]="docs/branching/later/_strategist-tool-loop.md|strategist-tool-loop|B"
PRESETS["scalper-mode"]="docs/branching/later/_scalper-mode.md|scalper-mode|B"
PRESETS["ui-unification"]="docs/branching/later/_ui-unification.md|ui-unification|B"
PRESETS["ui-config-cleanup"]="docs/branching/later/_ui-config-cleanup.md|ui-config-cleanup|B"
PRESETS["comp-audit-risk-core"]="docs/branching/X-comp-audit-risk-core.md|comp-audit-risk-core|X"
PRESETS["comp-audit-trigger-cadence"]="docs/branching/X-comp-audit-trigger-cadence.md|comp-audit-trigger-cadence|X"
PRESETS["comp-audit-indicators-prompts"]="docs/branching/X-comp-audit-indicators-prompts.md|comp-audit-indicators-prompts|X"
PRESETS["comp-audit-metrics-parity"]="docs/branching/X-comp-audit-metrics-parity.md|comp-audit-metrics-parity|X"
PRESETS["judge-feedback-enforcement"]="docs/branching/X-judge-feedback-enforcement.md|judge-feedback-enforcement|X"

status_from_path() {
  local base
  base="$(basename "$1")"
  if [[ "$base" == X-* ]]; then
    printf "%s" "complete"
    return
  fi
  if [[ "$base" == _* ]]; then
    printf "%s" "backlog"
    return
  fi
  if [[ "$base" =~ ^[0-9][0-9]- ]]; then
    printf "%s" "queue"
    return
  fi
  printf "%s" "unspecified"
}

list_presets() {
  printf -- "Available presets (queue order):\n"
  for name in "${PRESET_ORDER[@]}"; do
    entry="${PRESETS[$name]}"
    IFS='|' read -r path branch priority <<<"$entry"
    status="$(status_from_path "$path")"
    printf -- "- %s -> %s (%s, %s, %s)\n" "$name" "$path" "$branch" "$priority" "$status"
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
- Use the Worktree Setup section if working in parallel on the same machine.
- Do not widen scope without approval.
- Do not commit until:
  1) tests listed in the runbook pass or the user provides explicit test output, AND
  2) Human Verification Evidence is filled in (when required).
- Update the runbook Change Log section with a dated summary of changes and files touched.
- If a test can’t be run, ask the human for pasted output and record it in the runbook.
- Keep changes isolated to the runbook’s files; avoid conflicts with other branches.
- Queue guidance: work in numeric runbook order when possible, and rename the runbook with an X- prefix when complete.

Deliverables
- Implement the runbook scope.
- Run the runbook tests and paste output into the runbook’s Test Evidence section.
- Request human verification when required and paste the human feedback into the runbook’s Human Verification Evidence section.
- Update the runbook Change Log section.
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
