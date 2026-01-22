#!/usr/bin/env bash
set -euo pipefail

CACHE_DIR="${1:-.cache/strategy_plans}"
OUTPUT="${2:-strategy_export.json}"
# Optional glob-style filter for run ids / cache directories (e.g., "p1*" or "*202102*")
RUN_FILTER="${3:-*}"

# Pick a Python interpreter that exists (prefer python3, fall back to python)
if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "No python interpreter found in PATH" >&2
  exit 1
fi

"$PYTHON_BIN" - "$CACHE_DIR" "$OUTPUT" "$RUN_FILTER" <<'PY'
import json
import fnmatch
import sys
from datetime import datetime, timezone
from pathlib import Path

cache_dir = Path(sys.argv[1]).expanduser().resolve()
output_path = Path(sys.argv[2]).expanduser().resolve()
run_filter = sys.argv[3]

if not cache_dir.exists():
    raise SystemExit(f"Cache directory not found: {cache_dir}")

runs_to_process: list[Path] = []
if (cache_dir / "daily_reports").exists():
    if fnmatch.fnmatch(cache_dir.name, run_filter):
        runs_to_process.append(cache_dir)
else:
    for p in sorted(cache_dir.iterdir()):
        if not fnmatch.fnmatch(p.name, run_filter):
            continue
        # Allow either directories (current pattern) or files ending in .cache
        if p.is_dir() or p.suffix == ".cache":
            runs_to_process.append(p)

def _compress(obj):
    """
    Trim bulky sections so compiled report stays light:
    - For known noisy keys, keep count and small sample.
    - For large generic lists, keep a small head/tail sample.
    """
    noisy_keys = {"blocked_calls", "blocked_trades", "blocked", "trades", "executions", "tool_calls", "llm_calls"}
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in noisy_keys and isinstance(v, list):
                out[k] = {"count": len(v), "sample": v[:3]}
            else:
                out[k] = _compress(v)
        return out
    if isinstance(obj, list):
        # avoid blowing up the export: keep small sample
        if len(obj) > 50:
            return {"count": len(obj), "head": [_compress(x) for x in obj[:3]], "tail": [_compress(x) for x in obj[-3:]]}
        return [_compress(x) for x in obj]
    return obj

runs: list[dict] = []
for run_dir in runs_to_process:
    # Handle raw .cache files that contain a single json payload
    if run_dir.is_file():
        try:
            content = json.loads(run_dir.read_text())
        except json.JSONDecodeError:
            content = {"error": "invalid_json"}
        runs.append(
            {
                "run_id": run_dir.stem,
                "plans": [],
                "daily_reports": [],
                "run_summary": {"path": str(run_dir.name), "content": _compress(content)},
            }
        )
        continue

    run_id = run_dir.name if run_dir != cache_dir else cache_dir.name
    run_data: dict = {"run_id": run_id, "plans": [], "daily_reports": []}
    run_summary_path = run_dir / "run_summary.json"
    if run_summary_path.exists():
        try:
            summary_content = json.loads(run_summary_path.read_text())
            run_data["run_summary"] = {
                "path": str(run_summary_path.relative_to(cache_dir if cache_dir != run_dir else run_dir)),
                "content": _compress(summary_content),
            }
        except json.JSONDecodeError:
            run_data["run_summary"] = {
                "path": str(run_summary_path.relative_to(cache_dir if cache_dir != run_dir else run_dir)),
                "error": "invalid_json",
            }
    daily_dir = run_dir / "daily_reports"
    if daily_dir.exists():
        for daily_file in sorted(daily_dir.glob("*.json")):
            try:
                run_data["daily_reports"].append(
                    {
                        "path": str(daily_file.relative_to(cache_dir if cache_dir != run_dir else run_dir)),
                        "content": _compress(json.loads(daily_file.read_text())),
                    }
                )
            except json.JSONDecodeError:
                run_data["daily_reports"].append(
                    {"path": str(daily_file.relative_to(cache_dir if cache_dir != run_dir else run_dir)), "error": "invalid_json"}
                )

    for plan_file in sorted(run_dir.rglob("*.json")):
        try:
            if plan_file.is_relative_to(daily_dir):
                continue
        except AttributeError:
            if daily_dir in plan_file.parents:
                continue
        try:
            plan_data = json.loads(plan_file.read_text())
        except json.JSONDecodeError:
            plan_data = {"error": "invalid_json"}
        rel_base = cache_dir if cache_dir != run_dir else run_dir
        rel_path = str(plan_file.relative_to(rel_base))
        run_data["plans"].append({"path": rel_path, "content": _compress(plan_data)})

    runs.append(run_data)

payload = {
    "compiled_at": datetime.now(timezone.utc).isoformat(),
    "cache_dir": str(cache_dir),
    "run_count": len(runs),
    "runs": runs,
}

output_path.write_text(json.dumps(payload, indent=2))
print(f"Wrote {output_path} ({len(runs)} runs aggregated)")
PY
