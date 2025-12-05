#!/usr/bin/env bash
set -euo pipefail

CACHE_DIR="${1:-.cache/strategy_plans}"
OUTPUT="${2:-strategy_export.json}"

python - "$CACHE_DIR" "$OUTPUT" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

cache_dir = Path(sys.argv[1]).expanduser().resolve()
output_path = Path(sys.argv[2]).expanduser().resolve()

if not cache_dir.exists():
    raise SystemExit(f"Cache directory not found: {cache_dir}")

runs_to_process: list[Path] = []
if (cache_dir / "daily_reports").exists():
    runs_to_process.append(cache_dir)
else:
    runs_to_process.extend(sorted([p for p in cache_dir.iterdir() if p.is_dir()]))

runs: list[dict] = []
for run_dir in runs_to_process:
    run_id = run_dir.name if run_dir != cache_dir else cache_dir.name
    run_data: dict = {"run_id": run_id, "plans": [], "daily_reports": []}
    run_summary_path = run_dir / "run_summary.json"
    if run_summary_path.exists():
        try:
            summary_content = json.loads(run_summary_path.read_text())
            run_data["run_summary"] = {
                "path": str(run_summary_path.relative_to(cache_dir if cache_dir != run_dir else run_dir)),
                "content": summary_content,
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
                        "content": json.loads(daily_file.read_text()),
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
        run_data["plans"].append({"path": rel_path, "content": plan_data})

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
