# Legacy Surface (Quarantined)

This package tracks legacy/live-only modules that are **not** part of the agent stack. Access is opt-in via `TRADING_STACK=legacy_live` and should not be imported when running in agent mode.

## Contents (see manifest.yml)
- Legacy workflows and services that predate the agent stack.
- Temporary parking for code slated for deletion or migration.

## Usage
- Set `TRADING_STACK=legacy_live` to enable discovery by `worker/legacy_live_worker.py`.
- Agent mode (`TRADING_STACK=agent`) must not import anything under `legacy/` or `workflows/`.

## Sunset
- Each entry in `legacy/manifest.yml` must have an owner and deletion trigger/date.
