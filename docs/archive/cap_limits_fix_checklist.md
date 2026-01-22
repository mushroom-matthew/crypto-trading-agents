# Cap Limit Fix Checklist

- Refactor `agents/strategies/plan_provider.py:_enrich_plan` so derived caps are computed separately; when fixed-caps flag is enabled, leave `max_trades_per_day` and `max_triggers_per_symbol_per_day` unchanged and only attach `_derived_*`.
- Gate cap overwrites in `services/strategist_plan_service.py` (`generate_plan_for_run` and `_enforce_derived_trade_cap`) behind the fixed-caps flag; keep judge min() logic on policy caps; always surface derived caps in plan limits for telemetry.
- Align backtesting/session handling: `_session_cap_reached` should prefer derived cap if present but never lower policy caps; ensure trigger budget trimming uses the policy cap.
- Add unit tests covering fixed vs legacy behavior (daily budget 10%, per-trade 1%, env cap 30 → fixed keeps 30 with derived recorded; legacy clamps to derived; mirror for trigger cap).
- Expand logging/plan_limits payload to record policy vs derived caps along with budget pct and per-trade risk inputs.
