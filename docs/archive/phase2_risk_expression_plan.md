## Phase 2: Risk Expression & Budget Alignment

Context: Phase 1 telemetry shows the plumbing is solid (PnL components, brakes, trigger telemetry) but risk usage remains low while caps still fire. Phase 2 moves to regime-aware risk surfaces, trade-quality telemetry, and goal-oriented control.

- **Risk regimes (mandatory):** Define risk tiers keyed to regime (trend/range/chop × vol) with per-trade risk, daily budget, caps, session multipliers, and `regime_confidence`. Strategist emits `{regime_label, regime_confidence}`; judge interpolates tiers (<0.6 conservative mixed, 0.6–0.8 interpolate, >0.8 full tier). Elasticity: trade risk, budget, caps scale with vol_state, liquidity/spread_cost, recent RPR. Handle regime flips (regime_delta, regime_conf_delta) with clamps/smoothing.
- **Return-per-Risk (RPR) telemetry:** Per trigger/timeframe/hour: RPR, risk_used_per_trade, win/loss rates, mean/median R, MAE/MFE, relative efficiency (R/MFE, R/|MAE|), asymmetry, response_latency, signal_decay, trigger_load. Add PnL attribution by regime/conf bucket, archetype, timeframe, hour.
- **Factorial experiments:** Test interactions (flatten × risk × session × timeframe × regime), not single-axis sweeps.
- **Goal-oriented judge:** Targets (e.g., utilization 40–70%, execution 20–40% with SNR scaling, 1h share 70–90%, fee_pct_per_trade <5% of gross). Judge outputs deltas (sizing/budget, session boosts, trigger pruning, risk_smoothing_factor, archetype_bias, turnover_penalty, execution_rate_target) and clamps on regime flips.
- **Conditional flattening:** Rules: flatten_when_risk_over(X), regime change, vol_state == high, fees_pct_of_gross > threshold, loss > k*ATR, regime-scaled MAE/MFE (MAE/ATR, MFE/ATR), plus base modes (`none`, `daily_close`, `session_close_utc`).
- **Trigger ROC/pruning:** Rank triggers by RPR/EV and MAE/MFE vectors; cluster into archetypes (trend continuation/reversal, breakout, mean reversion, vol compression/expansion, scalps, exits); prune/tune at archetype level; include edge persistence.

### Workstreams & Checks
1) **Risk utilization (done)**  
   - `risk_budget_utilization_pct` in daily/run summaries; judge reacts to low-util wins / high-util losses.
   - Progress: run summaries now emit mean/median utilization + band splits, but judge wiring is still missing; future agents should pipe `risk_budget_utilization_pct` into judge context before raising caps.

2) **Risk regimes (todo)**  
   - Regime → risk tier mapping with regime_confidence.  
   - Judge interpolates tiers by confidence and applies elasticity (vol_state, liquidity/spread, recent RPR).  
   - Handle regime transitions: detect `regime_delta`, `regime_conf_delta`; clamp risk and apply temporary `risk_smoothing_factor`.
   - Status: no regime-tier logic in backtester/judge today; strategist outputs `regime` strings only. Implement regime_confidence plumbing end-to-end before tuning tiers.

3) **RPR telemetry (high priority)**  
   - Daily/run: per trigger/timeframe/hour RPR, win rate, mean/median R, MAE/MFE, relative efficiency, asymmetry, response_latency, signal_decay, trigger_load.  
   - PnL attribution: by regime, regime confidence bucket (0–0.6 / 0.6–0.8 / 0.8–1.0), archetype, timeframe, hour.  
   - Drive trigger pruning, archetype bias, stop/exit tuning, cap bias (favor high RPR/low load/low latency).
   - Status: daily/run summaries now include `trigger_quality`/`timeframe_quality`/`hour_quality` with RPR, win rate, mean R, latency, MAE/MFE/response_decay, relative efficiency (R/|MAE|, R/MFE), asymmetry (win vs loss magnitude), and avg trigger_load per timeframe/hour. Load blocking is coarse (`TRIGGER_LOAD_THRESHOLD` default 12); refine to archetype-level pruning before experiments.

4) **Trade cap alignment (in progress)**  
   - Enforce derived caps; session/timeframe caps (1h favored, 4h throttle/prune).  
   - Incorporate trigger_load: reduce size or prune redundant archetype triggers when load exceeds threshold.  
   - Allow session caps to lift when utilization < target and returns ≥ 0.
   - Progress: timeframe caps and session trade multipliers now block executions inside `_process_orders_with_limits` using plan-derived trade caps (multiplied by session window). Still missing: trigger_load-aware pruning and elasticity; monitor blocked reason `timeframe_cap`/`session_cap`.

5) **Flatten policy (expanded)**  
   - Base modes + conditional rules: flatten_when_risk_over(X), regime change, vol_state == high, fees_pct_of_gross > threshold, loss > k*ATR, regime-scaled MAE/MFE (MAE/ATR, MFE/ATR).  
   - A/B: `none`, `daily_close`, `session_close_utc` + conditional triggers; compare `flattening_pct_mean`, `fees_pct_mean`, utilization, drawdown.
   - Status: `flatten_policy` now drives behavior (daily_close ➜ end-of-day flatten; session_close_utc ➜ `flatten_session_hour`, none ➜ no forced flatten). Conditional flatten rules still unimplemented; add regime/ATR/fee-based hooks next.

6) **Budget & per-trade risk sweeps (todo)**  
   - Daily budget: 1.0 / 3.75 / 7.5%.  
   - Per-trade risk: 0.25 / 0.75 / 1.5% with fixed budget.  
   - Metrics: utilization bands, blocked_by_risk_budget vs daily_cap, trade_count, exec rate, net/gross/fees, turnover volatility.
   - Note: P1 budget sweeps exist in `scripts/ab_backtest_runner.py`; no P2 per-trade sweep outputs checked in. Run fresh sweeps after RPR telemetry lands to avoid rework.

7) **Loss-cap vs budget stress (todo)**  
   - Volatile windows: compare with/without tight `max_daily_loss_pct`; confirm independent trigger of loss cap vs budget.
   - Status: no targeted runs yet; `run_summary` lacks loss-cap attribution. Add loss-cap hit counts to daily/run summaries before testing.

8) **Factorial sweeps (todo)**  
   - Use A/B helper to combine axes: flatten × risk × session multipliers × timeframe caps × regime windows; inspect interaction effects.
   - Blocker: session/timeframe caps not enforced; postpone factorial runs until cap enforcement is wired.

9) **Judge as optimizer (todo)**  
   - Encode targets; judge outputs deltas: `adjust_sizing`, `adjust_daily_budget`, `session_risk_boost`, `prune_triggers`, `flatten_condition`, `risk_smoothing_factor`, `archetype_bias`, `turnover_penalty`, `execution_rate_target` (SNR-scaled).  
   - Incorporate regime transitions: clamp + smoothing on flips.
   - Status: judge currently score-only; no adjustment deltas emitted. After RPR and cap telemetry, add structured deltas with clamps on regime flips.

10) **LLM cadence/value (later)**  
    - After risk regimes/RPR are in place: test `llm_calls_per_day` 1 vs 4 vs off; weigh utilization/returns vs cost.
    - Note: defer until telemetry/pruning are stable to avoid confounding cost-benefit.

11) **Risk turnover stability (todo)**  
    - Track `risk_turnover_pct = abs(position_change_notional)/equity` and turnover volatility.  
    - Penalize high turnover in judge adjustments; reward stable moderate turnover.
    - Status: turnover not recorded in reports; add daily/run stats before judge hooks.

12) **Trigger responsiveness (todo)**  
    - Track `response_latency` and `signal_decay`; de-emphasize signals with rapid decay outside active hours or high-load bars.
    - Status: no latency/decay capture in slot reports; instrument trigger evaluation timestamps to compute latency/decay before pruning.

13) **Edge persistence (todo)**  
    - Compute `edge_persistence = corr(trigger_rpr_day_t, trigger_rpr_day_t-1:t-N)` per trigger/archetype to distinguish stable vs episodic edges and bias pruning/betting accordingly.
    - Status: blocked on RPR telemetry. Add rolling correlations once RPR per trigger/day exists.

14) **Risk layering order (explicit)**  
    1. regime_label & regime_confidence  
    2. base tier selection  
    3. tier interpolation by confidence  
    4. elasticity adjustments (vol, liquidity/spread, recent RPR)  
    5. judge deltas (sizing, budget, archetype_bias, session_risk_boost, turnover_penalty, execution_rate_target)  
    6. smoothing (`risk_smoothing_factor`, transition clamps on regime flips)  
    7. cap derivation (per-session, per-timeframe, daily) and trigger_load mitigation  
    8. position sizing application  
    9. trade pathing  
    10. flatten conditions (including regime-scaled MAE/MFE thresholds)

### Regime Windows to Reuse
- Bull: 2020-10-01 → 2021-04-14; late bull: 2021-07-20 → 2021-11-10
- Bear: 2018-01-01 → 2018-12-15; 2022-04-01 → 2022-11-10; Covid crash: 2020-02-20 → 2020-03-15
- Range: 2019-01-01 → 2019-10-01; 2021-05-01 → 2021-07-20; 2023-06-01 → 2023-10-01
- High vol: 2021-05-01 → 2021-06-01 (Elon/China); 2022-11-01 → 2022-11-20 (FTX)

### A/B Helper
- `scripts/ab_backtest_runner.py` supports scenario matrices, regime overrides, and `--smoke-days` trimming. Phases: P0 (baseline), P1 (budget sweep), P2 (per-trade risk), P3 (loss-cap stress), P4 (flatten matrix), P5 (timeframe/session), P6 (cadence). Use factorial runs by combining phases/regimes as needed.
