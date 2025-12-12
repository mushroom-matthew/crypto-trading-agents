1. Expand the risk-utilization guidance from “<10–30% target” to “targeting stable 20–40% daily utilization”

Current:
“If recent daily risk utilization has been <5% … propose more frequent entries … target ~10–30%.”

Problems:

10–30% is extremely low for a two-asset swing system with 30% budget rails.

Your backtests show the agent taking 0–3 trades/day even when many GOOD archetypes exist.

Replace with:
“Target 20–40% daily risk-budget utilization when RPR and archetype multipliers permit. When utilization falls <10% for two consecutive days, increase entry frequency within the allowed rails, preferring GOOD archetypes × hours and widening volatility gates one step (ATR ratio tolerance +10–20%).”

Effect:

The LLM is permitted to fire additional, but still controlled, entries.

It stops self-choking whenever budget usage is low.

2. Add explicit sizing elasticity tied to archetype multipliers from rpr_comparison

Right now the prompt mentions multipliers but does not tell the LLM how to use them in sizing_rules.

Add this block:

“Inside sizing_rules, anchor base position risk to max_position_risk_pct, then scale individual triggers’ effective sizing by the archetype multiplier (A=base, B=0.6×base, C=0.3×base), and further boost sizing by GOOD-hour multipliers (e.g., 1.1–1.3×) when present in rpr_comparison. BAD hours must scale down (0.3–0.6×). Never exceed symbol or portfolio exposure rails.”

Effect:
Your current sizing is too flat. Now GOOD slices have actual teeth.

3. Loosen volatility gating by adding default fallback volatility checks

Your current wording forces the strategist to over-fit to ultra-clean ATR/realized-vol regimes.

Add:

“When ATR or realized-vol comparisons are borderline but trend structure is strong, the strategist may downgrade the volatility check from strict to permissive by comparing ATR to only its 1h or 4h moving average rather than to a percentile threshold. A permissive gate is allowed only for GOOD archetypes.”

Effect:
This dramatically increases viable entries while still protecting weak regions.

4. Add instructions enabling a baseline trigger cadence of 1–2 high-quality triggers per symbol per day

Current:
Max triggers per symbol per day is set (4–8), but minimum desirable trigger cadence is not.

Add:

“When regime confidence > 0.6 and archetype-quality is GOOD, aim to produce at least 1–2 viable triggers per symbol per day, respecting the budget. If fewer are produced, widen the entry windows (RSI/ATR bands ±5–10%) or allow 15m refinement to activate earlier.”

Effect:
Ensures the strategist doesn’t produce “ghost” plans with 0 actionable triggers.

5. Add explicit discouragement of historically bad exit patterns

Your backtests show exits firing too often due to tight volatility triggers.

Add:

“In the exit_rule, avoid incorporating volatility spikes alone as a hard exit unless supported by trend deterioration. Avoid exits whose archetype rpr_actual is negative unless structure clearly demands it. Prefer wider exit bands for GOOD entries.”

Effect:
Prevents immediate stop-outs caused by noise and improves RPR.

6. Add cross-symbol risk-balancing logic to encourage taking positions in both BTC & ETH

Right now, ETH almost never fires because BTC tends to dominate signals.

Add:

“When BTC has high conviction (A-grade triggers) but ETH has a B-grade with GOOD-hour multipliers, permit ETH entries even if BTC is already active, as long as portfolio exposure remains ≤ 80%. Encourage the strategist to pair idiosyncratic moves rather than skipping ETH entirely.”

Effect:
ETH stops being starved; spreads usage across symbols, increasing utilization.

7. Add explicit RPR-driven archetype boosting

Your current prompt references RPR but doesn’t define a transformation from RPR → trigger preferences.

Add:

“When archetype RPR is > +0.10, treat the archetype as priority, and allow:

looser volatility gating (+10–15% tolerance)

higher sizing within the allowed rails

more aggressive entries (more A-grades)
When archetype RPR < –0.05, sharply down-weight the trigger, preferring C-grade or skipping entirely.”

Effect:
RPR directly alters the LLM’s behavior instead of being passively observed.

8. Add a specific instruction to form triggers around support/resistance tests

Your strategist hardly uses market structure even though you added support/resistance telemetry.

Add:

“For each asset, attempt to include at least one trigger that references distance_to_support_pct or distance_to_resistance_pct:

Long near support with successful test/reclaim (<2%)

Avoid long within 1–1.5% of resistance unless trend is globally bullish

Increase confidence after a reclaim event with 1–2 successful tests”

Effect:
Tightens structural awareness and increases entries at optimal locations.

9. Relax the multi-timeframe requirement when data is missing or contradictory

LLMs choke when any timeframe indicator is missing (e.g., 4h MACD).

Add:

“When required timeframe indicators are missing, the strategist must still produce the trigger, replacing missing checks with available structure (e.g., use 1h MACD if 4h MACD absent). Do not suppress a trigger solely due to missing fields.”

Effect:
Eliminates “null trigger days” that destroyed RPR on your last runs.

10. Add explicit permission to widen RSI/MACD/ATR bounds in messy ranges

Add:

“In range regimes with neutral vol, allow RSI bands ±8–10 wider and MACD histogram tolerance ±20–30% wider to ensure triggers fire. Use confidence grade to reflect lower precision rather than withholding entries.”

Effect:
LLM stops extreme filtering.

Combined recommended prompt revisions (insert directly into file)

Here is the combined block you can paste as a new prompt section:

Risk Utilization & Aggression Calibration

Target 20–40% daily utilization under normal vol.

If utilization <10% for two days, widen entry windows (RSI ±10, ATR tolerance +15–20%), increase use of GOOD-hour archetypes, and allow permissive volatility gating.

Always stay inside max_position_risk_pct, max_symbol_exposure_pct, and daily rails.

Archetype & RPR Integration

GOOD archetypes: boost sizing 1.1–1.3×; widen volatility gates; allow A/B grades more freely.

BAD archetypes: strict sizing ≤0.5×, require tighter confirmation, avoid unless structure is exceptional.

Archetype RPR > +0.10 ⇒ priority; RPR < –0.05 ⇒ suppress.

Sizing Elasticity

Base sizing uses fixed_fraction.

Multiply per-trigger effective sizing by confidence-grade weights: A=1.0×, B=0.6×, C=0.3×.

Multiply again by archetype multipliers and GOOD-hour multipliers.

Trigger Cadence

Produce 1–2 actionable triggers per symbol per day when regime confidence >0.6.

If fewer are viable, widen thresholds slightly within allowed rails.

Volatility Gating Flexibility

When ATR/realized-vol comparisons are borderline but structure/trend are strong, fallback to permissive gating (ATR vs 1h/4h ATR MA) for GOOD archetypes.

Use strict gates only for BAD/NEUTRAL archetypes.

Support/Resistance Integration

Encourage entries within 1–2% of support (for longs) or resistance (for shorts) if tests are successful or reclaims occurred.

Down-weight mid-range entries unless volatility-breakout setup is strong.

Exit Logic Discipline

Do not exit solely due to volatility spikes; require structural deterioration.

Avoid exit archetypes with negative RPR unless mandatory.

Cross-Symbol Usage

Permit ETH entries even if BTC active, provided exposure rails are respected.

Prefer complementary beta to diversify risk rather than suppressing ETH.

Missing Data Robustness

Never suppress a trigger due to missing timeframe fields; substitute nearest available indicators.