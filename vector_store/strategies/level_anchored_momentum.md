---
title: Level Anchored Momentum
type: strategy
regimes: [bull, volatile]
tags: [trend, momentum, htf, level, stop_anchor]
identifiers: [htf_daily_low, htf_prev_daily_low, htf_daily_high, htf_5d_high,
              donchian_upper_short, below_stop, above_target, stop_hit, target_hit,
              sma_medium, macd_hist, rsi_14, atr_14, trend_state]
template_file:
---
# Level Anchored Momentum

Context
- Momentum trade with stops and targets anchored to HTF structural levels (Runbook 41/42).
- Favors strong uptrends where `trend_state` is "uptrend" with HTF daily structure intact.
- Requires Runbook 42 stop anchors active (stop_price_abs set at fill time).

Entry patterns
- Trend continuation: `close > sma_medium` and `macd_hist > 0` and `rsi_14` between 50–70.
- Structural breakout: `close > htf_daily_high` (closing above prior day high).

Exit / risk reduce
- Stop triggered: `below_stop` (stop_hit is canonical direction-aware alias).
- Target: `above_target` (target_hit alias) or next HTF resistance.
- Momentum fade: `macd_hist < 0` or `rsi_14 < 45`.

Regime alerts
- HTF trend break: `close < htf_daily_low` — structural damage, flatten.
- Vol expansion: `atr_14` rising sharply without price follow-through → suspicious.
