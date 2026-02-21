---
title: ATR-Based Stops
type: playbook
regimes: [bull, bear, range, volatile]
tags: [atr, risk_management, stops, stop_distance]
identifiers: [atr_14, close, high, low, vol_state, stop_loss_atr_mult]
hypothesis: "A 1.5x ATR stop (stop_loss_atr_mult=1.5, the current default) achieves
  a better win_rate vs. max_adverse_excursion tradeoff than 1.0x or 2.0x ATR stops
  across all vol_state regimes. The optimal multiplier is lower in low-vol and higher
  in high-vol regimes (vol_state-specific tuning)."
min_sample_size: 30
playbook_id: atr_stops
---
# ATR-Based Stops

Use ATR to set volatility-adaptive stop distances.

Patterns
- Default stop: `stop_loss_atr_mult = 1.5` — the current system default (Runbook 42).
  Balances noise avoidance with trend capture across normal volatility.
- Tight stop (low vol): use `1.0x atr_14` when `vol_state = "low"`. Price moves are
  meaningful; stops can be closer without excessive noise-outs.
- Wide stop (high/extreme vol): use `2.5–3.0x atr_14` when `vol_state = "high"` or
  `"extreme"`. Volatility is elevated; normal ATR-sized stops fire on noise.
- Trailing stop: once trade is at 1R profit, trail stop to entry price (breakeven).
  Once at 2R, trail stop to prior bar's low (for longs).

Notes
- ATR adapts automatically to regime changes; no manual recalibration needed.
- Combine with position sizing: smaller ATR = tighter stop = larger position; larger ATR
  = wider stop = smaller position — risk per trade stays constant.
- In trending markets, trail the stop; in mean-reversion setups, use a fixed ATR stop.
- Do NOT use ATR alone for stop placement when level-anchored stops (Runbook 42) are
  available — structural stops (HTF lows, Donchian boundaries) are preferable when they
  exist within 1.5x ATR of entry.

## Research Trade Attribution
<!-- Conditions that tag a research trade to this playbook in the research budget -->
- Stop anchor type is `atr` (stop_loss_atr_mult set on the trigger), OR
- Research hypothesis tests ATR multiplier sensitivity (1.0x vs 1.5x vs 2.5x), OR
- Entry trigger has `stop_loss_atr_mult` field set explicitly (non-default)

## Validation Evidence
<!-- Auto-updated by PlaybookOutcomeAggregator. Do not edit manually. -->
status: insufficient_data
n_trades: 0
win_rate: null
avg_r: null
median_bars_to_outcome: null
avg_mae_pct: null
stop_fired_rate: null
atr_mult_breakdown: null
last_updated: null
judge_notes: null
