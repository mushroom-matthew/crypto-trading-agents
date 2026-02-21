---
title: Donchian Breakout
type: playbook
regimes: [range, volatile, bull]
tags: [donchian, breakout, momentum]
identifiers: [donchian_upper_short, donchian_lower_short, close, atr_14, volume,
              breakout_confirmed, vol_burst, volume_multiple]
hypothesis: "A close above donchian_upper_short (prior bar's channel) with vol_burst=1
  confirms a range breakout that reaches 1R within 48 bars at win_rate > 55%, and has
  lower false-breakout rate than closes without vol_burst."
min_sample_size: 20
playbook_id: donchian_breakout
---
# Donchian Breakout

Use Donchian channel extremes to detect range breakouts.

Patterns
- Long breakout: `close > donchian_upper_short` (prior bar's channel) signals a bullish
  range escape. Stronger when `atr_14` is expanding and `vol_burst = 1`.
- Short breakout: `close < donchian_lower_short` signals a bearish range escape.
  Confirm with rising `atr_14`.
- False breakout guard: if price closes back inside the channel within 2 bars, exit
  immediately — the break was not sustained.

Notes
- Best after prolonged range regimes where channels have compressed.
- **Always use the PRIOR bar's Donchian** for breakout detection — `close > donchian_upper`
  of the CURRENT bar is always false because `high >= close` in OHLCV data.
- Require at least one confirmation candle above/below the channel before adding to position.
- Place stops just inside the opposite channel boundary at time of entry.

## Research Trade Attribution
<!-- Conditions that tag a research trade to this playbook in the research budget -->
- Entry rule includes `donchian_upper_short` or `donchian_lower_short`, OR
- `breakout_confirmed > 0` at entry bar, OR
- Entry trigger category is `volatility_breakout` with Donchian identifier present

## Validation Evidence
<!-- Auto-updated by PlaybookOutcomeAggregator. Do not edit manually. -->
status: insufficient_data
n_trades: 0
win_rate: null
avg_r: null
median_bars_to_outcome: null
false_breakout_rate: null
last_updated: null
judge_notes: null
