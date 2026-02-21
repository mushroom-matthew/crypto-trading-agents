---
title: Volume Confirmation
type: playbook
regimes: [bull, bear, volatile]
tags: [volume, confirmation, momentum, breakout_quality]
identifiers: [volume, close, atr_14, donchian_upper_short, donchian_lower_short,
              volume_multiple, vol_burst]
hypothesis: "Breakout entries where volume_multiple > 1.5 (vol_burst=1) achieve win_rate
  at least 12 percentage points higher than structurally identical breakouts with
  volume_multiple < 1.0. This validates volume as a conviction filter worth the
  missed entries it causes."
min_sample_size: 25
playbook_id: volume_confirmation
---
# Volume Confirmation

Use volume as a conviction filter for breakout and momentum entries.

Patterns
- Breakout confirmation: `close > donchian_upper_short` is stronger when `volume_multiple`
  exceeds 1.5. Low-volume breakouts (`volume_multiple < 1.0`) have materially higher
  false-breakout rates and should be treated as C-grade entries or skipped.
- Exhaustion signal: sharp price move with declining `volume_multiple` suggests a move
  is losing steam. Consider reducing position or tightening stops.
- Accumulation: price consolidates near support with above-average `volume_multiple`;
  precedes breakout moves. Use as a setup condition, not an entry trigger.

Notes
- Volume is a leading indicator; price follows conviction.
- In crypto, volume spikes can indicate liquidation cascades rather than organic demand;
  cross-reference with `atr_14` expansion. A vol_burst with atr expansion is institutional;
  a vol_burst without atr expansion may be a liquidation event.
- Avoid entries on `volume_multiple < 0.8` regardless of price signal quality — thin
  markets move easily but reverse easily.
- `vol_burst` (binary) and `volume_multiple` (continuous) are complementary: use
  `vol_burst = 1` as a gate, then `volume_multiple` magnitude for grade (A/B/C).

## Research Trade Attribution
<!-- Conditions that tag a research trade to this playbook in the research budget -->
- Entry rule includes `volume_multiple` or `vol_burst`, OR
- Research hypothesis specifically tests the volume-filtered vs. unfiltered comparison, OR
- Two parallel experiment specs active: one requiring vol_burst=1, one without

## Validation Evidence
<!-- Auto-updated by PlaybookOutcomeAggregator. Do not edit manually. -->
status: insufficient_data
n_trades: 0
win_rate: null
avg_r: null
median_bars_to_outcome: null
win_rate_high_volume: null
win_rate_low_volume: null
last_updated: null
judge_notes: null
