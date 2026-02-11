---
title: Volume Confirmation
type: playbook
regimes: [bull, bear, volatile]
tags: [volume, confirmation, momentum]
identifiers: [volume, close, atr_14, donchian_upper_short, donchian_lower_short]
---
# Volume Confirmation

Use volume as a conviction filter for breakout and momentum entries.

Patterns
- Breakout confirmation: `close > donchian_upper_short` is stronger when `volume` exceeds 1.5x its recent average. Low-volume breakouts are more likely to fail.
- Exhaustion signal: sharp price move with declining `volume` suggests a move is losing steam. Consider reducing position or tightening stops.
- Accumulation: price consolidates near support with above-average `volume`; precedes breakout moves.

Notes
- Volume is a leading indicator; price follows conviction.
- In crypto, volume spikes can also indicate liquidation cascades rather than organic demand; cross-reference with `atr_14` expansion.
- Avoid entries on below-average volume regardless of price signal quality.
