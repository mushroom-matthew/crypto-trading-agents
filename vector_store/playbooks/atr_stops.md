---
title: ATR-Based Stops
type: playbook
regimes: [bull, bear, range, volatile]
tags: [atr, risk_management, stops]
identifiers: [atr_14, close, high, low]
---
# ATR-Based Stops

Use ATR to set volatility-adaptive stop distances.

Patterns
- Trailing stop: place stop at `close - 2 * atr_14` for longs, `close + 2 * atr_14` for shorts. The 2x multiplier avoids noise whipsaws while capturing trend reversals.
- Tight stop: use `1 * atr_14` in low-volatility regimes where price moves are meaningful.
- Wide stop: use `3 * atr_14` in volatile regimes to avoid premature exits.

Notes
- ATR adapts automatically to regime changes; no manual recalibration needed.
- Combine with position sizing: smaller ATR = tighter stop = larger position; larger ATR = wider stop = smaller position, keeping risk constant.
- In trending markets, trail the stop; in mean-reversion setups, use a fixed ATR-based stop.
