---
title: Donchian Breakout
type: playbook
regimes: [range, volatile, bull]
tags: [donchian, breakout, momentum]
identifiers: [donchian_upper_short, donchian_lower_short, close, atr_14, volume]
---
# Donchian Breakout

Use Donchian channel extremes to detect range breakouts.

Patterns
- Long breakout: `close > donchian_upper_short` signals a bullish range escape. Stronger when `atr_14` is expanding and `volume` exceeds recent average.
- Short breakout: `close < donchian_lower_short` signals a bearish range escape. Confirm with rising `atr_14`.

Notes
- Best after prolonged range regimes where channels have compressed.
- False breakouts are common in choppy markets; require at least one confirmation candle above/below the channel.
- Place stops just inside the opposite channel boundary.
