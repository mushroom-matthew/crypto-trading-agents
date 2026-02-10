---
title: Bollinger Squeeze
type: playbook
regimes: [range, volatile]
tags: [bollinger, volatility]
identifiers: [bollinger_upper, bollinger_lower, bollinger_middle, atr_14, realized_vol_short, close]
---
# Bollinger Squeeze

Use volatility contraction to anticipate breakout moves.

Patterns
- Compression: band width narrows (price oscillates near `bollinger_middle`) with low `realized_vol_short`.
- Expansion trigger: `close` breaks above `bollinger_upper` or below `bollinger_lower` with `atr_14` rising.

Notes
- Avoid forcing direction; use follow-through confirmation after the band break.
