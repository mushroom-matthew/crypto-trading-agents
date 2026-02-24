---
title: Compression Breakout
type: strategy
regimes: [range, volatile]
tags: [breakout, compression, volatility, consolidation]
identifiers: [compression_flag, bb_bandwidth_pct_rank, expansion_flag, breakout_confirmed, is_impulse_candle, is_inside_bar, vol_burst, donchian_upper_short, donchian_lower_short, volume_multiple, atr_14, candle_strength]
template_file: compression_breakout
---
# Compression Breakout

Context
- Price is in a consolidation phase: low BB bandwidth, contracting ATR, inside bars.
- Setup forms over multiple bars (5–20) before a directional expansion.
- `compression_flag == 1` is the primary setup gate.

Entry patterns
- Long: `compression_flag > 0.5` and `breakout_confirmed > 0.5` and `is_impulse_candle > 0.5`
- Short: same conditions with close below `donchian_lower_short`
- Require `vol_burst > 0` and `volume_multiple > 1.5` for A-grade entry.

Exit / risk reduce
- False breakout: close returns inside compression range → `risk_off` immediately.
- Scale out at 1:1 R (`risk_reduce`, `exit_fraction: 0.5`) when measured move reached.
- Trail remainder: exit when `expansion_flag` drops to 0 (bandwidth contracting again).

Stop placement
- Long: below compression range low at entry (`donchian_lower_short` at entry bar).
- Short: above compression range high at entry (`donchian_upper_short` at entry bar).
- Buffer: 0.3–0.5% beyond the level to avoid spread noise.
- Maximum stop: 1.5× ATR from entry.

Regime alerts
- Failed breakout: close back inside range within 2 bars.
- Volume divergence: `breakout_confirmed == 1` but `volume_multiple < 1.0` (institutional
  non-participation — treat as lower grade, reduce size).
