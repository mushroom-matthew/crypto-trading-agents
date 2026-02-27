---
title: Compression Breakout Short
type: strategy
direction: short
regimes: [range, volatile]
tags: [breakout, compression, short, breakdown]
identifiers: [compression_flag, bb_bandwidth_pct_rank, expansion_flag, breakout_confirmed, is_impulse_candle, is_inside_bar, vol_burst, donchian_lower_short, volume_multiple, atr_14, candle_strength, close]
template_file: compression_breakout_short
---

Context: Compression at or above range midpoint (price_position_in_range >= 0.65); expect downside expansion (breakdown).
Entry: `compression_flag > 0.5` and `breakout_confirmed > 0.5` and `close < donchian_lower_short` and `vol_burst > 0.5`.
Stop: above `donchian_upper_short` at entry bar (max 1.5× ATR).
Target: measured move downward (range width projected below breakout) or `htf_daily_low`. Mandatory — must specify `target_anchor_type` or `target_price_abs`.
False-break exit: close returns inside range within 2 bars (anchored to entry-bar stop).
Grade A: `breakout_confirmed == 1` and `volume_multiple > 2.0`; Grade B: `volume_multiple > 1.5`.
