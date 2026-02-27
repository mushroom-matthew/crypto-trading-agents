---
title: Compression Breakout Long
type: strategy
direction: long
regimes: [range, volatile]
tags: [breakout, compression, long]
identifiers: [compression_flag, bb_bandwidth_pct_rank, expansion_flag, breakout_confirmed, is_impulse_candle, is_inside_bar, vol_burst, donchian_upper_short, volume_multiple, atr_14, candle_strength, close]
template_file: compression_breakout_long
---

Context: Compression at or below range midpoint (price_position_in_range <= 0.35); expect upside expansion.
Entry: `compression_flag > 0.5` and `breakout_confirmed > 0.5` and `close > donchian_upper_short` and `vol_burst > 0.5`.
Stop: below `donchian_lower_short` at entry bar (max 1.5× ATR).
Target: measured move (range width projected from breakout point) or `htf_daily_high`. Mandatory — must specify `target_anchor_type` or `target_price_abs`.
False-break exit: close returns inside range within 2 bars (anchored to entry-bar stop, not rolling Donchian).
Grade A: `breakout_confirmed == 1` and `volume_multiple > 2.0`; Grade B: `volume_multiple > 1.5`.
