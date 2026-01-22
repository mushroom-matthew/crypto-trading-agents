# Market-Structure Telemetry (Support, Resistance, Tests, Reclaims)

This note defines how we will surface market-structure signals (support/resistance, swing structure, tests/reclaims) to both the backtesting stack and the LLM strategist/judge. The goal is to land deterministic, explainable telemetry first, then iterate on prompts and risk logic in later phases.

## Concepts & Detection Rules
- **Swing points:** Local pivots found via left/right window comparisons on highs/lows. Pivot lows seed support; pivot highs seed resistance.
- **Level width:** Use ATR (14) when available as the base buffer; fall back to a small percent of price (0.5%) to avoid zero-width zones.
- **Support/Resistance levels:** Pivot-derived price with:
  - `strength`: how many pivots contributed inside the buffer.
  - `last_touched`: most recent bar that traded into the buffer.
  - `source`: `swing_low` or `swing_high` (later: volume nodes).
- **Market structure:** Compare the last two swing highs and lows:
  - Higher highs + higher lows → `uptrend`
  - Lower highs + lower lows → `downtrend`
  - Mixed/inside → `range`
  - Missing pivots → `unclear`
- **Tests/Reclaims:** When intrabar high/low enters a level buffer:
  - Support hold: close >= level
  - Support failure: close < level
  - Resistance reject: close <= level
  - Resistance breakout: close > level
  - Reclaim: price loses then re-crosses and holds the level on the next touch.

## Telemetry Schema (draft)
Emitted per symbol/timeframe slice; intended for daily reports, strategy exports, and LLM context.

```json
{
  "timestamp": "2024-06-01T12:00:00Z",
  "symbol": "BTC-USD",
  "timeframe": "1h",
  "trend": "uptrend",
  "last_swing_high": 44750.0,
  "last_swing_low": 43200.0,
  "nearest_support": 43200.0,
  "nearest_resistance": 44750.0,
  "distance_to_support_pct": -3.2,
  "distance_to_resistance_pct": 2.1,
  "support_levels": [43200.0, 42500.0],
  "resistance_levels": [44750.0, 45500.0],
  "recent_tests": [
    {
      "timestamp": "2024-06-01T08:00:00Z",
      "level": 44750.0,
      "side": "resistance",
      "result": "failed_breakout",
      "attempts": 3,
      "window": "4h"
    }
  ]
}
```

Represented in code by `MarketStructureTelemetry`, `SupportLevel`, `ResistanceLevel`, `MarketStructureState`, and `LevelTestEvent` in `agents/analytics/market_structure.py`.

## Module & Responsibility Map
- **Detection (`agents/analytics/market_structure.py`):**
  - Find swing highs/lows (`find_swing_points`).
  - Build support/resistance sets with ATR-sized buffers (`compute_support_resistance_levels`).
  - Infer market structure (`infer_market_structure_state`).
  - Detect test/retest/reclaim events (`detect_level_tests`).
  - Produce a serializable telemetry bundle (`build_market_structure_snapshot`).
- **LLM input (`schemas/llm_strategist.LLMInput`):**
  - New `market_structure` map carries the latest per-symbol snapshot (mirrors `market_structure` in slot reports).
  - `global_context.market_structure` mirrors the same payload for prompt-side consumption.
- **Backtesting ingestion (`backtesting/llm_strategist_runner.py`):**
  - Compute a per-slot `market_structure` brief (primary timeframe) for daily reports and LLM input.
  - Per-trade logs include the entry snapshot; TODO: thread into `plan_log` and `run_summary.json`.
- **Indicator plumbing (`agents/analytics/indicator_snapshots.py`):**
  - TODO: attach market-structure telemetry alongside `AssetState` for LLM input once the schema is agreed.
- **Strategy exports (`scripts/export_strategy_cache.sh`):**
  - TODO: include market-structure slices per day/timeframe in `strategy_export.json` for downstream analysis and prompt replay.

## Metrics to Track (backtests & live)
- Trade-level: entry distance (% and ATR) to nearest support/resistance; classify whether entry was near support, near resistance, or mid-range.
- Outcome splits: win rate, R multiple, MAE/MFE by zone (support vs. resistance vs. mid-range).
- Event-aware: performance after successful reclaims vs. failed tests; frequency of repeated tests before break/reject.
- Regime-aware: trend (`uptrend/downtrend/range/unclear`) vs. risk usage and trigger category mix.
- Logging hooks:
  - Per-trade log rows should carry `market_structure_snapshot` at entry.
  - Daily summaries should aggregate test counts and average distance-to-level.
  - Run summary should surface distribution buckets for review and prompt tuning.

## Incremental Roadmap
- **Phase 1 (this PR):** Land detection module, telemetry schema, doc, unit tests, and backtester scaffolding field for `market_structure`.
- **Phase 2:** Wire telemetry into LLM strategist/judge inputs and prompts; add per-trade logging (entry distance, level side) and daily aggregates.
- **Phase 3:** Use structure-aware risk/sizing (e.g., cap risk if price trades into resistance repeatedly; boost after reclaim with trend alignment); gate triggers by mid-range avoidance.
- **Phase 4:** Extend level sources (volume profile nodes, session VWAP bands) and add robustness checks (look-ahead bias guards, multi-timeframe confluence).
