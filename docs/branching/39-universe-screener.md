# Branch: universe-screener

## Purpose
The system currently requires the user to specify which instruments to trade. This runbook adds an autonomous universe screener: a service that monitors a configurable set of crypto symbols (default: top 200 by market cap, or a user-defined list), computes per-symbol anomaly scores every N minutes, and surfaces the top candidates to the LLM with enough context for the LLM to make a reasoned instrument recommendation.

The LLM's role shifts from "write RSI thresholds for BTC and ETH" to "here are 5 symbols showing unusual activity — pick one, explain why, and initialize the strategy type." This is a better use of LLM cognition: contextual market analysis vs. rote indicator formulation.

This runbook does NOT require the LLM to scan 200 symbols. The screener is a lightweight deterministic pass that reduces the universe to ~5-10 candidates. The LLM then decides among pre-filtered options.

## Scope
1. **`services/universe_screener_service.py`** — anomaly scoring logic for a symbol list
2. **`schemas/screener.py`** — `ScreenerResult`, `InstrumentRecommendation` Pydantic models
3. **`workflows/universe_screener_workflow.py`** — Temporal workflow (periodic screening cadence)
4. **`services/strategist_plan_service.py`** — consume instrument recommendation at plan initialization
5. **`worker/agent_worker.py`** — register new workflow
6. **`ops_api/routers/screener.py`** — read-only endpoint for UI to display screening results
7. **`prompts/instrument_recommendation.txt`** — new LLM prompt for instrument selection
8. **`tests/test_universe_screener.py`** — unit tests for scoring logic
9. **`tests/test_screener_workflow.py`** — workflow integration tests

## Out of Scope
- Stock market data (equity universe expansion — separate runbook)
- Real-time tick-level screening (OHLCV snapshots at 1m–15m cadence is sufficient)
- Automatic execution without LLM confirmation (screener recommends; user or LLM still decides)
- Screener-driven backtesting (instrument selection can only be validated via paper trading)

## Key Files
- `services/universe_screener_service.py` (new)
- `schemas/screener.py` (new)
- `workflows/universe_screener_workflow.py` (new)
- `services/strategist_plan_service.py` (modify: accept instrument recommendation context)
- `worker/agent_worker.py` (modify: register UniverseScreenerWorkflow)
- `ops_api/routers/screener.py` (new)
- `prompts/instrument_recommendation.txt` (new)
- `tests/test_universe_screener.py` (new)

## Implementation Steps

### Step 1: Define schemas in `schemas/screener.py`

```python
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class SymbolAnomalyScore(BaseModel):
    """Per-symbol anomaly score from the screener."""
    model_config = {"extra": "forbid"}

    symbol: str
    as_of: datetime
    # Raw metrics
    volume_z: float              # (current_vol - vol_sma_20) / vol_std_20
    atr_expansion: float         # (atr_5 / atr_20) - 1.0 (positive = expanding)
    range_expansion_z: float     # (current_range - range_sma_20) / range_std_20
    bb_bandwidth_pct_rank: float # Percentile rank of BB bandwidth in 50-bar window (0=compressed)
    # Context
    close: float
    trend_state: str             # "uptrend" | "downtrend" | "range" | "unclear"
    vol_state: str               # "low" | "normal" | "high" | "extreme"
    dist_to_prior_high_pct: float   # % distance from prior session high (negative = below)
    dist_to_prior_low_pct: float    # % distance from prior session low (positive = above)
    # Composite score
    composite_score: float       # Weighted sum of anomaly components (higher = more interesting)
    score_components: dict       # Breakdown for transparency


class ScreenerResult(BaseModel):
    """Output of one screening pass."""
    model_config = {"extra": "forbid"}

    run_id: str
    as_of: datetime
    universe_size: int
    top_candidates: List[SymbolAnomalyScore]  # Top N by composite_score
    screener_config: dict  # weights, thresholds used


class InstrumentRecommendation(BaseModel):
    """LLM-generated instrument recommendation from screener candidates."""
    model_config = {"extra": "forbid"}

    selected_symbol: str
    thesis: str                  # Why this symbol now
    strategy_type: str           # Which template to apply (e.g., "compression_breakout", "mean_reversion")
    regime_view: str             # Current regime assessment
    key_levels: Optional[dict]   # {"support": float, "resistance": float, "pivot": float}
    expected_hold_timeframe: str # "5m" | "15m" | "1h" | "4h"
    confidence: str              # "high" | "medium" | "low"
    disqualified_symbols: List[str]  # Screener candidates rejected and why (brief)
    disqualification_reasons: dict   # {symbol: reason}
```

### Step 2: Implement `services/universe_screener_service.py`

```python
class UniverseScreenerService:
    """Computes per-symbol anomaly scores from OHLCV snapshots."""

    DEFAULT_UNIVERSE = [
        "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD",
        "DOGE-USD", "AVAX-USD", "LINK-USD", "DOT-USD", "MATIC-USD",
        # ... extend to top 50-200 via config
    ]

    SCORE_WEIGHTS = {
        # Two-phase weights — tune via env: SCREENER_COMPRESSION_WEIGHT, SCREENER_EXPANSION_WEIGHT
        # Default 50/50: surfaces both compression setups (early) and expansion confirmations (momentum).
        # Increase compression weight to find more pre-breakout coiling.
        # Increase expansion weight to find more momentum follow-through.
        "compression": float(os.environ.get("SCREENER_COMPRESSION_WEIGHT", "0.50")),
        "expansion":   float(os.environ.get("SCREENER_EXPANSION_WEIGHT",   "0.50")),
    }

    def __init__(self, universe: list[str] | None = None):
        self.universe = universe or self.DEFAULT_UNIVERSE

    async def screen(self, timeframe: str = "1h", lookback_bars: int = 50) -> ScreenerResult:
        """Fetch OHLCV for all symbols, compute scores, return top N."""
        results = []
        for symbol in self.universe:
            try:
                score = await self._score_symbol(symbol, timeframe, lookback_bars)
                results.append(score)
            except Exception as e:
                logger.warning("Screener: failed to score %s: %s", symbol, e)

        results.sort(key=lambda x: x.composite_score, reverse=True)
        top_n = int(os.environ.get("SCREENER_TOP_N", "8"))
        return ScreenerResult(
            run_id=str(uuid4()),
            as_of=datetime.utcnow(),
            universe_size=len(results),
            top_candidates=results[:top_n],
            screener_config={"weights": self.SCORE_WEIGHTS, "top_n": top_n},
        )

    async def _score_symbol(self, symbol: str, timeframe: str, lookback_bars: int) -> SymbolAnomalyScore:
        df = await self._fetch_ohlcv(symbol, timeframe, lookback_bars)
        if len(df) < 30:
            raise ValueError(f"Insufficient data for {symbol}")

        # Volume anomaly
        vol = df["volume"]
        vol_sma = vol.rolling(20).mean()
        vol_std = vol.rolling(20).std().replace(0, np.nan)
        volume_z = float(((vol.iloc[-1] - vol_sma.iloc[-1]) / vol_std.iloc[-1]).clip(-5, 5))

        # ATR expansion
        atr_series = tech.atr(df, 14).series_list[0].series
        atr_short = atr_series.rolling(5).mean().iloc[-1]
        atr_long = atr_series.rolling(20).mean().iloc[-1]
        atr_expansion = float((atr_short / max(atr_long, 1e-9)) - 1.0)

        # Range expansion
        bar_range = df["high"] - df["low"]
        range_sma = bar_range.rolling(20).mean()
        range_std = bar_range.rolling(20).std().replace(0, np.nan)
        range_expansion_z = float(((bar_range.iloc[-1] - range_sma.iloc[-1]) / range_std.iloc[-1]).clip(-5, 5))

        # BB bandwidth percentile (compression detection)
        bb = tech.bollinger_bands(df, 20, 2.0)
        bandwidth = bb.series_list[3].series  # "bandwidth" key
        bb_rank = float(bandwidth.rank(pct=True).iloc[-1])  # 0=compressed, 1=expanded

        # Two-phase scoring: compression candidates vs expansion candidates
        # Phase A — compression (early-stage setup, before breakout):
        #   Bias: low bandwidth rank = high compression score.
        #   Surfaces coins coiling before a move.
        compression_score = max(0, 1.0 - bb_rank)  # 1.0 when maximally compressed

        # Phase B — expansion (breakout underway, momentum confirmation):
        #   Bias: high volume_z + high atr_expansion = high expansion score.
        #   Surfaces coins that have already broken out and are following through.
        expansion_score = (
            0.45 * max(0, volume_z) +
            0.35 * max(0, atr_expansion) +
            0.20 * max(0, range_expansion_z)
        )

        # Composite: weight compression higher to surface EARLY-STAGE setups.
        # The compression breakout template needs symbols that are coiling NOW,
        # not symbols that already broke out (those are often too late to enter).
        # If you want already-moving symbols, swap the weights.
        score = (
            self.SCORE_WEIGHTS.get("compression", 0.50) * compression_score +
            self.SCORE_WEIGHTS.get("expansion",    0.50) * expansion_score
        )

        # Expose both sub-scores in the context packet for LLM reasoning
        score_components = {
            "compression_score": compression_score,
            "expansion_score": expansion_score,
            "volume_z": volume_z,
            "atr_expansion": atr_expansion,
            "range_expansion_z": range_expansion_z,
            "bb_bandwidth_rank": bb_rank,
            "composite": score,
        }

        # Context fields
        prior_high = df["high"].iloc[-2]
        prior_low = df["low"].iloc[-2]
        close = df["close"].iloc[-1]

        return SymbolAnomalyScore(
            symbol=symbol,
            as_of=datetime.utcnow(),
            volume_z=volume_z,
            atr_expansion=atr_expansion,
            range_expansion_z=range_expansion_z,
            bb_bandwidth_pct_rank=bb_rank,
            close=close,
            trend_state=self._classify_trend(df),
            vol_state=self._classify_vol(df, atr_series),
            dist_to_prior_high_pct=float((close - prior_high) / prior_high * 100),
            dist_to_prior_low_pct=float((close - prior_low) / prior_low * 100),
            composite_score=score,
            score_components={
                "volume_z": volume_z,
                "atr_expansion": atr_expansion,
                "range_expansion_z": range_expansion_z,
                "bb_bandwidth_rank": bb_rank,
                "bb_bonus": bb_bonus,
            },
        )
```

### Step 3: Create `prompts/instrument_recommendation.txt`

```
You are the instrument selection analyst. You receive pre-screened symbol candidates with anomaly scores. Your task:
1. Select ONE symbol to trade for the next session.
2. Explain the thesis in 2-3 sentences.
3. Specify which strategy template fits (compression_breakout, mean_reversion, trend_continuation, reversal).
4. Identify key structural levels (support, resistance, pivot).
5. Specify the expected holding timeframe.

Selection rules:
- Prefer symbols with volume_z > 1.5 AND atr_expansion > 0.2 (real activity, not noise).
- If bb_bandwidth_pct_rank < 0.20 (compressed), prefer compression_breakout template.
- If volume_z is high but atr_expansion is low, the move may be exhausted — prefer mean_reversion.
- Avoid symbols in "extreme" vol_state unless thesis is specifically volatility-driven.
- Provide disqualification reasons for rejected candidates (1 sentence each).

Output must be valid JSON matching the InstrumentRecommendation schema.
```

### Step 4: Create `workflows/universe_screener_workflow.py`

```python
@workflow.defn
class UniverseScreenerWorkflow:
    """Runs universe screening on a configurable cadence."""

    @workflow.run
    async def run(self, config: dict) -> None:
        cadence_minutes = config.get("cadence_minutes", 15)
        while True:
            result = await workflow.execute_activity(
                run_universe_screen,
                config,
                start_to_close_timeout=timedelta(minutes=5),
            )
            # Signal downstream workflows with the recommendation
            await workflow.execute_activity(
                emit_screener_result,
                result,
                start_to_close_timeout=timedelta(seconds=30),
            )
            await asyncio.sleep(cadence_minutes * 60)
```

### Step 5: Add Ops API endpoint in `ops_api/routers/screener.py`

```python
@router.get("/screener/latest")
async def get_latest_screener_result() -> ScreenerResult:
    """Return the most recent screening pass."""
    ...

@router.get("/screener/recommendation")
async def get_instrument_recommendation() -> InstrumentRecommendation:
    """Return the most recent LLM instrument recommendation."""
    ...
```

## Environment Variables
```
SCREENER_UNIVERSE_FILE=data/universe.json    # Optional override for symbol list
SCREENER_TOP_N=8                             # Candidates sent to LLM (default 8)
SCREENER_CADENCE_MINUTES=15                  # How often to run (default 15)
SCREENER_TIMEFRAME=1h                        # OHLCV timeframe for scoring (default 1h)
SCREENER_COMPRESSION_WEIGHT=0.50            # Weight on compression_score (early-stage setups)
SCREENER_EXPANSION_WEIGHT=0.50              # Weight on expansion_score (momentum follow-through)
```

> **Tuning note:** During the paper trading validation phase, run the screener in parallel with
> both 100% compression weight and 100% expansion weight. Compare which mode surfaces better
> setups for the compression breakout template. Adjust weights accordingly before live capital.

## Test Plan
```bash
# Unit: scoring logic with mock OHLCV
uv run pytest tests/test_universe_screener.py -vv

# Unit: schema validation
uv run pytest -k "screener" -vv

# Integration: workflow smoke test
uv run pytest tests/test_screener_workflow.py -vv
```

## Test Evidence
```
TODO
```

## Acceptance Criteria
- [ ] `UniverseScreenerService.screen()` returns `ScreenerResult` with correctly ranked candidates
- [ ] `volume_z`, `atr_expansion`, `range_expansion_z` computed correctly from OHLCV data
- [ ] `SymbolAnomalyScore` and `ScreenerResult` serialize/deserialize without error
- [ ] `UniverseScreenerWorkflow` runs on schedule and emits results
- [ ] LLM receives candidates and returns valid `InstrumentRecommendation` JSON
- [ ] Ops API endpoint returns latest screener result
- [ ] Unit tests cover: normal scoring, edge cases (insufficient data, zero volume, all-same-close)

## Human Verification Evidence
```
TODO: Run the screener for one hour of live data. Verify:
- Top candidates change as market conditions shift
- Composite scores are plausible (high-volume breakout symbols rank higher than flat ones)
- LLM recommendation reasoning is coherent and tied to the score data
```

## Change Log
| Date | Change | Author |
|------|--------|--------|
| 2026-02-18 | Runbook created from product strategy audit | Claude |

## Worktree Setup
```bash
git fetch
git worktree add -b feat/universe-screener ../wt-universe-screener main
cd ../wt-universe-screener

# When finished (after merge)
git worktree remove ../wt-universe-screener
```

## Git Workflow
```bash
git checkout main
git pull
git checkout -b feat/universe-screener

# ... implement changes ...

git add schemas/screener.py \
  services/universe_screener_service.py \
  workflows/universe_screener_workflow.py \
  ops_api/routers/screener.py \
  prompts/instrument_recommendation.txt \
  worker/agent_worker.py \
  tests/test_universe_screener.py \
  tests/test_screener_workflow.py

uv run pytest tests/test_universe_screener.py tests/test_screener_workflow.py -vv
git commit -m "Add universe screener: anomaly scoring + LLM instrument recommendation"
```
