"""Opportunity scanner service.

Runbook 74: OpportunityCard Scorer and Scanner Service.

Scores each symbol in the trading universe using a 6-component weighted formula.
Designed to run every 5-15 min (non-blocking; integrated as a Temporal activity).

Score formula:
  opportunity_score = 0.28*vol_edge + 0.24*structure_edge + 0.18*trend_edge
                    + 0.20*liquidity_score - 0.07*spread_penalty - 0.03*instability_penalty
  opportunity_score_norm = clamp((opportunity_score + 0.10) / 1.00, 0, 1)

All component scores are individually clamped to [0, 1] before weighting.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from schemas.llm_strategist import IndicatorSnapshot
from schemas.opportunity import OpportunityCard, OpportunityRanking
from schemas.structure_engine import StructureSnapshot

# ---------------------------------------------------------------------------
# Score weights (must sum to 1.0 across positive terms)
# ---------------------------------------------------------------------------
_W_VOL_EDGE = 0.28
_W_STRUCTURE_EDGE = 0.24
_W_TREND_EDGE = 0.18
_W_LIQUIDITY = 0.20
_W_SPREAD_PENALTY = 0.07
_W_INSTABILITY_PENALTY = 0.03

# Normalisation offset and range for [0, 1] clamping
# opportunity_score_norm = clamp((score + 0.10) / 1.00, 0, 1)
_NORM_OFFSET = 0.10
_NORM_RANGE = 1.00

# ATR expansion threshold for "high expansion" (score = 1.0)
_ATR_HIGH_EXPANSION_RATIO = 1.5   # atr / atr_20 >= 1.5 → vol_edge = 1.0
_STRUCTURE_LEVELS_FULL = 10       # levels count for structure_edge = 1.0
_STRUCTURE_NEAR_PRICE_PCT = 0.02  # within 2% of price = "near" for bonus
_ADX_HIGH = 50.0                  # ADX >= 50 → trend component = 1.0
_VOLUME_RATIO_HIGH = 2.0          # volume_ratio >= 2.0 → liquidity_score = 1.0
_SPREAD_HIGH_PCT = 0.01           # spread >= 1% of mid → spread_penalty = 1.0
_INSTABILITY_FAILURES_MAX = 3     # consecutive failures / max → clamped to 1.0


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _score_vol_edge(indicator: IndicatorSnapshot) -> tuple[float, str]:
    """ATR expansion vs 20-bar mean.

    Uses atr_14 as the current ATR and derives the 20-bar mean from
    the 3-period history fields (atr_14_prev1, atr_14_prev2, atr_14_prev3)
    when a direct atr_20 field is not available.
    """
    atr = indicator.atr_14
    if atr is None or atr <= 0:
        return 0.5, "atr unavailable — neutral score"

    # Build 20-bar ATR proxy from available history
    prevs = [
        indicator.atr_14_prev1,
        indicator.atr_14_prev2,
        indicator.atr_14_prev3,
    ]
    avail = [p for p in prevs if p is not None and p > 0]
    if not avail:
        return 0.5, f"atr={atr:.4f} but no history — neutral score"

    avg_atr = sum(avail) / len(avail)
    ratio = atr / avg_atr

    # Normalise: ratio 1.0 → score 0.5; ratio 1.5 → score 1.0; ratio 0.5 → score 0.0
    # score = clamp((ratio - 0.5) / (1.5 - 0.5), 0, 1)
    score = _clamp((ratio - 0.5) / (_ATR_HIGH_EXPANSION_RATIO - 0.5))
    direction = "contracting" if ratio < 1.0 else "expanding"
    return score, f"atr={atr:.4f} ratio={ratio:.2f} ({direction})"


def _score_structure_edge(
    indicator: IndicatorSnapshot,
    structure: Optional[StructureSnapshot],
) -> tuple[float, str]:
    """Level count and clarity from structure snapshot."""
    if structure is None:
        return 0.3, "no structure snapshot"

    levels = structure.levels
    level_count = len(levels)

    # Base score from level count
    base = _clamp(level_count / _STRUCTURE_LEVELS_FULL)

    # Bonus for levels near current price (within 2%)
    price = structure.reference_price
    near_count = sum(
        1
        for lvl in levels
        if abs(lvl.price - price) / price <= _STRUCTURE_NEAR_PRICE_PCT
    )
    # Bonus capped at +0.2
    near_bonus = _clamp(near_count / 3) * 0.2

    score = _clamp(base + near_bonus)
    return score, f"levels={level_count} near_price={near_count} base={base:.2f}"


def _score_trend_edge(indicator: IndicatorSnapshot) -> tuple[float, str]:
    """EMA alignment score + ADX component."""
    price = indicator.close

    # EMA alignment: count how many EMAs are below price (for long bias)
    ema_fields = [
        indicator.ema_short,
        indicator.ema_medium,
        indicator.ema_long,
        indicator.ema_50,
        indicator.ema_200,
    ]
    available = [e for e in ema_fields if e is not None]
    if not available:
        ema_score = 0.5
        ema_desc = "no EMAs"
    else:
        below = sum(1 for e in available if e < price)
        above = sum(1 for e in available if e > price)
        # Alignment = max(below, above) / total → 1.0 when all agree
        alignment = max(below, above) / len(available)
        ema_score = _clamp(alignment)
        ema_desc = f"below={below}/{len(available)}"

    # ADX component: directional momentum strength
    adx = indicator.adx_14
    if adx is not None:
        adx_score = _clamp(adx / _ADX_HIGH)
        adx_desc = f"adx={adx:.1f}"
    else:
        adx_score = 0.3
        adx_desc = "adx unavailable"

    # Combine EMA alignment (60%) + ADX (40%)
    score = 0.6 * ema_score + 0.4 * adx_score
    return score, f"ema={ema_desc} {adx_desc}"


def _score_liquidity(indicator: IndicatorSnapshot) -> tuple[float, str]:
    """Volume ratio (current volume / 20-bar mean)."""
    vol_ratio = getattr(indicator, "volume_multiple", None)
    if vol_ratio is None:
        # Try to derive from raw volume (not ideal but graceful)
        return 0.3, "volume_multiple unavailable"

    score = _clamp(vol_ratio / _VOLUME_RATIO_HIGH)
    return score, f"volume_multiple={vol_ratio:.2f}"


def _score_spread_penalty(ticker: Optional[Dict[str, Any]]) -> tuple[float, str]:
    """Bid/ask spread as fraction of mid price."""
    if ticker is None:
        return 0.0, "no ticker data — no penalty"

    bid = ticker.get("bid") or ticker.get("bidPrice")
    ask = ticker.get("ask") or ticker.get("askPrice")
    last = ticker.get("last") or ticker.get("close")

    if bid is None or ask is None:
        if last:
            return 0.0, "bid/ask missing — no penalty"
        return 0.0, "no price data — no penalty"

    bid = float(bid)
    ask = float(ask)
    mid = (bid + ask) / 2.0
    if mid <= 0:
        return 0.0, "mid=0 — no penalty"

    spread_pct = (ask - bid) / mid
    score = _clamp(spread_pct / _SPREAD_HIGH_PCT)
    return score, f"spread={spread_pct*100:.3f}%"


def _score_instability_penalty(consecutive_failures: int) -> tuple[float, str]:
    """Price feed instability based on consecutive fetch failures."""
    score = _clamp(consecutive_failures / _INSTABILITY_FAILURES_MAX)
    return score, f"consecutive_failures={consecutive_failures}"


def _compute_hold_horizon(
    indicator: IndicatorSnapshot,
    vol_edge: float,
    trend_edge: float,
    structure_edge: float,
) -> str:
    """Derive expected hold horizon from score components.

    scalp:    ATR contracting (vol_edge < 0.3)
    swing:    strong trend + good structure (trend_edge > 0.6 and structure_edge > 0.5)
    intraday: everything else
    """
    if vol_edge < 0.3:
        return "scalp"
    if trend_edge > 0.6 and structure_edge > 0.5:
        return "swing"
    return "intraday"


def score_symbol(
    symbol: str,
    indicator: IndicatorSnapshot,
    structure: Optional[StructureSnapshot] = None,
    ticker: Optional[Dict[str, Any]] = None,
    consecutive_price_failures: int = 0,
    playbook_win_rate: Optional[float] = None,
    playbook_r_expectancy: Optional[float] = None,
    attribution_sample_size: int = 0,
) -> OpportunityCard:
    """Score a single symbol and return its OpportunityCard.

    Args:
        symbol: Trading pair symbol (e.g. "POL/USDT").
        indicator: IndicatorSnapshot for the symbol (current bar).
        structure: StructureSnapshot (optional; reduces structure_edge if absent).
        ticker: Raw exchange ticker dict with bid/ask/last (optional).
        consecutive_price_failures: From SessionState.consecutive_price_failures.

    Returns:
        OpportunityCard with all 6 component scores and explanations.
    """
    vol_edge, vol_desc = _score_vol_edge(indicator)
    structure_edge, struct_desc = _score_structure_edge(indicator, structure)
    trend_edge, trend_desc = _score_trend_edge(indicator)
    liquidity_score, liq_desc = _score_liquidity(indicator)
    spread_penalty, spread_desc = _score_spread_penalty(ticker)
    instability_penalty, inst_desc = _score_instability_penalty(consecutive_price_failures)

    raw_score = (
        _W_VOL_EDGE * vol_edge
        + _W_STRUCTURE_EDGE * structure_edge
        + _W_TREND_EDGE * trend_edge
        + _W_LIQUIDITY * liquidity_score
        - _W_SPREAD_PENALTY * spread_penalty
        - _W_INSTABILITY_PENALTY * instability_penalty
    )
    # R79: attribution bonus — +0.05 * clamp(r_expectancy, 0, 2) / 2
    # Only applied when sufficient prior data exists.
    if playbook_r_expectancy is not None and attribution_sample_size >= 5:
        _attr_bonus = 0.05 * _clamp(playbook_r_expectancy, 0.0, 2.0) / 2.0
        raw_score += _attr_bonus
    norm_score = _clamp((raw_score + _NORM_OFFSET) / _NORM_RANGE)

    horizon = _compute_hold_horizon(indicator, vol_edge, trend_edge, structure_edge)

    # Nearest support/resistance from structure
    nearest_support: Optional[float] = None
    nearest_resistance: Optional[float] = None
    levels_count = 0
    if structure is not None:
        levels_count = len(structure.levels)
        price = structure.reference_price
        supports = [lvl.price for lvl in structure.levels if lvl.role_now == "support"]
        resistances = [lvl.price for lvl in structure.levels if lvl.role_now == "resistance"]
        if supports:
            nearest_support = max(s for s in supports if s < price) if any(s < price for s in supports) else min(supports)
        if resistances:
            nearest_resistance = min(r for r in resistances if r > price) if any(r > price for r in resistances) else max(resistances)

    now = datetime.now(timezone.utc)
    return OpportunityCard(
        symbol=symbol,
        opportunity_score=raw_score,
        opportunity_score_norm=norm_score,
        vol_edge=vol_edge,
        structure_edge=structure_edge,
        trend_edge=trend_edge,
        liquidity_score=liquidity_score,
        spread_penalty=spread_penalty,
        instability_penalty=instability_penalty,
        expected_hold_horizon=horizon,
        scored_at=now,
        indicator_as_of=indicator.as_of,
        component_explanation={
            "vol_edge": vol_desc,
            "structure_edge": struct_desc,
            "trend_edge": trend_desc,
            "liquidity_score": liq_desc,
            "spread_penalty": spread_desc,
            "instability_penalty": inst_desc,
            "expected_hold_horizon": f"horizon={horizon}",
        },
        nearest_support=nearest_support,
        nearest_resistance=nearest_resistance,
        structure_levels_count=levels_count,
        playbook_win_rate=playbook_win_rate,
        playbook_r_expectancy=playbook_r_expectancy,
        attribution_sample_size=attribution_sample_size,
    )


def rank_universe(
    symbols: List[str],
    indicator_snapshots: Dict[str, IndicatorSnapshot],
    structure_snapshots: Optional[Dict[str, StructureSnapshot]] = None,
    tickers: Optional[Dict[str, Dict[str, Any]]] = None,
    consecutive_failures_by_symbol: Optional[Dict[str, int]] = None,
    episode_priors: Optional[Dict[str, Dict[str, Any]]] = None,
    top_n: int = 10,
) -> "OpportunityRanking":
    """Score all symbols and return a ranked OpportunityRanking.

    Args:
        symbols: Universe of symbols to score.
        indicator_snapshots: Dict mapping symbol → IndicatorSnapshot.
        structure_snapshots: Optional dict mapping symbol → StructureSnapshot.
        tickers: Optional dict mapping symbol → raw exchange ticker.
        consecutive_failures_by_symbol: Per-symbol failure counts.
        episode_priors: Optional dict mapping symbol → prior dict from
            compute_playbook_prior() (R79). Keys: win_rate, r_expectancy,
            sample_size, sufficient_data.
        top_n: How many cards to include in the ranking.

    Returns:
        OpportunityRanking sorted by opportunity_score_norm descending.
    """
    start_ms = int(time.time() * 1000)
    cards: List[OpportunityCard] = []

    for symbol in symbols:
        indicator = indicator_snapshots.get(symbol)
        if indicator is None:
            continue  # skip symbols with no indicator data

        structure = (structure_snapshots or {}).get(symbol)
        ticker = (tickers or {}).get(symbol)
        failures = (consecutive_failures_by_symbol or {}).get(symbol, 0)

        # R79: attribution priors
        prior = (episode_priors or {}).get(symbol) or {}
        _win_rate = prior.get("win_rate") if prior.get("sufficient_data") else None
        _r_exp = prior.get("r_expectancy") if prior.get("sufficient_data") else None
        _sample = prior.get("sample_size", 0)

        try:
            card = score_symbol(
                symbol=symbol,
                indicator=indicator,
                structure=structure,
                ticker=ticker,
                consecutive_price_failures=failures,
                playbook_win_rate=_win_rate,
                playbook_r_expectancy=_r_exp,
                attribution_sample_size=_sample,
            )
            cards.append(card)
        except Exception:
            # Scoring failure is non-fatal — skip symbol
            pass

    cards.sort(key=lambda c: c.opportunity_score_norm, reverse=True)
    top_cards = cards[:top_n]

    elapsed_ms = int(time.time() * 1000) - start_ms
    return OpportunityRanking(
        ranked_at=datetime.now(timezone.utc),
        cards=top_cards,
        universe_size=len(symbols),
        scan_duration_ms=elapsed_ms,
        top_n=len(top_cards),
    )
