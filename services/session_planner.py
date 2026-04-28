"""Session planner service (R76).

Generates a SessionIntent — symbol selection, risk allocation, and per-symbol thesis —
before the strategy LLM runs. This decouples symbol selection from trigger generation
and makes the session's intention explicit (AI-led portfolio planning).

The planner uses an LLM call with the top-N OpportunityCards + portfolio state as input.
On failure, it falls back to user-specified symbols with neutral allocation.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from schemas.opportunity import OpportunityCard
from schemas.session_intent import SessionIntent, SymbolIntent

logger = logging.getLogger(__name__)

# How many top-scored symbols to surface to the LLM for selection.
_TOP_N_FOR_LLM = int(os.environ.get("SESSION_PLANNER_TOP_N", "10"))
# How many symbols to select for the session (LLM chooses from top-N).
_SELECT_K = int(os.environ.get("SESSION_PLANNER_SELECT_K", "3"))

_SYSTEM_PROMPT = """\
You are an AI portfolio planner for a crypto paper-trading system.
Your job is to select the best symbols for the upcoming session,
allocate risk budgets, and write a 1-2 sentence thesis for each symbol.

You will receive:
- An OPPORTUNITY_RANKING block: scored symbols with component breakdowns.
- A PORTFOLIO_STATE block: current cash, open positions, session config.

Output ONLY a valid JSON object matching this schema (no markdown, no commentary):
{
  "selected_symbols": ["SYM1", "SYM2", ...],
  "regime_summary": "1-2 sentences on overall market regime",
  "planner_rationale": "Why these symbols over alternatives (1-2 sentences)",
  "planned_trade_cadence_min": <int, trades per 24h low>,
  "planned_trade_cadence_max": <int, trades per 24h high>,
  "symbol_intents": [
    {
      "symbol": "SYM1",
      "opportunity_score_norm": <float 0-1>,
      "risk_budget_fraction": <float 0-1, sum across all symbols ≤ 1.0>,
      "playbook_id": "<string or null>",
      "thesis_summary": "1-2 sentence thesis for this symbol",
      "expected_hold_horizon": "<scalp|intraday|swing>",
      "direction_bias": "<long|short|neutral>"
    }
  ]
}

Rules:
- Select 1 to {select_k} symbols from the OPPORTUNITY_RANKING.
- risk_budget_fraction values must sum to ≤ 1.0.
- Each risk_budget_fraction must be > 0.0 and ≤ 1.0.
- Prefer symbols with high opportunity_score_norm.
- direction_bias should align with trend_edge and structure signals.
- expected_hold_horizon: use the OpportunityCard's expected_hold_horizon as a guide.
- Do NOT include symbols not in the OPPORTUNITY_RANKING.
""".replace("{select_k}", str(_SELECT_K))


def _build_opportunity_block(cards: List[OpportunityCard]) -> str:
    """Format OpportunityCards as a structured text block for the LLM."""
    lines = ["OPPORTUNITY_RANKING (top candidates by opportunity_score_norm):"]
    for rank, card in enumerate(cards, 1):
        lines.append(
            f"  {rank}. {card.symbol}"
            f"  score={card.opportunity_score_norm:.2f}"
            f"  vol={card.vol_edge:.2f}"
            f"  trend={card.trend_edge:.2f}"
            f"  struct={card.structure_edge:.2f}"
            f"  liq={card.liquidity_score:.2f}"
            f"  horizon={card.expected_hold_horizon}"
        )
        if card.component_explanation:
            trend_exp = card.component_explanation.get("trend_edge", "")
            if trend_exp:
                lines.append(f"     trend: {trend_exp}")
    return "\n".join(lines)


def _build_portfolio_block(
    portfolio_state: Dict[str, Any],
    session_config: Dict[str, Any],
) -> str:
    """Format portfolio state + session config as a text block for the LLM."""
    cash = portfolio_state.get("cash", 0.0)
    positions = portfolio_state.get("positions", {})
    open_pos = [f"{sym}({side})" for sym, pos in positions.items()
                for side in [pos.get("side", "?")] if pos]
    regime = session_config.get("screener_regime") or "unknown"
    tf = session_config.get("indicator_timeframe", "1h")

    lines = [
        "PORTFOLIO_STATE:",
        f"  cash={cash:.2f}",
        f"  open_positions={', '.join(open_pos) if open_pos else 'none'}",
        f"  regime_hint={regime}",
        f"  indicator_timeframe={tf}",
        f"  direction_bias={session_config.get('direction_bias', 'neutral')}",
    ]
    return "\n".join(lines)


def _fallback_intent(
    symbols: List[str],
    cards: List[OpportunityCard],
    regime_summary: str = "",
) -> SessionIntent:
    """Build a neutral SessionIntent from user-specified symbols (LLM fallback)."""
    score_map = {c.symbol: c.opportunity_score_norm for c in cards}
    horizon_map = {c.symbol: c.expected_hold_horizon for c in cards}

    n = max(1, len(symbols))
    fraction = round(1.0 / n, 4)

    symbol_intents = [
        SymbolIntent(
            symbol=sym,
            opportunity_score_norm=score_map.get(sym, 0.5),
            risk_budget_fraction=fraction,
            thesis_summary="Fallback allocation — LLM planner unavailable.",
            expected_hold_horizon=horizon_map.get(sym, "intraday"),
            direction_bias="neutral",
        )
        for sym in symbols
    ]

    return SessionIntent(
        selected_symbols=symbols,
        symbol_intents=symbol_intents,
        total_risk_budget_fraction=1.0,
        regime_summary=regime_summary,
        planner_rationale="Fallback: user-specified symbols with equal allocation.",
        is_fallback=True,
        use_ai_planner=True,
    )


def _parse_llm_response(raw: str, cards: List[OpportunityCard]) -> SessionIntent:
    """Parse and validate the LLM's JSON response into a SessionIntent.

    Raises ValueError on parse/validation failures so the caller can fall back.
    """
    # Strip markdown fences if present
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(
            line for line in lines
            if not line.startswith("```")
        ).strip()

    data = json.loads(text)

    # Validate symbols against the card set
    card_symbols = {c.symbol for c in cards}
    selected = [s for s in data.get("selected_symbols", []) if s in card_symbols]
    if not selected:
        raise ValueError("LLM selected no valid symbols from the opportunity ranking")

    # Validate symbol_intents
    intent_data = data.get("symbol_intents", [])
    symbol_intents = []
    for item in intent_data:
        sym = item.get("symbol", "")
        if sym not in card_symbols:
            continue
        symbol_intents.append(SymbolIntent(**item))

    # If LLM omitted symbol_intents, build from selected with equal allocation
    if not symbol_intents:
        score_map = {c.symbol: c.opportunity_score_norm for c in cards}
        horizon_map = {c.symbol: c.expected_hold_horizon for c in cards}
        n = max(1, len(selected))
        fraction = round(1.0 / n, 4)
        for sym in selected:
            symbol_intents.append(SymbolIntent(
                symbol=sym,
                opportunity_score_norm=score_map.get(sym, 0.5),
                risk_budget_fraction=fraction,
                thesis_summary="See planner rationale.",
                expected_hold_horizon=horizon_map.get(sym, "intraday"),
                direction_bias="neutral",
            ))

    return SessionIntent(
        selected_symbols=selected,
        symbol_intents=symbol_intents,
        total_risk_budget_fraction=1.0,
        regime_summary=data.get("regime_summary", ""),
        planner_rationale=data.get("planner_rationale", ""),
        planned_trade_cadence_min=int(data.get("planned_trade_cadence_min", 5)),
        planned_trade_cadence_max=int(data.get("planned_trade_cadence_max", 10)),
        use_ai_planner=True,
        is_fallback=False,
    )


async def generate_session_intent(
    opportunity_ranking: List[OpportunityCard],
    portfolio_state: Dict[str, Any],
    session_config: Dict[str, Any],
    fallback_symbols: List[str],
    llm_model: Optional[str] = None,
) -> SessionIntent:
    """Generate a SessionIntent using the AI portfolio planner.

    Args:
        opportunity_ranking: Sorted list of OpportunityCards (highest score first).
            Typically from OpportunityScanner.score_symbols().
        portfolio_state: Dict with keys: cash, positions, equity.
        session_config: Dict with keys: screener_regime, indicator_timeframe,
            direction_bias, and any other session-level config.
        fallback_symbols: Symbols to use when LLM call fails.
        llm_model: Optional model override. Defaults to OPENAI_MODEL env var.

    Returns:
        SessionIntent — either LLM-generated or fallback.
    """
    # Take top N cards for the LLM
    top_cards = opportunity_ranking[:_TOP_N_FOR_LLM]

    if not top_cards:
        logger.warning("session_planner: no opportunity cards available, using fallback")
        return _fallback_intent(fallback_symbols, [], regime_summary="no opportunity data")

    opp_block = _build_opportunity_block(top_cards)
    port_block = _build_portfolio_block(portfolio_state, session_config)

    user_message = f"{opp_block}\n\n{port_block}\n\nGenerate the session intent JSON."

    # Attempt LLM call
    try:
        import openai

        _model = llm_model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        client = openai.AsyncOpenAI()

        response = await client.chat.completions.create(
            model=_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,
            max_tokens=1024,
            timeout=30,
        )
        raw = response.choices[0].message.content or ""
        intent = _parse_llm_response(raw, top_cards)
        logger.info(
            "session_planner: generated intent via LLM — symbols=%s is_fallback=%s",
            intent.selected_symbols,
            intent.is_fallback,
        )
        return intent

    except Exception as exc:
        logger.warning(
            "session_planner: LLM call failed (%s), using fallback for symbols=%s",
            exc,
            fallback_symbols,
        )
        regime_hint = session_config.get("screener_regime") or ""
        return _fallback_intent(
            fallback_symbols or [c.symbol for c in top_cards[:_SELECT_K]],
            top_cards,
            regime_summary=regime_hint,
        )


async def build_session_intent_from_indicator_snapshots(
    symbols: List[str],
    indicator_snapshots_raw: Dict[str, Any],
    portfolio_state: Dict[str, Any],
    session_config: Dict[str, Any],
    llm_model: Optional[str] = None,
) -> Optional[SessionIntent]:
    """Build a SessionIntent directly from serialized indicator snapshots.

    This is the shared service-layer entrypoint used by the API before workflow
    start and by any explicit operator refresh path. It keeps SessionIntent
    generation out of workflow branching.
    """
    try:
        from datetime import datetime as _dt
        from schemas.llm_strategist import IndicatorSnapshot as _IndicatorSnapshot
        from services.opportunity_scanner import rank_universe as _rank_universe

        snapshots: Dict[str, Any] = {}
        timeframe = session_config.get("indicator_timeframe", "1h")
        for sym, raw in indicator_snapshots_raw.items():
            if not isinstance(raw, dict):
                continue
            try:
                snapshots[sym] = _IndicatorSnapshot.model_validate({
                    **raw,
                    "symbol": sym,
                    "timeframe": timeframe,
                    "as_of": raw.get("as_of", _dt.now(timezone.utc)),
                })
            except Exception:
                logger.debug("session_planner: invalid snapshot skipped for %s", sym, exc_info=True)

        ranking = _rank_universe(
            symbols=list(snapshots.keys()) or symbols,
            indicator_snapshots=snapshots,
            top_n=10,
        )
        return await generate_session_intent(
            opportunity_ranking=ranking.cards,
            portfolio_state=portfolio_state,
            session_config=session_config,
            fallback_symbols=symbols,
            llm_model=llm_model,
        )
    except Exception as exc:
        logger.warning("session_planner: failed to build session intent from snapshots: %s", exc)
        return None
