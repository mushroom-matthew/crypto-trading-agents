"""Central prompt/template metadata and active-path helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"
STRATEGIES_DIR = PROMPTS_DIR / "strategies"


@dataclass(frozen=True)
class StrategyTemplateMeta:
    id: str
    name: str
    description: str
    status: str = "active"
    visible_in_ui: bool = True


STRATEGY_TEMPLATES: dict[str, StrategyTemplateMeta] = {
    "aggressive_active": StrategyTemplateMeta(
        id="aggressive_active",
        name="Aggressive / Active",
        description="Legacy broad active-trading template. Hidden by default while the prompt surface is pruned.",
        status="legacy",
        visible_in_ui=False,
    ),
    "balanced_hybrid": StrategyTemplateMeta(
        id="balanced_hybrid",
        name="Balanced / Hybrid",
        description="Legacy broad hybrid template. Hidden by default while the prompt surface is pruned.",
        status="legacy",
        visible_in_ui=False,
    ),
    "bear_defensive": StrategyTemplateMeta(
        id="bear_defensive",
        name="Bear Defensive",
        description="Short-only defensive template for bear-regime experiments.",
        status="flagged",
        visible_in_ui=False,
    ),
    "compression_breakout": StrategyTemplateMeta(
        id="compression_breakout",
        name="Compression Breakout",
        description="Legacy generic breakout template. Directional variants should replace it.",
        status="legacy",
        visible_in_ui=False,
    ),
    "compression_breakout_long": StrategyTemplateMeta(
        id="compression_breakout_long",
        name="Compression Breakout Long",
        description="Directional long breakout template. Hidden until refreshed against the current schema.",
        status="flagged",
        visible_in_ui=False,
    ),
    "compression_breakout_short": StrategyTemplateMeta(
        id="compression_breakout_short",
        name="Compression Breakout Short",
        description="Directional short breakout template. Hidden until refreshed against the current schema.",
        status="flagged",
        visible_in_ui=False,
    ),
    "conservative_defensive": StrategyTemplateMeta(
        id="conservative_defensive",
        name="Conservative / Defensive",
        description="Legacy defensive archetype. Hidden by default while broad prompts are de-emphasized.",
        status="legacy",
        visible_in_ui=False,
    ),
    "mean_reversion": StrategyTemplateMeta(
        id="mean_reversion",
        name="Mean Reversion",
        description="Legacy broad mean-reversion template. Range-specific templates should replace it.",
        status="legacy",
        visible_in_ui=False,
    ),
    "momentum_trend_following": StrategyTemplateMeta(
        id="momentum_trend_following",
        name="Momentum / Trend Following",
        description="Legacy broad trend template. Hidden by default while specialized prompts are favored.",
        status="legacy",
        visible_in_ui=False,
    ),
    "range_long": StrategyTemplateMeta(
        id="range_long",
        name="Range Long",
        description="Buy the dip in a confirmed range and target the mean.",
    ),
    "range_short": StrategyTemplateMeta(
        id="range_short",
        name="Range Short",
        description="Sell the rally in a confirmed range and target the mean.",
    ),
    "scalper_fast": StrategyTemplateMeta(
        id="scalper_fast",
        name="Scalper / Fast",
        description="Fast-timeframe template for VWAP, fast EMA, and volume-burst driven trading.",
    ),
    "volatile_breakout_long": StrategyTemplateMeta(
        id="volatile_breakout_long",
        name="Volatile Breakout Long",
        description="Directional high-volatility breakout template. Hidden until refreshed against the current schema.",
        status="flagged",
        visible_in_ui=False,
    ),
    "volatile_breakout_short": StrategyTemplateMeta(
        id="volatile_breakout_short",
        name="Volatile Breakout Short",
        description="Directional high-volatility breakout template. Hidden until refreshed against the current schema.",
        status="flagged",
        visible_in_ui=False,
    ),
    "volatility_breakout": StrategyTemplateMeta(
        id="volatility_breakout",
        name="Volatility Breakout",
        description="Legacy generic breakout template. Hidden by default while the prompt surface is pruned.",
        status="legacy",
        visible_in_ui=False,
    ),
    "default": StrategyTemplateMeta(
        id="default",
        name="Default (Current)",
        description="The strategist base prompt currently used by the runtime.",
    ),
}


def current_strategist_prompt_path() -> Path:
    """Return the strategist prompt file used by the runtime by default."""
    prompt_name = os.environ.get("STRATEGIST_PROMPT", "simple").strip().lower()
    if prompt_name == "full":
        return PROMPTS_DIR / "llm_strategist_prompt.txt"
    return PROMPTS_DIR / "llm_strategist_simple.txt"


def current_strategist_prompt_id() -> str:
    """Return the strategist prompt mode used by the runtime by default."""
    prompt_name = os.environ.get("STRATEGIST_PROMPT", "simple").strip().lower()
    return "full" if prompt_name == "full" else "simple"


def show_hidden_templates() -> bool:
    return os.environ.get("PROMPT_SHOW_HIDDEN_STRATEGIES", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def strategy_template_meta(strategy_id: str) -> StrategyTemplateMeta | None:
    return STRATEGY_TEMPLATES.get(strategy_id)


def strategy_template_visible(strategy_id: str) -> bool:
    meta = strategy_template_meta(strategy_id)
    if meta is None:
        return True
    return meta.visible_in_ui or show_hidden_templates()
