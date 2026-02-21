"""API router for LLM prompt management."""

from __future__ import annotations

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/prompts", tags=["prompts"])

# Prompt file paths
PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"
VERSIONS_DIR = PROMPTS_DIR / "versions"
STRATEGIES_DIR = PROMPTS_DIR / "strategies"
STRATEGIST_PROMPT_FILE = PROMPTS_DIR / "llm_strategist_prompt.txt"
JUDGE_PROMPT_FILE = PROMPTS_DIR / "llm_judge_prompt.txt"


class PromptResponse(BaseModel):
    """Response model for prompt retrieval."""
    name: str
    content: str
    file_path: str
    version: Optional[str] = None


class PromptUpdateRequest(BaseModel):
    """Request model for prompt update."""
    content: str


class PromptListResponse(BaseModel):
    """Response model for listing available prompts."""
    prompts: list[str]


class PromptVersion(BaseModel):
    """A single prompt version."""
    version_id: str
    timestamp: str
    file_path: str
    size_bytes: int


class PromptVersionsResponse(BaseModel):
    """Response model for listing prompt versions."""
    name: str
    versions: List[PromptVersion]


class StrategyInfo(BaseModel):
    """Information about a strategy template."""
    id: str
    name: str
    description: str
    file_path: str


class StrategiesListResponse(BaseModel):
    """Response model for listing available strategies."""
    strategies: List[StrategyInfo]


# Strategy metadata - maps file names to display info
STRATEGY_METADATA = {
    "momentum_trend_following": {
        "name": "Momentum / Trend Following",
        "description": "Ride strong trends with wide stops, let winners run. Best for trending markets.",
    },
    "mean_reversion": {
        "name": "Mean Reversion",
        "description": "Buy oversold, sell overbought with quick profits at the mean. Best for ranging markets.",
    },
    "volatility_breakout": {
        "name": "Volatility Breakout",
        "description": "Trade range expansions and squeeze breakouts. Best after consolidation periods.",
    },
    "conservative_defensive": {
        "name": "Conservative / Defensive",
        "description": "Capital preservation focus with strict filters and small positions. Lower risk.",
    },
    "aggressive_active": {
        "name": "Aggressive / Active",
        "description": "Many trades in both directions with higher risk tolerance. Higher potential returns.",
    },
    "balanced_hybrid": {
        "name": "Balanced / Hybrid",
        "description": "Adapts to market regime with core + tactical positions. Well-rounded approach.",
    },
    "compression_breakout": {
        "name": "Compression Breakout",
        "description": "Wait for BB bandwidth compression, then trade the expansion breakout with volume confirmation. Ideal for range/consolidation markets.",
    },
    "default": {
        "name": "Default (Current)",
        "description": "The current strategist prompt with full feature set.",
    },
}


# Default judge prompt (extracted from judge_agent_client.py)
DEFAULT_JUDGE_PROMPT = """You are an expert performance judge for a multi-asset crypto strategist. Use the factual data below to critique execution quality, trigger discipline, volatility handling, and adaptability.

PERFORMANCE SNAPSHOT:
{performance_summary}

RECENT TRANSACTIONS:
{transaction_summary}

TRADING STATISTICS:
- Total transactions: {total_transactions}
- Buy orders: {buy_count}
- Sell orders: {sell_count}
- Symbols traded: {symbols}

MARKET-STRUCTURE TELEMETRY (DRAFT):
- When payloads include market_structure (nearest_support/resistance, distances, trend, recent_tests), use it to judge if fills were taken into strength or weakness.
- Reduce allowed risk or cap triggers when price is mid-range without a reclaim/test edge, or when support/resistance has been tapped repeatedly without resolution.
- Reward plans that enter near defended support in uptrends or after reclaiming prior resistance; flag chasing into resistance or weak supports as veto candidates.

FACTOR EXPOSURES (DRAFT):
- When factor_exposures are present (beta_market, beta_eth_beta, idiosyncratic_vol), prefer neutral or diversified beta when risk is elevated.
- If auto_hedge_mode == "market", nudge sizing toward beta ≈ 0; otherwise, veto piling into high-beta longs without offsets.

Respond with **two sections only**:

NOTES:
- Provide up to five concise sentences covering momentum/timing, sizing discipline, volatility handling, and any regime corrections needed. Reference concrete metrics above when possible.

JSON:
{
  "score": <float 0-100 representing overall decision quality>,
  "constraints": {
    "max_trades_per_day": <integer or null>,
    "max_triggers_per_symbol_per_day": <integer or null>,
    "risk_mode": "normal" | "conservative" | "emergency",
    "disabled_trigger_ids": ["trigger_id"],
    "disabled_categories": ["trend_continuation", "reversal", "volatility_breakout", "mean_reversion", "emergency_exit", "other"]
  },
  "strategist_constraints": {
    "must_fix": ["text"],
    "vetoes": ["text"],
    "boost": ["text"],
    "regime_correction": "<text or null>",
    "sizing_adjustments": {"SYMBOL": "<instruction>"}
  }
}

Rules:
- Always include every key above even if arrays are empty.
- Use null for unknown numeric limits.
- disabled_trigger_ids should name concrete trigger IDs observed; leave empty if none.
- disabled_categories must use the provided category names.
- Base risk_mode on drawdown/volatility evidence from the snapshot.
- Keep the JSON compact, valid, and free of commentary after the closing brace."""


def _ensure_prompts_dir():
    """Ensure the prompts directory exists."""
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)


def _ensure_versions_dir():
    """Ensure the versions directory exists."""
    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)


def _ensure_judge_prompt_exists():
    """Ensure the judge prompt file exists with default content."""
    _ensure_prompts_dir()
    if not JUDGE_PROMPT_FILE.exists():
        JUDGE_PROMPT_FILE.write_text(DEFAULT_JUDGE_PROMPT)
        logger.info(f"Created default judge prompt at {JUDGE_PROMPT_FILE}")


def _create_version(prompt_name: str, source_file: Path) -> str:
    """Create a versioned backup of the prompt before updating.

    Returns the version ID (timestamp string).
    """
    if not source_file.exists():
        return ""

    _ensure_versions_dir()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    version_filename = f"{prompt_name}_{timestamp}.txt"
    version_path = VERSIONS_DIR / version_filename

    shutil.copy2(source_file, version_path)
    logger.info(f"Created version backup: {version_path}")

    return timestamp


def _get_prompt_file(prompt_name: str) -> Path:
    """Get the file path for a prompt name."""
    if prompt_name == "strategist":
        return STRATEGIST_PROMPT_FILE
    elif prompt_name == "judge":
        return JUDGE_PROMPT_FILE
    else:
        raise ValueError(f"Unknown prompt: {prompt_name}")


@router.get("/", response_model=PromptListResponse)
async def list_prompts() -> PromptListResponse:
    """List available prompts."""
    _ensure_judge_prompt_exists()
    prompts = []
    if STRATEGIST_PROMPT_FILE.exists():
        prompts.append("strategist")
    if JUDGE_PROMPT_FILE.exists():
        prompts.append("judge")
    return PromptListResponse(prompts=prompts)


@router.get("/{prompt_name}", response_model=PromptResponse)
async def get_prompt(prompt_name: str) -> PromptResponse:
    """Get a specific prompt by name.

    Args:
        prompt_name: Either 'strategist' or 'judge'
    """
    _ensure_judge_prompt_exists()

    if prompt_name == "strategist":
        if not STRATEGIST_PROMPT_FILE.exists():
            raise HTTPException(status_code=404, detail="Strategist prompt not found")
        content = STRATEGIST_PROMPT_FILE.read_text()
        return PromptResponse(
            name="strategist",
            content=content,
            file_path=str(STRATEGIST_PROMPT_FILE)
        )
    elif prompt_name == "judge":
        if not JUDGE_PROMPT_FILE.exists():
            raise HTTPException(status_code=404, detail="Judge prompt not found")
        content = JUDGE_PROMPT_FILE.read_text()
        return PromptResponse(
            name="judge",
            content=content,
            file_path=str(JUDGE_PROMPT_FILE)
        )
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown prompt name: {prompt_name}. Valid names are: strategist, judge"
        )


@router.put("/{prompt_name}", response_model=PromptResponse)
async def update_prompt(prompt_name: str, request: PromptUpdateRequest) -> PromptResponse:
    """Update a specific prompt.

    Creates a versioned backup before saving.

    Args:
        prompt_name: Either 'strategist' or 'judge'
        request: The new prompt content
    """
    _ensure_prompts_dir()

    if prompt_name not in ("strategist", "judge"):
        raise HTTPException(
            status_code=400,
            detail=f"Unknown prompt name: {prompt_name}. Valid names are: strategist, judge"
        )

    prompt_file = _get_prompt_file(prompt_name)

    # Create a version backup before overwriting
    version_id = _create_version(prompt_name, prompt_file)

    # Write the new content
    prompt_file.write_text(request.content)
    logger.info(f"Updated {prompt_name} prompt ({len(request.content)} chars), backed up as version {version_id}")

    return PromptResponse(
        name=prompt_name,
        content=request.content,
        file_path=str(prompt_file),
        version=version_id
    )


@router.post("/{prompt_name}/reset", response_model=PromptResponse)
async def reset_prompt(prompt_name: str) -> PromptResponse:
    """Reset a prompt to its default content.

    Args:
        prompt_name: Either 'strategist' or 'judge'
    """
    if prompt_name == "judge":
        _ensure_prompts_dir()
        JUDGE_PROMPT_FILE.write_text(DEFAULT_JUDGE_PROMPT)
        logger.info("Reset judge prompt to default")
        return PromptResponse(
            name="judge",
            content=DEFAULT_JUDGE_PROMPT,
            file_path=str(JUDGE_PROMPT_FILE)
        )
    elif prompt_name == "strategist":
        # For strategist, we can't reset since we don't have a default stored in code
        raise HTTPException(
            status_code=400,
            detail="Cannot reset strategist prompt - no default available. Please restore from git."
        )
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown prompt name: {prompt_name}. Valid names are: strategist, judge"
        )


@router.get("/{prompt_name}/versions", response_model=PromptVersionsResponse)
async def list_versions(prompt_name: str) -> PromptVersionsResponse:
    """List all saved versions of a prompt.

    Args:
        prompt_name: Either 'strategist' or 'judge'
    """
    if prompt_name not in ("strategist", "judge"):
        raise HTTPException(
            status_code=400,
            detail=f"Unknown prompt name: {prompt_name}. Valid names are: strategist, judge"
        )

    _ensure_versions_dir()

    versions = []
    pattern = f"{prompt_name}_*.txt"

    for version_file in sorted(VERSIONS_DIR.glob(pattern), reverse=True):
        # Extract timestamp from filename: {name}_{timestamp}.txt
        filename = version_file.stem  # Remove .txt
        parts = filename.split("_", 1)
        if len(parts) == 2:
            timestamp_str = parts[1]
            # Parse timestamp back to ISO format
            try:
                ts = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                ts = ts.replace(tzinfo=timezone.utc)
                iso_timestamp = ts.isoformat()
            except ValueError:
                iso_timestamp = timestamp_str

            versions.append(PromptVersion(
                version_id=timestamp_str,
                timestamp=iso_timestamp,
                file_path=str(version_file),
                size_bytes=version_file.stat().st_size
            ))

    return PromptVersionsResponse(name=prompt_name, versions=versions)


@router.post("/{prompt_name}/versions/{version_id}/restore", response_model=PromptResponse)
async def restore_version(prompt_name: str, version_id: str) -> PromptResponse:
    """Restore a prompt to a previous version.

    Creates a backup of the current version before restoring.

    Args:
        prompt_name: Either 'strategist' or 'judge'
        version_id: The version timestamp to restore
    """
    if prompt_name not in ("strategist", "judge"):
        raise HTTPException(
            status_code=400,
            detail=f"Unknown prompt name: {prompt_name}. Valid names are: strategist, judge"
        )

    _ensure_versions_dir()

    version_filename = f"{prompt_name}_{version_id}.txt"
    version_path = VERSIONS_DIR / version_filename

    if not version_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Version {version_id} not found for {prompt_name} prompt"
        )

    prompt_file = _get_prompt_file(prompt_name)

    # Create a backup of current version before restoring
    backup_version = _create_version(prompt_name, prompt_file)

    # Restore the old version
    content = version_path.read_text()
    prompt_file.write_text(content)

    logger.info(f"Restored {prompt_name} prompt to version {version_id} (backed up current as {backup_version})")

    return PromptResponse(
        name=prompt_name,
        content=content,
        file_path=str(prompt_file),
        version=version_id
    )


# ============================================================================
# Strategy Templates Endpoints
# ============================================================================

@router.get("/strategies/", response_model=StrategiesListResponse)
async def list_strategies() -> StrategiesListResponse:
    """List all available strategy templates."""
    strategies = []

    # Add default strategy (current prompt)
    if STRATEGIST_PROMPT_FILE.exists():
        meta = STRATEGY_METADATA.get("default", {})
        strategies.append(StrategyInfo(
            id="default",
            name=meta.get("name", "Default"),
            description=meta.get("description", "Current strategist prompt"),
            file_path=str(STRATEGIST_PROMPT_FILE)
        ))

    # Add strategies from the strategies directory
    if STRATEGIES_DIR.exists():
        for strategy_file in sorted(STRATEGIES_DIR.glob("*.txt")):
            strategy_id = strategy_file.stem  # filename without extension
            meta = STRATEGY_METADATA.get(strategy_id, {})
            strategies.append(StrategyInfo(
                id=strategy_id,
                name=meta.get("name", strategy_id.replace("_", " ").title()),
                description=meta.get("description", f"Strategy template: {strategy_id}"),
                file_path=str(strategy_file)
            ))

    return StrategiesListResponse(strategies=strategies)


@router.get("/strategies/{strategy_id}", response_model=PromptResponse)
async def get_strategy(strategy_id: str) -> PromptResponse:
    """Get a specific strategy template by ID.

    Args:
        strategy_id: Strategy identifier (e.g., 'default', 'momentum_trend_following')
    """
    if strategy_id == "default":
        if not STRATEGIST_PROMPT_FILE.exists():
            raise HTTPException(status_code=404, detail="Default strategy not found")
        content = STRATEGIST_PROMPT_FILE.read_text()
        return PromptResponse(
            name="default",
            content=content,
            file_path=str(STRATEGIST_PROMPT_FILE)
        )

    strategy_file = STRATEGIES_DIR / f"{strategy_id}.txt"
    if not strategy_file.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Strategy '{strategy_id}' not found"
        )

    content = strategy_file.read_text()
    meta = STRATEGY_METADATA.get(strategy_id, {})

    return PromptResponse(
        name=meta.get("name", strategy_id),
        content=content,
        file_path=str(strategy_file)
    )
