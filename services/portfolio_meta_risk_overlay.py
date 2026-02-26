"""Deterministic portfolio meta-risk overlay evaluator (Runbook 60 Phase M4).

Evaluates portfolio-level risk conditions against an active PortfolioMetaRiskPolicy
and returns the preapproved actions that should be executed.

All functions are pure and deterministic — no LLM calls, no I/O.
The caller supplies resolved portfolio metrics; this module checks each enabled
condition against the policy thresholds and returns triggered actions.

Three exit classes remain distinct:
  1. PositionExitContract  — per-position, strategy-authored, at-entry
  2. PortfolioMetaRiskPolicy — portfolio-wide, deterministic overlay (THIS MODULE)
  3. emergency_exit           — system/market safety interrupt (separate path)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from schemas.position_exit_contract import PortfolioMetaAction, PortfolioMetaRiskPolicy

logger = logging.getLogger(__name__)

OVERLAY_EVALUATOR_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Portfolio metrics snapshot (input to evaluator)
# ---------------------------------------------------------------------------


@dataclass
class PortfolioMetrics:
    """Resolved portfolio-level metrics for overlay condition evaluation.

    Callers compute these from live portfolio state before calling
    evaluate_portfolio_risk_conditions().
    """

    total_equity: float = 0.0
    """Total portfolio equity (cash + mark-to-market positions)."""

    gross_exposure_pct: float = 0.0
    """Gross exposure as % of equity (sum of abs position values / equity × 100)."""

    portfolio_drawdown_pct: float = 0.0
    """Current portfolio drawdown from high-water mark, as a positive percentage."""

    symbol_concentrations: Dict[str, float] = field(default_factory=dict)
    """Per-symbol concentration as % of equity. Key = symbol, value = % (0–100)."""

    cluster_concentrations: Dict[str, float] = field(default_factory=dict)
    """Per-cluster concentration as % of equity. Key = cluster_id, value = % (0–100)."""

    max_pairwise_correlation: float = 0.0
    """Maximum observed pairwise correlation across open positions (0–1)."""

    current_regime: Optional[str] = None
    """Current regime label (e.g., 'hostile', 'uncertain'). Used for hostile_regime_reduce."""


# ---------------------------------------------------------------------------
# Overlay evaluation result
# ---------------------------------------------------------------------------


@dataclass
class OverlayConditionFired:
    """A single condition that fired, with the triggering metric value."""

    condition_id: str
    condition_kind: str
    threshold: float
    observed_value: float
    detail: str


@dataclass
class OverlayEvaluationResult:
    """Result of evaluating all conditions in a PortfolioMetaRiskPolicy."""

    policy_id: str
    conditions_fired: List[OverlayConditionFired] = field(default_factory=list)
    triggered_actions: List[PortfolioMetaAction] = field(default_factory=list)

    # Audit snapshot (what was passed in)
    metrics_snapshot: Dict[str, Any] = field(default_factory=dict)

    @property
    def any_condition_fired(self) -> bool:
        return bool(self.conditions_fired)


# ---------------------------------------------------------------------------
# Condition identifiers (match condition_id in PortfolioMetaAction)
# ---------------------------------------------------------------------------

COND_SYMBOL_CONCENTRATION = "symbol_concentration_exceeded"
COND_CLUSTER_CONCENTRATION = "cluster_concentration_exceeded"
COND_PORTFOLIO_DRAWDOWN = "portfolio_drawdown_exceeded"
COND_CORRELATION_SPIKE = "correlation_spike"
COND_HOSTILE_REGIME = "hostile_regime"

_HOSTILE_REGIME_LABELS = frozenset({"hostile", "crash", "extreme_bear"})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate_portfolio_risk_conditions(
    policy: PortfolioMetaRiskPolicy,
    metrics: PortfolioMetrics,
) -> OverlayEvaluationResult:
    """Evaluate all enabled conditions in ``policy`` against ``metrics``.

    Returns an OverlayEvaluationResult containing:
    - every condition that fired (with observed vs threshold)
    - the preapproved PortfolioMetaActions linked to fired conditions

    Only actions whose ``condition_id`` matches a fired condition are included.
    The overlay may NOT invent arbitrary actions at runtime.

    Args:
        policy:  Active PortfolioMetaRiskPolicy with thresholds and action registry.
        metrics: Resolved portfolio metrics (caller computes these from live state).

    Returns:
        OverlayEvaluationResult (never raises; invalid/disabled conditions are skipped).
    """
    result = OverlayEvaluationResult(
        policy_id=policy.policy_id,
        metrics_snapshot={
            "total_equity": metrics.total_equity,
            "gross_exposure_pct": metrics.gross_exposure_pct,
            "portfolio_drawdown_pct": metrics.portfolio_drawdown_pct,
            "max_pairwise_correlation": metrics.max_pairwise_correlation,
            "current_regime": metrics.current_regime,
            "symbol_concentrations": dict(metrics.symbol_concentrations),
            "cluster_concentrations": dict(metrics.cluster_concentrations),
        },
    )

    if not policy.enabled:
        return result

    fired_condition_ids: set[str] = set()

    # ----------------------------------------------------------------
    # Condition 1: symbol concentration
    # ----------------------------------------------------------------
    if policy.max_symbol_concentration_pct is not None:
        threshold = policy.max_symbol_concentration_pct
        for symbol, conc_pct in metrics.symbol_concentrations.items():
            if conc_pct > threshold:
                cond = OverlayConditionFired(
                    condition_id=COND_SYMBOL_CONCENTRATION,
                    condition_kind="max_symbol_concentration_pct",
                    threshold=threshold,
                    observed_value=conc_pct,
                    detail=(
                        f"Symbol {symbol} concentration {conc_pct:.1f}% exceeds "
                        f"max {threshold:.1f}%"
                    ),
                )
                result.conditions_fired.append(cond)
                fired_condition_ids.add(COND_SYMBOL_CONCENTRATION)
                logger.debug(cond.detail)
                break  # only fire once per policy check cycle

    # ----------------------------------------------------------------
    # Condition 2: cluster/sector concentration
    # ----------------------------------------------------------------
    if policy.max_sector_or_cluster_concentration_pct is not None:
        threshold = policy.max_sector_or_cluster_concentration_pct
        for cluster_id, conc_pct in metrics.cluster_concentrations.items():
            if conc_pct > threshold:
                cond = OverlayConditionFired(
                    condition_id=COND_CLUSTER_CONCENTRATION,
                    condition_kind="max_sector_or_cluster_concentration_pct",
                    threshold=threshold,
                    observed_value=conc_pct,
                    detail=(
                        f"Cluster {cluster_id} concentration {conc_pct:.1f}% exceeds "
                        f"max {threshold:.1f}%"
                    ),
                )
                result.conditions_fired.append(cond)
                fired_condition_ids.add(COND_CLUSTER_CONCENTRATION)
                logger.debug(cond.detail)
                break

    # ----------------------------------------------------------------
    # Condition 3: portfolio drawdown
    # ----------------------------------------------------------------
    if policy.portfolio_drawdown_reduce_threshold_pct is not None:
        threshold = policy.portfolio_drawdown_reduce_threshold_pct
        observed = metrics.portfolio_drawdown_pct
        if observed > threshold:
            cond = OverlayConditionFired(
                condition_id=COND_PORTFOLIO_DRAWDOWN,
                condition_kind="portfolio_drawdown_reduce_threshold_pct",
                threshold=threshold,
                observed_value=observed,
                detail=(
                    f"Portfolio drawdown {observed:.1f}% exceeds "
                    f"threshold {threshold:.1f}%"
                ),
            )
            result.conditions_fired.append(cond)
            fired_condition_ids.add(COND_PORTFOLIO_DRAWDOWN)
            logger.warning(cond.detail)

    # ----------------------------------------------------------------
    # Condition 4: correlation spike
    # ----------------------------------------------------------------
    if policy.correlation_reduce_threshold is not None:
        threshold = policy.correlation_reduce_threshold
        observed = metrics.max_pairwise_correlation
        if observed > threshold:
            cond = OverlayConditionFired(
                condition_id=COND_CORRELATION_SPIKE,
                condition_kind="correlation_reduce_threshold",
                threshold=threshold,
                observed_value=observed,
                detail=(
                    f"Max pairwise correlation {observed:.3f} exceeds "
                    f"threshold {threshold:.3f}"
                ),
            )
            result.conditions_fired.append(cond)
            fired_condition_ids.add(COND_CORRELATION_SPIKE)
            logger.warning(cond.detail)

    # ----------------------------------------------------------------
    # Condition 5: hostile regime
    # ----------------------------------------------------------------
    if policy.hostile_regime_reduce_enabled and metrics.current_regime is not None:
        if metrics.current_regime.lower() in _HOSTILE_REGIME_LABELS:
            cond = OverlayConditionFired(
                condition_id=COND_HOSTILE_REGIME,
                condition_kind="hostile_regime_reduce_enabled",
                threshold=1.0,
                observed_value=1.0,
                detail=f"Regime '{metrics.current_regime}' is hostile; overlay triggered",
            )
            result.conditions_fired.append(cond)
            fired_condition_ids.add(COND_HOSTILE_REGIME)
            logger.warning(cond.detail)

    # ----------------------------------------------------------------
    # Collect preapproved actions for fired conditions
    # ----------------------------------------------------------------
    for action in policy.actions:
        if action.condition_id in fired_condition_ids:
            result.triggered_actions.append(action)

    if result.triggered_actions:
        action_kinds = [a.kind for a in result.triggered_actions]
        logger.info(
            "Portfolio overlay: %d condition(s) fired, %d action(s) triggered: %s",
            len(result.conditions_fired),
            len(result.triggered_actions),
            action_kinds,
        )

    return result


def compute_symbol_concentration(
    positions: Dict[str, float],
    prices: Dict[str, float],
    total_equity: float,
) -> Dict[str, float]:
    """Compute per-symbol concentration as % of equity.

    Args:
        positions:    {symbol: quantity} from portfolio state.
        prices:       {symbol: mark_price} for marking positions.
        total_equity: Total portfolio equity (cash + positions mark-to-market).

    Returns:
        {symbol: concentration_pct} where concentration_pct is 0–100.
        Returns empty dict when total_equity <= 0.
    """
    if total_equity <= 0:
        return {}
    result: Dict[str, float] = {}
    for symbol, qty in positions.items():
        if qty == 0:
            continue
        price = prices.get(symbol, 0.0)
        if price <= 0:
            continue
        notional = abs(qty) * price
        result[symbol] = (notional / total_equity) * 100.0
    return result
