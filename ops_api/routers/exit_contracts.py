"""Ops API router for position exit contracts and portfolio meta-risk overlay (Runbook 60).

Exposes active exit contracts and overlay actions for operator inspection.
Allows distinguishing three exit classes in telemetry:
  - strategy_contract (per-position, precommitted)
  - portfolio_overlay  (portfolio-wide policy actions)
  - emergency          (safety interrupt)
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/exit-contracts", tags=["exit-contracts"])

TASK_QUEUE = os.environ.get("TASK_QUEUE", "mcp-tools")


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class ExitLegResponse(BaseModel):
    leg_id: str
    kind: str
    trigger_mode: str
    fraction: float
    price_abs: Optional[float] = None
    r_multiple: Optional[float] = None
    priority: int = 0
    enabled: bool = True
    fired: bool = False


class TimeExitRuleResponse(BaseModel):
    max_hold_bars: Optional[int] = None
    max_hold_minutes: Optional[int] = None
    session_boundary_action: str = "reassess"


class PositionExitContractResponse(BaseModel):
    contract_id: str
    contract_version: str
    position_id: str
    symbol: str
    side: str
    created_at: str
    entry_price: float
    initial_qty: float
    stop_price_abs: float
    target_legs: List[ExitLegResponse] = []
    time_exit: Optional[TimeExitRuleResponse] = None
    amendment_policy: str = "tighten_only"
    allow_portfolio_overlay_trims: bool = True
    remaining_qty: Optional[float] = None
    source_trigger_id: str
    source_category: Optional[str] = None
    playbook_id: Optional[str] = None
    template_id: Optional[str] = None
    # Computed helpers (populated server-side)
    stop_r_distance: Optional[float] = None
    has_price_target: bool = False
    active_legs_count: int = 0
    exit_class: str = "strategy_contract"


class ExitContractsSessionResponse(BaseModel):
    session_id: str
    contracts: List[PositionExitContractResponse]
    total_count: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _contract_dict_to_response(contract_dict: Dict[str, Any]) -> PositionExitContractResponse:
    """Convert a raw contract dict (from workflow state) to API response model."""
    legs = contract_dict.get("target_legs", [])
    time_exit_raw = contract_dict.get("time_exit")

    entry = float(contract_dict.get("entry_price", 0.0))
    stop = float(contract_dict.get("stop_price_abs", 0.0))
    stop_r = abs(entry - stop) if entry and stop else None

    active_legs = [
        leg for leg in legs if leg.get("enabled", True) and not leg.get("fired", False)
    ]
    has_price_target = any(
        leg.get("trigger_mode") in ("price_level", "r_multiple") for leg in active_legs
    )

    created_at = contract_dict.get("created_at", "")
    if hasattr(created_at, "isoformat"):
        created_at = created_at.isoformat()

    return PositionExitContractResponse(
        contract_id=contract_dict.get("contract_id", ""),
        contract_version=contract_dict.get("contract_version", "1.0.0"),
        position_id=contract_dict.get("position_id", ""),
        symbol=contract_dict.get("symbol", ""),
        side=contract_dict.get("side", "long"),
        created_at=str(created_at),
        entry_price=entry,
        initial_qty=float(contract_dict.get("initial_qty", 0.0)),
        stop_price_abs=stop,
        target_legs=[
            ExitLegResponse(
                leg_id=leg.get("leg_id", ""),
                kind=leg.get("kind", "full_exit"),
                trigger_mode=leg.get("trigger_mode", "price_level"),
                fraction=float(leg.get("fraction", 1.0)),
                price_abs=leg.get("price_abs"),
                r_multiple=leg.get("r_multiple"),
                priority=int(leg.get("priority", 0)),
                enabled=bool(leg.get("enabled", True)),
                fired=bool(leg.get("fired", False)),
            )
            for leg in legs
        ],
        time_exit=TimeExitRuleResponse(**time_exit_raw) if time_exit_raw else None,
        amendment_policy=contract_dict.get("amendment_policy", "tighten_only"),
        allow_portfolio_overlay_trims=bool(contract_dict.get("allow_portfolio_overlay_trims", True)),
        remaining_qty=contract_dict.get("remaining_qty"),
        source_trigger_id=contract_dict.get("source_trigger_id", ""),
        source_category=contract_dict.get("source_category"),
        playbook_id=contract_dict.get("playbook_id"),
        template_id=contract_dict.get("template_id"),
        stop_r_distance=stop_r,
        has_price_target=has_price_target,
        active_legs_count=len(active_legs),
        exit_class="strategy_contract",
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/sessions/{session_id}",
    response_model=ExitContractsSessionResponse,
    summary="List active position exit contracts for a paper trading session",
)
async def get_session_exit_contracts(session_id: str) -> ExitContractsSessionResponse:
    """Return all active PositionExitContracts for the given paper trading session.

    Queries the PaperTradingWorkflow state (exit_contracts dict, keyed by symbol).
    Returns an empty list if the session has no active contracts or doesn't exist.

    Exit class for all contracts here is ``strategy_contract`` — distinct from
    ``portfolio_overlay`` and ``emergency`` exit classes.
    """
    try:
        from temporalio.client import Client  # noqa: PLC0415
        client = await Client.connect(
            os.environ.get("TEMPORAL_ADDRESS", "localhost:7233"),
            namespace=os.environ.get("TEMPORAL_NAMESPACE", "default"),
        )
        handle = client.get_workflow_handle(session_id)
        state: Dict[str, Any] = await handle.query("get_session_status")
    except Exception as exc:
        err_str = str(exc).lower()
        if "not found" in err_str or "does not exist" in err_str:
            return ExitContractsSessionResponse(
                session_id=session_id,
                contracts=[],
                total_count=0,
            )
        logger.warning("Failed to query session %s for exit contracts: %s", session_id, exc)
        raise HTTPException(status_code=502, detail=f"Workflow query failed: {exc}") from exc

    exit_contracts_raw: Dict[str, Any] = state.get("exit_contracts", {})
    contracts = [
        _contract_dict_to_response(contract_dict)
        for contract_dict in exit_contracts_raw.values()
        if isinstance(contract_dict, dict)
    ]
    return ExitContractsSessionResponse(
        session_id=session_id,
        contracts=contracts,
        total_count=len(contracts),
    )


@router.get(
    "/sessions/{session_id}/{symbol}",
    response_model=Optional[PositionExitContractResponse],
    summary="Get active exit contract for a specific symbol in a session",
)
async def get_symbol_exit_contract(
    session_id: str,
    symbol: str,
) -> Optional[PositionExitContractResponse]:
    """Return the active PositionExitContract for a specific symbol in a session.

    Returns null (404) if no contract exists for that symbol.
    """
    try:
        from temporalio.client import Client  # noqa: PLC0415
        client = await Client.connect(
            os.environ.get("TEMPORAL_ADDRESS", "localhost:7233"),
            namespace=os.environ.get("TEMPORAL_NAMESPACE", "default"),
        )
        handle = client.get_workflow_handle(session_id)
        state: Dict[str, Any] = await handle.query("get_session_status")
    except Exception as exc:
        err_str = str(exc).lower()
        if "not found" in err_str or "does not exist" in err_str:
            raise HTTPException(status_code=404, detail="Session not found") from exc
        raise HTTPException(status_code=502, detail=f"Workflow query failed: {exc}") from exc

    exit_contracts_raw: Dict[str, Any] = state.get("exit_contracts", {})
    contract_dict = exit_contracts_raw.get(symbol)
    if contract_dict is None:
        raise HTTPException(
            status_code=404,
            detail=f"No active exit contract for symbol '{symbol}' in session '{session_id}'",
        )
    return _contract_dict_to_response(contract_dict)
