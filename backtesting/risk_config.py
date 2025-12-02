"""Helpers for injecting risk limits into backtests from CLI or config files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping

from schemas.strategy_run import RiskLimitSettings


def _load_structured_config(path: Path) -> Dict[str, Any]:
    """Load a dict from JSON or YAML file."""

    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - only hit when yaml missing
            raise RuntimeError("PyYAML is required to parse YAML risk configs") from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Risk config at {path} must be a JSON/YAML object")
    return data


def _filtered_payload(data: Mapping[str, Any]) -> Dict[str, float]:
    """Return only recognized risk limit keys from the provided mapping."""

    allowed = {
        "max_position_risk_pct",
        "max_symbol_exposure_pct",
        "max_portfolio_exposure_pct",
        "max_daily_loss_pct",
    }
    payload: Dict[str, float] = {}
    source = data.get("risk_limits") if isinstance(data.get("risk_limits"), Mapping) else data
    if not isinstance(source, Mapping):
        return payload
    for key in allowed:
        value = source.get(key)
        if value is None:
            continue
        payload[key] = float(value)
    return payload


def resolve_risk_limits(config_path: Path | None, cli_overrides: Mapping[str, float | None]) -> RiskLimitSettings:
    """Return RiskLimitSettings derived from an optional config file and CLI overrides.

    The resolution order is:
    1. Base defaults from RiskLimitSettings() (sane defaults).
    2. Values loaded from `config_path` if provided (JSON or YAML). Configs may either
       be a bare dict with the risk keys or contain a nested `risk_limits` mapping.
    3. CLI overrides, where any non-None value supersedes the merged config/default.
    """

    limits = RiskLimitSettings()
    if config_path:
        data = _load_structured_config(config_path)
        payload = _filtered_payload(data)
        if payload:
            limits = limits.model_copy(update=payload)

    cli_payload: MutableMapping[str, float] = {}
    for key, value in cli_overrides.items():
        if value is None:
            continue
        cli_payload[key] = float(value)
    if cli_payload:
        limits = limits.model_copy(update=dict(cli_payload))
    return limits


__all__ = ["resolve_risk_limits"]
