from __future__ import annotations

from pathlib import Path
import sys

import json

import pytest

from backtesting.risk_config import resolve_risk_limits


def test_resolve_risk_limits_from_json(tmp_path):
    payload = {
        "risk_limits": {
            "max_position_risk_pct": 1.5,
            "max_symbol_exposure_pct": 10.0,
            "max_portfolio_exposure_pct": 40.0,
        }
    }
    path = tmp_path / "risk.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    limits = resolve_risk_limits(path, {"max_daily_loss_pct": None})
    assert limits.max_position_risk_pct == pytest.approx(1.5)
    assert limits.max_symbol_exposure_pct == pytest.approx(10.0)
    assert limits.max_portfolio_exposure_pct == pytest.approx(40.0)
    assert limits.max_daily_loss_pct == pytest.approx(3.0)  # default


def test_cli_overrides_take_priority(tmp_path):
    payload = {"max_position_risk_pct": 2.5, "max_daily_loss_pct": 4.0}
    path = tmp_path / "risk.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    limits = resolve_risk_limits(
        path,
        {
            "max_position_risk_pct": 0.5,
            "max_symbol_exposure_pct": None,
            "max_portfolio_exposure_pct": None,
            "max_daily_loss_pct": 2.0,
        },
    )
    assert limits.max_position_risk_pct == pytest.approx(0.5)
    assert limits.max_daily_loss_pct == pytest.approx(2.0)


def test_yaml_requires_dependency(tmp_path, monkeypatch):
    yaml_path = tmp_path / "risk.yaml"
    yaml_path.write_text("max_position_risk_pct: 3", encoding="utf-8")
    monkeypatch.setitem(sys.modules, "yaml", None)
    with pytest.raises(RuntimeError):
        resolve_risk_limits(yaml_path, {})
