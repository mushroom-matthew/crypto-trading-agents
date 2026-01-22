"""Simple JSON-backed storage for planner outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


class StrategyConfigStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or Path("data/strategy_configs.json")
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load_all(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {}
        with self.path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def save(self, symbol: str, payload: Dict[str, Any]) -> None:
        data = self._load_all()
        data[symbol] = payload
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, symbol: str) -> Optional[Dict[str, Any]]:
        data = self._load_all()
        return data.get(symbol)

    def load_all(self) -> Dict[str, Any]:
        return self._load_all()


store = StrategyConfigStore()


def save_plan(symbol: str, plan: Dict[str, Any]) -> None:
    store.save(symbol, plan)


def load_plan(symbol: str) -> Optional[Dict[str, Any]]:
    return store.load(symbol)


def load_all_plans() -> Dict[str, Any]:
    return store.load_all()
