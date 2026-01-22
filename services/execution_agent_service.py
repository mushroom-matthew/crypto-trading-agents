"""Execution agent service responsible for placing orders on Coinbase."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict


@dataclass
class ExecutionAgentService:
    api_key: str | None = None
    api_secret: str | None = None

    def execute(self, approved_intents: List[dict]) -> List[Dict[str, str]]:
        """Translate approved intents into orders.

        TODO: integrate with real Coinbase client. Currently logs and returns dummy receipts.
        """

        receipts: List[Dict[str, str]] = []
        for intent in approved_intents:
            receipts.append({"symbol": intent["intent"]["symbol"], "status": "queued"})
        return receipts
