"""LLM client facade used by the strategist backtester."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Callable, Protocol

from openai import OpenAI

from schemas.llm_strategist import LLMInput, PositionSizingRule, RiskConstraint, StrategyPlan


class CompletionTransport(Protocol):
    def __call__(self, payload: str) -> str: ...


class LLMClient:
    """Thin wrapper over OpenAI's Responses API with a deterministic fallback."""

    def __init__(self, transport: CompletionTransport | None = None, model: str | None = None) -> None:
        self.transport = transport
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self._client = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI()
        return self._client

    def generate_plan(self, llm_input: LLMInput, prompt_template: str | None = None) -> StrategyPlan:
        if self.transport:
            raw = self.transport(llm_input.to_json())
            return StrategyPlan.from_json(raw)
        try:
            system_prompt = prompt_template or os.environ.get("LLM_STRATEGIST_PROMPT", "")
            completion = self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": llm_input.to_json()},
                ],
                temperature=0.1,
                max_output_tokens=1200,
            )
            content = completion.output[0].content[0].text
            return StrategyPlan.model_validate_json(content)
        except Exception:
            return self._fallback_plan(llm_input)

    def _fallback_plan(self, llm_input: LLMInput) -> StrategyPlan:
        now = datetime.now(timezone.utc)
        risk_dict = llm_input.risk_params
        constraints = RiskConstraint(
            max_position_risk_pct=float(risk_dict.get("max_position_risk_pct", 2.0)),
            max_symbol_exposure_pct=float(risk_dict.get("max_symbol_exposure_pct", 25.0)),
            max_portfolio_exposure_pct=float(risk_dict.get("max_portfolio_exposure_pct", 80.0)),
            max_daily_loss_pct=float(risk_dict.get("max_daily_loss_pct", 3.0)),
        )
        sizing_rules = [
            PositionSizingRule(symbol=asset.symbol, sizing_mode="fixed_fraction", target_risk_pct=constraints.max_position_risk_pct)
            for asset in llm_input.assets
        ]
        return StrategyPlan(
            generated_at=now,
            valid_until=now + timedelta(days=1),
            global_view="fallback plan generated locally",
            regime="range",
            triggers=[],
            risk_constraints=constraints,
            sizing_rules=sizing_rules,
        )
