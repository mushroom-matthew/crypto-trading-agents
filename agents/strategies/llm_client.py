"""LLM client facade used by the strategist backtester."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Protocol

from openai import OpenAI
from pydantic import ValidationError

from agents.langfuse_utils import langfuse_span
from schemas.llm_strategist import LLMInput, PositionSizingRule, RiskConstraint, StrategyPlan


class CompletionTransport(Protocol):
    def __call__(self, payload: str) -> str: ...


class LLMClient:
    """Thin wrapper over OpenAI's Responses API with a deterministic fallback."""

    def __init__(
        self,
        transport: CompletionTransport | None = None,
        model: str | None = None,
        allow_fallback: bool | None = None,
    ) -> None:
        self.transport = transport
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        env_flag = os.environ.get("LLM_CLIENT_ALLOW_FALLBACK")
        if allow_fallback is None and env_flag is not None:
            allow_fallback = env_flag.lower() not in {"0", "false", "no"}
        self.allow_fallback = True if allow_fallback is None else allow_fallback
        self.max_retries = int(os.environ.get("LLM_CLIENT_MAX_RETRIES", "2"))
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
        last_error: Exception | None = None
        attempts = max(1, self.max_retries)
        for attempt in range(1, attempts + 1):
            try:
                system_prompt = prompt_template or os.environ.get("LLM_STRATEGIST_PROMPT", "") or self._default_prompt()
                with langfuse_span("llm_strategist.backtest", metadata={"model": self.model}) as span:
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
                    if span:
                        span.end(output=content)
                    plan_dict = json.loads(content)
                    plan_dict = self._sanitize_plan_dict(plan_dict)
                    return StrategyPlan.model_validate(plan_dict)
            except ValidationError as exc:
                last_error = exc
                logging.warning("Strategy plan validation failed (attempt %s/%s): %s", attempt, attempts, exc)
                continue
            except Exception as exc:
                last_error = exc
                logging.warning("LLM strategist call failed (attempt %s/%s): %s", attempt, attempts, exc)
                continue
        if not self.allow_fallback and last_error:
            raise last_error
        logging.warning("LLM strategist fallback triggered after retries: %s", last_error)
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

    def _sanitize_plan_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        triggers = data.get("triggers", [])
        cleaned: List[Dict[str, Any]] = []
        allowed = {"long", "short", "flat"}
        for trig in triggers:
            direction = trig.get("direction")
            if direction not in allowed:
                if isinstance(direction, str) and direction.lower() == "exit":
                    trig["direction"] = "flat"
                else:
                    continue
            cleaned.append(trig)
        data["triggers"] = cleaned
        return data

    @staticmethod
    @lru_cache(1)
    def _default_prompt() -> str:
        """Load bundled strategist prompt when no template/env override is provided."""

        prompt_path = Path(__file__).resolve().parents[2] / "prompts" / "llm_strategist_prompt.txt"
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8").strip()
        return ""
