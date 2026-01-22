"""LLM client facade used by the strategist backtester."""

from __future__ import annotations

import json
import logging
import os
import re
import time
import hashlib
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Protocol
from uuid import uuid4

from pydantic import ValidationError

from agents.langfuse_utils import langfuse_span
from agents.llm.client_factory import get_llm_client
from ops_api.event_store import EventStore
from ops_api.schemas import Event
from schemas.llm_strategist import LLMInput, PositionSizingRule, RiskConstraint, StrategyPlan


def _repair_json(raw: str) -> str:
    """Attempt to repair common JSON issues from LLM output.

    Handles:
    - Unterminated strings (truncates to last valid structure)
    - Trailing commas before ] or }
    - Unbalanced braces (adds missing closing braces)
    """
    if not raw:
        return raw

    # Remove trailing commas before ] or }
    raw = re.sub(r',\s*([\]\}])', r'\1', raw)

    # Try parsing as-is first
    try:
        json.loads(raw)
        return raw
    except json.JSONDecodeError:
        pass

    # Try to find a valid JSON substring by progressively truncating
    # Start from the end and work backwards looking for valid JSON
    for end_pos in range(len(raw), 0, -1):
        candidate = raw[:end_pos]

        # Count braces to check balance
        open_braces = candidate.count('{') - candidate.count('}')
        open_brackets = candidate.count('[') - candidate.count(']')

        # Add missing closing braces/brackets
        if open_braces > 0 or open_brackets > 0:
            # Check if we're in the middle of a string
            # Simple heuristic: if odd number of unescaped quotes, we're in a string
            quote_count = len(re.findall(r'(?<!\\)"', candidate))
            if quote_count % 2 != 0:
                # We're in a string - need to close it
                candidate = candidate.rstrip()
                if not candidate.endswith('"'):
                    candidate += '"'

            # Remove any trailing comma
            candidate = re.sub(r',\s*$', '', candidate)

            # Add closing brackets/braces
            candidate += ']' * open_brackets
            candidate += '}' * open_braces

        try:
            json.loads(candidate)
            logging.info("Repaired JSON by truncating to %d chars (was %d)", len(candidate), len(raw))
            return candidate
        except json.JSONDecodeError:
            continue

    # If nothing works, return original
    return raw


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
        self.last_generation_info: Dict[str, Any] = {}

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = get_llm_client()
        return self._client

    def generate_plan(
        self,
        llm_input: LLMInput,
        prompt_template: str | None = None,
        *,
        run_id: str | None = None,
        plan_id: str | None = None,
        prompt_hash: str | None = None,
        metadata: Dict[str, Any] | None = None,
        use_vector_store: bool = False,
        event_ts: datetime | None = None,
    ) -> StrategyPlan:
        def _emit_llm_event(payload: Dict[str, Any]) -> None:
            try:
                store = EventStore()
                ts = event_ts or datetime.now(timezone.utc)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                event = Event(
                    event_id=str(uuid4()),
                    ts=ts,
                    source="llm_client",
                    type="llm_call",  # type: ignore[arg-type]
                    payload=payload,
                    dedupe_key=None,
                    run_id=run_id,
                    correlation_id=plan_id or prompt_hash,
                )
                store.append(event)
            except Exception:
                pass

        input_hash = prompt_hash
        if not input_hash:
            payload_str = (prompt_template or "") + llm_input.to_json()
            input_hash = hashlib.sha256(payload_str.encode("utf-8")).hexdigest()

        if self.transport:
            raw = self.transport(llm_input.to_json())
            plan = StrategyPlan.from_json(raw)
            duration_ms = 0
            self.last_generation_info = {
                "source": "transport",
                "fallback_plan_used": False,
                "llm_failed_parse": False,
                "llm_failure_reason": None,
                "raw_output": raw,
                "prompt_hash": input_hash,
            }
            _emit_llm_event(
                {
                    "model": self.model,
                    "source": "transport",
                    "raw_len": len(raw or ""),
                    "prompt_hash": input_hash,
                    "plan_id": plan_id,
                    "duration_ms": duration_ms,
                    **(metadata or {}),
                }
            )
            return plan
        last_error: Exception | None = None
        raw_output: str | None = None
        attempts = max(1, self.max_retries)
        start_time = time.monotonic()
        for attempt in range(1, attempts + 1):
            try:
                vector_context = self._get_vector_context(llm_input) if use_vector_store else None
                system_prompt = self._build_system_prompt(prompt_template, vector_context)
                with langfuse_span("llm_strategist.backtest", metadata={"model": self.model}) as span:
                    completion = self.client.responses.create(
                        model=self.model,
                        input=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": llm_input.to_json()},
                        ],
                        temperature=0.1,
                        max_output_tokens=2500,
                    )
                    content = completion.output[0].content[0].text
                    raw_output = content
                    if span:
                        span.end(output=content)
                    plan_dict = self._extract_plan_json(content)
                    plan_dict = self._sanitize_plan_dict(plan_dict)
                    plan = StrategyPlan.model_validate(plan_dict)
                    duration_ms = int((time.monotonic() - start_time) * 1000)
                    self.last_generation_info = {
                        "source": "llm",
                        "fallback_plan_used": False,
                        "llm_failed_parse": False,
                        "llm_failure_reason": None,
                        "raw_output": raw_output,
                        "prompt_hash": input_hash,
                    }
                    _emit_llm_event(
                        {
                            "model": self.model,
                            "source": "llm",
                            "raw_len": len(raw_output or ""),
                            "tokens_in": completion.usage.input_tokens if hasattr(completion, "usage") else 0,
                            "tokens_out": completion.usage.output_tokens if hasattr(completion, "usage") else 0,
                            "prompt_hash": input_hash,
                            "plan_id": plan_id,
                            "duration_ms": duration_ms,
                            **(metadata or {}),
                        }
                    )
                    return plan
            except ValidationError as exc:
                last_error = exc
                logging.warning("Strategy plan validation failed (attempt %s/%s): %s", attempt, attempts, exc)
                continue
            except Exception as exc:
                last_error = exc
                logging.warning("LLM strategist call failed (attempt %s/%s): %s; raw_output=%s", attempt, attempts, exc, raw_output)
                continue
        if not self.allow_fallback and last_error:
            raise last_error
        logging.warning("LLM strategist fallback triggered after retries: %s; raw_output=%s", last_error, raw_output)
        plan = self._fallback_plan(llm_input)
        duration_ms = int((time.monotonic() - start_time) * 1000)
        self.last_generation_info = {
            "source": "fallback",
            "fallback_plan_used": True,
            "llm_failed_parse": True,
            "llm_failure_reason": str(last_error) if last_error else "unknown_failure",
            "raw_output": raw_output,
            "prompt_hash": input_hash,
        }
        _emit_llm_event(
            {
                "model": self.model,
                "source": "fallback",
                "raw_len": len(raw_output or ""),
                "llm_failure_reason": str(last_error) if last_error else None,
                "prompt_hash": input_hash,
                "plan_id": plan_id,
                "duration_ms": duration_ms,
                **(metadata or {}),
            }
        )
        return plan

    def _extract_plan_json(self, content: str) -> Dict[str, Any]:
        """Strip fences/prose and return parsed JSON with light validation."""

        raw = (content or "").strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            if len(parts) >= 3:
                raw = parts[1] if parts[1].strip() else parts[2]
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
        if not raw.startswith("{"):
            first = raw.find("{")
            last = raw.rfind("}")
            if first != -1 and last != -1 and last > first:
                raw = raw[first : last + 1]

        # Attempt to repair truncated/malformed JSON
        raw = _repair_json(raw)

        try:
            parsed = json.loads(raw)
        except Exception as exc:
            logging.warning("Failed to parse strategist output as JSON: %s; raw_output=%s", exc, raw[:500])
            raise
        if not isinstance(parsed, dict):
            raise ValueError("Strategist output is not a JSON object")
        triggers = parsed.get("triggers")
        if triggers is not None and not isinstance(triggers, list):
            raise ValueError("Strategist output 'triggers' must be a list")
        required_keys = {"risk_constraints", "sizing_rules"}
        missing = sorted(key for key in required_keys if key not in parsed)
        if missing:
            raise ValueError(f"Strategist output missing keys: {', '.join(missing)}")
        return parsed

    def _get_vector_context(self, llm_input: LLMInput) -> str | None:
        """Get relevant trigger examples from vector store."""
        try:
            from .trigger_vector_store import get_trigger_vector_store

            store = get_trigger_vector_store()
            if not store.examples:
                return None

            # Build context from input
            context = {
                "regime": llm_input.global_context.get("regime", "unknown") if llm_input.global_context else "unknown",
            }

            # Get trend and vol state from first asset
            if llm_input.assets:
                asset = llm_input.assets[0]
                context["trend_state"] = asset.trend_state or "unknown"
                context["vol_state"] = asset.vol_state or "normal"
                context["symbol"] = asset.symbol

                # Get RSI range if available
                if asset.indicators:
                    rsi = asset.indicators[0].rsi_14
                    if rsi is not None:
                        if rsi < 30:
                            context["rsi_range"] = "oversold"
                        elif rsi > 70:
                            context["rsi_range"] = "overbought"
                        else:
                            context["rsi_range"] = "neutral"

            return store.get_context_injection(context, top_k=3)
        except Exception as e:
            logging.warning("Failed to get vector context: %s", e)
            return None

    def _build_system_prompt(self, prompt_template: str | None, vector_context: str | None = None) -> str:
        base = prompt_template or os.environ.get("LLM_STRATEGIST_PROMPT", "") or self._default_prompt()
        schema = self._schema_prompt()
        if schema and schema not in base:
            base = f"{base}\n\n{schema}"
        if vector_context:
            base = f"{base}\n\n{vector_context}"
        return base

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
        allowed = {"long", "short", "flat", "exit", "flat_exit"}
        for trig in triggers:
            direction = trig.get("direction")
            if direction not in allowed:
                continue
            if isinstance(direction, str) and direction.lower() == "flat_exit":
                trig["direction"] = "exit"
            if trig.get("category") == "emergency_exit":
                exit_rule = (trig.get("exit_rule") or "").strip()
                entry_rule = (trig.get("entry_rule") or "").strip()
                if not exit_rule and entry_rule:
                    trig["exit_rule"] = entry_rule
                    trig["entry_rule"] = "false"
                elif not exit_rule:
                    trig["exit_rule"] = "false"
                    trig["entry_rule"] = "false"
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

    @staticmethod
    @lru_cache(1)
    def _schema_prompt() -> str:
        """Load shared StrategyPlan schema block for all strategist prompts."""

        schema_path = Path(__file__).resolve().parents[2] / "prompts" / "strategy_plan_schema.txt"
        if schema_path.exists():
            return schema_path.read_text(encoding="utf-8").strip()
        return ""
