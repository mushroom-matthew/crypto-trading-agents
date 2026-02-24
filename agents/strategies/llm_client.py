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
from agents.llm.model_utils import output_token_args, reasoning_args, temperature_args
from ops_api.event_store import EventStore
from ops_api.schemas import Event
from schemas.llm_strategist import LLMInput, PositionSizingRule, RiskConstraint, StrategyPlan

from .prompt_builder import build_prompt_context
from .trigger_vector_store import get_trigger_vector_store
from vector_store.retriever import get_strategy_vector_store, vector_store_enabled


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
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-5-mini")
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
                "retrieved_template_id": None,
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

        # Resolve strategy context and template once before the retry loop.
        strategy_context, retrieved_template_id = self._get_strategy_context(llm_input)
        # Explicit prompt_template overrides any retrieved template.
        effective_template = prompt_template
        if not effective_template and retrieved_template_id:
            template_path = (
                Path(__file__).resolve().parents[2]
                / "prompts" / "strategies"
                / f"{retrieved_template_id}.txt"
            )
            if template_path.exists():
                effective_template = template_path.read_text(encoding="utf-8").strip()
                logging.info("Using retrieved template '%s' for plan generation", retrieved_template_id)

        for attempt in range(1, attempts + 1):
            try:
                vector_context = self._get_vector_context(llm_input) if use_vector_store else None
                prompt_context = build_prompt_context(llm_input)
                system_prompt = self._build_system_prompt(
                    effective_template,
                    vector_context,
                    prompt_context,
                    strategy_context,
                )
                with langfuse_span("llm_strategist.backtest", metadata={"model": self.model}) as span:
                    completion = self.client.responses.create(
                        model=self.model,
                        input=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": llm_input.to_json()},
                        ],
                        **output_token_args(self.model, 2500),
                        **temperature_args(self.model, 0.1),
                        **reasoning_args(self.model, effort="low"),
                    )
                    content = completion.output_text
                    if not content:
                        raise ValueError(
                            f"LLM returned empty content; output={completion.output!r}"
                        )
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
                        "retrieved_template_id": retrieved_template_id,
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
            "retrieved_template_id": retrieved_template_id,
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
        # Only sizing_rules is strictly required - risk_constraints now optional (user config is authoritative)
        required_keys = {"sizing_rules"}
        missing = sorted(key for key in required_keys if key not in parsed)
        if missing:
            raise ValueError(f"Strategist output missing keys: {', '.join(missing)}")
        return parsed

    def _get_vector_context(self, llm_input: LLMInput) -> str | None:
        """Get relevant trigger examples from vector store."""
        try:
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

    def _get_strategy_context(self, llm_input: LLMInput) -> tuple[str | None, str | None]:
        """Get retrieved strategy/rule docs from the strategy vector store.

        Returns (context_block, template_id).
        """
        if not vector_store_enabled():
            return None, None
        try:
            store = get_strategy_vector_store()
            result = store.retrieve_context(llm_input)
            return result.context, result.template_id
        except Exception as exc:
            logging.warning("Failed to get strategy context: %s", exc)
            return None, None

    def _build_system_prompt(
        self,
        prompt_template: str | None,
        vector_context: str | None = None,
        prompt_context: str | None = None,
        strategy_context: str | None = None,
    ) -> str:
        base = self._default_prompt()
        schema = self._schema_prompt()
        if schema and schema not in base:
            base = f"{base}\n\n{schema}"
        if prompt_template:
            base = f"{base}\n\nSTRATEGY_GUIDANCE:\n{prompt_template.strip()}"
        if prompt_context:
            base = f"{base}\n\n{prompt_context}"
        if strategy_context:
            base = f"{base}\n\n{strategy_context}"
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
        data["regime_alerts"] = self._normalize_regime_alerts(data.get("regime_alerts"))
        data["sizing_hints"] = self._normalize_sizing_hints(data.get("sizing_hints"))
        data["trigger_budgets"] = self._normalize_trigger_budgets(data.get("trigger_budgets"))
        data["max_triggers_per_symbol_per_day"] = self._normalize_max_triggers_per_symbol_per_day(
            data.get("max_triggers_per_symbol_per_day"),
            data,
        )
        return data

    @staticmethod
    def _normalize_regime_alerts(value: Any) -> List[Dict[str, Any]]:
        if value is None:
            return []
        if isinstance(value, dict):
            value = [value]
        if not isinstance(value, list):
            logging.warning("Dropping invalid regime_alerts (expected list, got %s)", type(value).__name__)
            return []
        normalized: List[Dict[str, Any]] = []
        dropped_strings = False
        allowed_direction = {"above", "below", "crosses"}
        allowed_priority = {"high", "medium", "low"}
        for item in value:
            if isinstance(item, dict):
                indicator = item.get("indicator")
                threshold = item.get("threshold")
                direction = item.get("direction")
                symbol = item.get("symbol")
                interpretation = item.get("interpretation")
                if not all([indicator, threshold, direction, symbol, interpretation]):
                    continue
                try:
                    threshold_val = float(threshold)
                except (TypeError, ValueError):
                    continue
                direction_val = str(direction).lower()
                if direction_val not in allowed_direction:
                    continue
                priority_val = str(item.get("priority", "medium")).lower()
                if priority_val not in allowed_priority:
                    priority_val = "medium"
                normalized.append(
                    {
                        "indicator": str(indicator),
                        "threshold": threshold_val,
                        "direction": direction_val,
                        "symbol": str(symbol),
                        "interpretation": str(interpretation),
                        "priority": priority_val,
                    }
                )
            elif isinstance(item, str):
                dropped_strings = True
        if dropped_strings:
            logging.warning("Dropping string regime_alerts entries; expected structured objects.")
        return normalized

    @staticmethod
    def _normalize_sizing_hints(value: Any) -> List[Dict[str, Any]]:
        if value is None:
            return []
        if isinstance(value, dict):
            value = [value]
        if not isinstance(value, list):
            logging.warning("Dropping invalid sizing_hints (expected list, got %s)", type(value).__name__)
            return []
        normalized: List[Dict[str, Any]] = []
        dropped_strings = False
        for item in value:
            if isinstance(item, dict):
                symbol = item.get("symbol")
                suggested = item.get("suggested_risk_pct")
                rationale = item.get("rationale")
                if not all([symbol, suggested, rationale]):
                    continue
                try:
                    suggested_val = float(suggested)
                except (TypeError, ValueError):
                    continue
                normalized.append(
                    {
                        "symbol": str(symbol),
                        "suggested_risk_pct": suggested_val,
                        "rationale": str(rationale),
                    }
                )
            elif isinstance(item, str):
                dropped_strings = True
        if dropped_strings:
            logging.warning("Dropping string sizing_hints entries; expected structured objects.")
        return normalized

    @staticmethod
    def _normalize_trigger_budgets(value: Any) -> Dict[str, int]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            logging.warning("Dropping invalid trigger_budgets (expected dict, got %s)", type(value).__name__)
            return {}
        normalized: Dict[str, int] = {}
        for key, raw_val in value.items():
            if key is None:
                continue
            try:
                int_val = int(raw_val)
            except (TypeError, ValueError):
                continue
            if int_val < 0:
                continue
            normalized[str(key)] = int_val
        return normalized

    def _normalize_max_triggers_per_symbol_per_day(self, value: Any, data: Dict[str, Any]) -> int | None:
        if value is None:
            return None
        if isinstance(value, dict):
            # LLM sometimes emits per-symbol dict here; treat as trigger_budgets instead.
            migrated = self._normalize_trigger_budgets(value)
            if migrated:
                existing = data.get("trigger_budgets")
                if isinstance(existing, dict):
                    merged = {**existing, **migrated}
                    data["trigger_budgets"] = merged
                else:
                    data["trigger_budgets"] = migrated
                max_val = max(migrated.values()) if migrated else None
                logging.warning(
                    "Coerced max_triggers_per_symbol_per_day dict into trigger_budgets; using max=%s",
                    max_val,
                )
                return max_val
            return None
        try:
            int_val = int(value)
        except (TypeError, ValueError):
            logging.warning("Dropping invalid max_triggers_per_symbol_per_day (expected int, got %s)", type(value).__name__)
            return None
        if int_val < 0:
            return None
        return int_val

    @staticmethod
    @lru_cache(1)
    def _default_prompt() -> str:
        """Load bundled strategist prompt when no template/env override is provided.

        Prompt selection via STRATEGIST_PROMPT env var:
        - "simple" or unset -> llm_strategist_simple.txt (simplified prompt)
        - "full" -> llm_strategist_prompt.txt (legacy full prompt)
        """
        prompt_name = os.environ.get("STRATEGIST_PROMPT", "simple").lower()
        prompts_dir = Path(__file__).resolve().parents[2] / "prompts"

        if prompt_name == "simple":
            prompt_path = prompts_dir / "llm_strategist_simple.txt"
        else:
            prompt_path = prompts_dir / "llm_strategist_prompt.txt"

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
