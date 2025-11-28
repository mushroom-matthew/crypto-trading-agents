"""Prompt management utilities for execution and judge agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional


@dataclass
class PromptComponent:
    """Single prompt segment with optional conditional metadata."""

    name: str
    content: str
    priority: int = 0
    conditions: Dict[str, str] = field(default_factory=dict)

    def matches(self, context: Optional[Dict[str, str]] = None) -> bool:
        if not self.conditions:
            return True
        if not context:
            return False
        return all(context.get(key) == value for key, value in self.conditions.items())


@dataclass
class PromptTemplate:
    """Renderable collection of components."""

    name: str
    description: str
    components: List[PromptComponent]

    def render(self, context: Optional[Dict[str, str]] = None) -> str:
        context = context or {}
        active = [comp for comp in self.components if comp.matches(context)]
        active.sort(key=lambda comp: comp.priority, reverse=True)
        return "\n\n".join(comp.content.strip() for comp in active)


class PromptManager:
    """Simple prompt manager with default templates and variant generation."""

    def __init__(self, temporal_client: object | None = None) -> None:
        self.temporal_client = temporal_client
        self.default_components = self._build_default_components()
        self.templates = self._build_templates()

    def _build_default_components(self) -> Dict[str, PromptComponent]:
        return {
            "role_definition": PromptComponent(
                name="role_definition",
                priority=1000,
                content="You are an autonomous portfolio management agent tasked with balancing growth and risk.",
            ),
            "operational_workflow": PromptComponent(
                name="operational_workflow",
                priority=900,
                content="Follow the observe -> plan -> execute -> review loop on every decision cycle.",
            ),
            "decision_framework": PromptComponent(
                name="decision_framework",
                priority=800,
                content="Base decisions on validated signals, quantitative risk limits, and portfolio level context.",
            ),
            "risk_management": PromptComponent(
                name="risk_management",
                priority=700,
                content="Never violate provided risk caps. Prioritize capital preservation when volatility is elevated.",
            ),
            "risk_conservative": PromptComponent(
                name="risk_conservative",
                priority=650,
                content="Adopt a conservative stance: favor capital preservation and reduce leverage.",
                conditions={"risk_mode": "conservative"},
            ),
            "risk_aggressive": PromptComponent(
                name="risk_aggressive",
                priority=650,
                content="Aggressive mode enabled: pursue high-conviction setups while respecting hard stops.",
                conditions={"risk_mode": "aggressive"},
            ),
        }

    def _compose(self, *names: str) -> List[PromptComponent]:
        return [self.default_components[name] for name in names if name in self.default_components]

    def _build_templates(self) -> Dict[str, PromptTemplate]:
        return {
            "execution_agent_standard": PromptTemplate(
                name="execution_agent_standard",
                description="Balanced execution agent prompt",
                components=self._compose(
                    "role_definition",
                    "operational_workflow",
                    "decision_framework",
                    "risk_management",
                ),
            ),
            "execution_agent_conservative": PromptTemplate(
                name="execution_agent_conservative",
                description="Drawdown-aware conservative variant",
                components=self._compose(
                    "role_definition",
                    "operational_workflow",
                    "decision_framework",
                    "risk_management",
                    "risk_conservative",
                ),
            ),
            "execution_agent_performance": PromptTemplate(
                name="execution_agent_performance",
                description="Aggressive recovery template",
                components=self._compose(
                    "role_definition",
                    "operational_workflow",
                    "decision_framework",
                    "risk_management",
                    "risk_aggressive",
                ),
            ),
        }

    async def get_current_prompt(self, template_key: str, context: Optional[Dict[str, str]] = None) -> str:
        key = template_key
        if key not in self.templates:
            if key.startswith("execution_agent"):
                key = "execution_agent_standard"
            else:
                key = "execution_agent_standard"
        template = self.templates[key]
        return template.render(context)

    def generate_prompt_variants(self, template_key: str, performance_metrics: Dict[str, float]) -> List[tuple[str, str, str]]:
        """Return candidate prompt variants based on performance metrics."""

        variants: List[tuple[str, str, str]] = []
        template = self.templates.get(template_key)
        if template:
            variants.append(("baseline", template.description, template.render()))

        max_drawdown = performance_metrics.get("max_drawdown", 0.0)
        win_rate = performance_metrics.get("win_rate", 0.0)
        risk_score = performance_metrics.get("risk_management_score", 50.0)

        if max_drawdown >= 0.15 or risk_score < 70.0:
            conservative = self.templates["execution_agent_conservative"]
            variants.append(("conservative_risk", "High drawdown conservative variant", conservative.render({"risk_mode": "conservative"})))

        if win_rate >= 0.6 and max_drawdown < 0.1:
            aggressive = self.templates["execution_agent_performance"]
            variants.append(("confidence_boost", "Higher conviction plan", aggressive.render({"risk_mode": "aggressive"})))

        return variants
