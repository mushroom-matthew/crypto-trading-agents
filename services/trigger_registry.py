"""R96: Stateful trigger registry with incremental diff model.

Maintains a dict of active TriggerInstances across replans. The LLM outputs
a TriggerDiff (add/remove/modify) rather than a full replacement list, so
trigger identity is preserved across plan intervals.

POSITION_OPEN guard: exit triggers bound to an open position cannot be
removed or modified. Any diff that attempts this raises PositionProtectedTriggerMutation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from schemas.trigger_catalog import (
    TriggerDiff,
    TriggerInstance,
    instance_to_trigger_condition,
    ARCHETYPE_DIRECTIONS,
)


class PositionProtectedTriggerMutation(Exception):
    """Raised when a diff attempts to remove/modify an exit trigger bound to an open position."""

    def __init__(self, instance_id: str, reason: str) -> None:
        self.instance_id = instance_id
        super().__init__(f"Cannot mutate position-bound exit trigger '{instance_id}': {reason}")


class TriggerRegistry:
    """Mutable registry of TriggerInstances for a paper trading session.

    Serialisable for Temporal CaN via to_state() / from_state().
    """

    def __init__(self, session_id: str) -> None:
        self._session_id = session_id
        self._instances: Dict[str, TriggerInstance] = {}

    # ── Diff application ────────────────────────────────────────────────────

    def apply_diff(
        self,
        diff: TriggerDiff,
        policy_state: str = "IDLE",
        open_position_symbols: Optional[List[str]] = None,
    ) -> dict[str, Any]:
        """Apply a TriggerDiff to the registry.

        Args:
            diff: The LLM-produced diff.
            policy_state: Current PolicyStateMachine state string.
            open_position_symbols: Symbols with open positions. Exit triggers
                bound to these are protected from removal/modification.

        Returns:
            Summary dict with counts of adds/removes/modifications and any
            blocked mutations.
        """
        open_syms = set(open_position_symbols or [])
        in_position = policy_state == "POSITION_OPEN" or bool(open_syms)

        blocked: list[str] = []
        added = removed = modified = 0

        # Removals
        for iid in diff.to_remove:
            inst = self._instances.get(iid)
            if inst is None:
                continue
            if in_position and self._is_exit_trigger(inst) and inst.symbol in open_syms:
                blocked.append(iid)
                continue
            self._instances[iid] = inst.model_copy(update={"state": "removed"})
            removed += 1

        # Modifications
        for updated in diff.to_modify:
            existing = self._instances.get(updated.instance_id)
            if existing is None:
                continue
            if in_position and self._is_exit_trigger(existing) and existing.symbol in open_syms:
                blocked.append(updated.instance_id)
                continue
            self._instances[updated.instance_id] = updated
            modified += 1

        # Additions — deduplicate by instance_id
        for new_inst in diff.to_add:
            if new_inst.instance_id not in self._instances:
                self._instances[new_inst.instance_id] = new_inst
                added += 1
            else:
                # Already exists (same archetype + symbol) — treat as no-op
                pass

        return {
            "added": added,
            "removed": removed,
            "modified": modified,
            "blocked_mutations": blocked,
            "total_active": len(self.list_active()),
        }

    # ── Reads ───────────────────────────────────────────────────────────────

    def list_active(self) -> list[TriggerInstance]:
        return [i for i in self._instances.values() if i.state == "active"]

    def list_all(self) -> list[TriggerInstance]:
        return list(self._instances.values())

    def get(self, instance_id: str) -> Optional[TriggerInstance]:
        return self._instances.get(instance_id)

    def bind_position(self, instance_id: str, position_id: str) -> None:
        """Mark an entry trigger as bound to a fill (for exit-trigger protection)."""
        inst = self._instances.get(instance_id)
        if inst:
            self._instances[instance_id] = inst.model_copy(
                update={"bound_position_id": position_id}
            )

    # ── Conversion for trigger engine ───────────────────────────────────────

    def to_trigger_conditions(self) -> list[dict[str, Any]]:
        """Convert active instances to TriggerCondition-compatible dicts.

        The trigger engine accepts these dicts directly (same format as
        plan_dict["triggers"]).
        """
        result = []
        for inst in self.list_active():
            try:
                result.append(instance_to_trigger_condition(inst))
            except Exception:
                pass
        return result

    def to_context_block(self) -> str:
        """Format active triggers as an ACTIVE_TRIGGERS context block for the LLM prompt."""
        active = self.list_active()
        if not active:
            return "ACTIVE_TRIGGERS: (none — this is the first plan cycle)"
        lines = ["ACTIVE_TRIGGERS (current registry state — output a diff against this):"]
        for inst in active:
            param_str = ", ".join(f"{k}={v}" for k, v in inst.params.items())
            lines.append(
                f"  - {inst.instance_id}  archetype={inst.archetype_id}"
                f"  symbol={inst.symbol}  grade={inst.confidence_grade}"
                + (f"  params=[{param_str}]" if param_str else "")
            )
        return "\n".join(lines)

    # ── Serialisation for Temporal CaN ──────────────────────────────────────

    def to_state(self) -> dict[str, Any]:
        return {
            "session_id": self._session_id,
            "instances": {k: v.model_dump() for k, v in self._instances.items()},
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "TriggerRegistry":
        registry = cls(session_id=state.get("session_id", ""))
        for iid, inst_dict in state.get("instances", {}).items():
            try:
                registry._instances[iid] = TriggerInstance.model_validate(inst_dict)
            except Exception:
                pass
        return registry

    # ── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _is_exit_trigger(inst: TriggerInstance) -> bool:
        return ARCHETYPE_DIRECTIONS.get(inst.archetype_id) == "exit"
