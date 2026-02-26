"""PlaybookRegistry — loads PlaybookDefinition objects from vector_store/playbooks/*.md frontmatter.

Each .md file in the playbooks directory may contain YAML frontmatter between '---' delimiters.
If the frontmatter includes a `playbook_id` key, the file is parsed into a PlaybookDefinition.
Files without `playbook_id` (or with malformed YAML) are silently skipped.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from schemas.playbook_definition import (
    EntryRuleSet,
    PlaybookDefinition,
    RegimeEligibility,
    RiskRuleSet,
)

_DEFAULT_PLAYBOOK_DIR = Path(__file__).parent.parent / "vector_store" / "playbooks"


def _parse_frontmatter(text: str) -> dict:
    """Extract YAML frontmatter between '---' delimiters.

    Returns an empty dict if no frontmatter is found or if YAML parsing fails.
    """
    match = re.match(r"^---\s*\n(.*?)\n---", text, re.DOTALL)
    if not match:
        return {}
    try:
        return yaml.safe_load(match.group(1)) or {}
    except Exception:
        return {}


def load_playbook_from_md(path: Path) -> Optional[PlaybookDefinition]:
    """Load a PlaybookDefinition from a single .md file.

    Returns None if:
    - The file has no frontmatter
    - The frontmatter is missing a `playbook_id` key
    - YAML parsing fails (malformed frontmatter)
    """
    text = path.read_text(encoding="utf-8")
    fm = _parse_frontmatter(text)

    if not fm.get("playbook_id"):
        return None

    # Build RegimeEligibility from frontmatter 'regimes' field
    regimes_raw = fm.get("regimes", [])
    if isinstance(regimes_raw, str):
        regimes_raw = [r.strip() for r in regimes_raw.split(",")]

    regime_eligibility = RegimeEligibility(
        eligible_regimes=regimes_raw,
    )

    # Build EntryRuleSet from frontmatter
    entry_rules = EntryRuleSet(
        thesis_conditions=fm.get("thesis_conditions", []),
        activation_triggers=fm.get("activation_triggers", []),
        activation_timeout_bars=fm.get("activation_timeout_bars"),
        activation_refinement_mode=fm.get("activation_refinement_mode", "price_touch"),
    )

    # Build RiskRuleSet
    risk_rules = RiskRuleSet(
        stop_methods=fm.get("stop_methods", []),
        target_methods=fm.get("target_methods", []),
        minimum_structural_r_multiple=fm.get("minimum_structural_r_multiple"),
        require_structural_target=fm.get("require_structural_target", False),
        structural_target_sources=fm.get("structural_target_sources", []),
    )

    # identifiers — from frontmatter or default to []
    identifiers_raw = fm.get("identifiers", [])
    if isinstance(identifiers_raw, str):
        identifiers_raw = [i.strip() for i in identifiers_raw.split(",")]

    tags_raw = fm.get("tags", [])
    if isinstance(tags_raw, str):
        tags_raw = [t.strip() for t in tags_raw.split(",")]

    return PlaybookDefinition(
        playbook_id=str(fm["playbook_id"]),
        version=str(fm.get("version", "1.0.0")),
        template_id=fm.get("template_id"),
        policy_class=fm.get("policy_class"),
        regime_eligibility=regime_eligibility,
        entry_rules=entry_rules,
        risk_rules=risk_rules,
        description=fm.get("title"),
        identifiers=identifiers_raw,
        tags=tags_raw,
    )


class PlaybookRegistry:
    """In-memory registry of PlaybookDefinition objects loaded from .md files.

    Automatically scans all .md files in the configured directory on construction.
    Files that lack a valid `playbook_id` in their frontmatter are silently skipped,
    so non-playbook markdown files in the same directory are safe to coexist.
    """

    def __init__(self, playbook_dir: Path = _DEFAULT_PLAYBOOK_DIR) -> None:
        self._dir = playbook_dir
        self._playbooks: Dict[str, PlaybookDefinition] = {}
        self._load_all()

    def _load_all(self) -> None:
        if not self._dir.exists():
            return
        for path in sorted(self._dir.glob("*.md")):
            pd = load_playbook_from_md(path)
            if pd is not None:
                self._playbooks[pd.playbook_id] = pd

    def get(self, playbook_id: str) -> Optional[PlaybookDefinition]:
        """Return the PlaybookDefinition for the given ID, or None if not found."""
        return self._playbooks.get(playbook_id)

    def list_all(self) -> List[PlaybookDefinition]:
        """Return all loaded playbooks as a list."""
        return list(self._playbooks.values())

    def list_eligible(self, regime: str) -> List[PlaybookDefinition]:
        """Return playbooks whose regime_eligibility includes the given regime.

        A playbook is eligible if:
        - `regime` is in its `eligible_regimes` list, AND
        - `regime` is NOT in its `disallowed_regimes` list.
        """
        return [
            pb
            for pb in self._playbooks.values()
            if regime in pb.regime_eligibility.eligible_regimes
            and regime not in pb.regime_eligibility.disallowed_regimes
        ]

    def size(self) -> int:
        """Return the number of successfully loaded playbooks."""
        return len(self._playbooks)
