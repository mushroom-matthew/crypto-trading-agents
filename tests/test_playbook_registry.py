"""Tests for services/playbook_registry.py (Runbook 52).

Validates:
- load_playbook_from_md parses valid frontmatter into PlaybookDefinition
- load_playbook_from_md returns None when playbook_id is absent
- load_playbook_from_md handles malformed YAML gracefully
- PlaybookRegistry loads real playbooks from vector_store/playbooks/
- get / list_all / list_eligible / size API contracts
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from services.playbook_registry import PlaybookRegistry, load_playbook_from_md
from schemas.playbook_definition import PlaybookDefinition


# ---------------------------------------------------------------------------
# load_playbook_from_md — unit tests using tmp paths
# ---------------------------------------------------------------------------


class TestLoadPlaybookFromMd:
    def test_valid_frontmatter_returns_definition(self, tmp_path):
        """A .md file with playbook_id in frontmatter returns a PlaybookDefinition."""
        md = tmp_path / "test_pb.md"
        md.write_text(textwrap.dedent("""\
            ---
            title: Test Playbook
            playbook_id: test_pb
            regimes: [range, volatile]
            tags: [test, unit]
            identifiers: [rsi_14, close]
            ---
            # Test Playbook
            Some content here.
        """))
        pb = load_playbook_from_md(md)
        assert pb is not None
        assert isinstance(pb, PlaybookDefinition)
        assert pb.playbook_id == "test_pb"

    def test_regimes_loaded_into_eligible_regimes(self, tmp_path):
        md = tmp_path / "range_pb.md"
        md.write_text(textwrap.dedent("""\
            ---
            playbook_id: range_pb
            regimes: [range, bull]
            ---
            content
        """))
        pb = load_playbook_from_md(md)
        assert pb is not None
        assert "range" in pb.regime_eligibility.eligible_regimes
        assert "bull" in pb.regime_eligibility.eligible_regimes

    def test_returns_none_when_playbook_id_absent(self, tmp_path):
        """A .md file without playbook_id in frontmatter returns None."""
        md = tmp_path / "no_id.md"
        md.write_text(textwrap.dedent("""\
            ---
            title: No ID Playbook
            regimes: [range]
            ---
            # No ID
            This file has no playbook_id.
        """))
        result = load_playbook_from_md(md)
        assert result is None

    def test_returns_none_when_no_frontmatter(self, tmp_path):
        """A .md file with no frontmatter at all returns None."""
        md = tmp_path / "no_fm.md"
        md.write_text("# Just a heading\nNo frontmatter at all.\n")
        result = load_playbook_from_md(md)
        assert result is None

    def test_malformed_yaml_returns_none(self, tmp_path):
        """A .md file with invalid YAML frontmatter returns None without raising."""
        md = tmp_path / "bad_yaml.md"
        md.write_text(textwrap.dedent("""\
            ---
            title: Bad YAML
            broken: [unclosed bracket
            playbook_id: broken_pb
            ---
            # Bad YAML
        """))
        result = load_playbook_from_md(md)
        assert result is None

    def test_identifiers_loaded_as_list(self, tmp_path):
        """Identifiers from YAML list frontmatter are parsed as a Python list."""
        md = tmp_path / "ident_pb.md"
        md.write_text(textwrap.dedent("""\
            ---
            playbook_id: ident_pb
            identifiers: [rsi_14, macd_line, close]
            ---
            content
        """))
        pb = load_playbook_from_md(md)
        assert pb is not None
        assert isinstance(pb.identifiers, list)
        assert "rsi_14" in pb.identifiers
        assert "macd_line" in pb.identifiers

    def test_identifiers_string_parsed_as_list(self, tmp_path):
        """Identifiers given as a comma-separated string are split into a list."""
        md = tmp_path / "ident_str.md"
        md.write_text(textwrap.dedent("""\
            ---
            playbook_id: ident_str
            identifiers: "rsi_14, macd_line, close"
            ---
            content
        """))
        pb = load_playbook_from_md(md)
        assert pb is not None
        assert isinstance(pb.identifiers, list)
        assert "rsi_14" in pb.identifiers

    def test_tags_loaded_as_list(self, tmp_path):
        md = tmp_path / "tags_pb.md"
        md.write_text(textwrap.dedent("""\
            ---
            playbook_id: tags_pb
            tags: [momentum, breakout]
            ---
            content
        """))
        pb = load_playbook_from_md(md)
        assert pb is not None
        assert isinstance(pb.tags, list)
        assert "momentum" in pb.tags

    def test_description_set_from_title(self, tmp_path):
        """The `title` frontmatter key is mapped to PlaybookDefinition.description."""
        md = tmp_path / "title_pb.md"
        md.write_text(textwrap.dedent("""\
            ---
            playbook_id: title_pb
            title: My Playbook Title
            ---
            content
        """))
        pb = load_playbook_from_md(md)
        assert pb is not None
        assert pb.description == "My Playbook Title"

    def test_version_defaults_to_1_0_0(self, tmp_path):
        md = tmp_path / "v_pb.md"
        md.write_text(textwrap.dedent("""\
            ---
            playbook_id: v_pb
            ---
            content
        """))
        pb = load_playbook_from_md(md)
        assert pb is not None
        assert pb.version == "1.0.0"

    def test_custom_version_loaded(self, tmp_path):
        md = tmp_path / "cv_pb.md"
        md.write_text(textwrap.dedent("""\
            ---
            playbook_id: cv_pb
            version: "2.1.0"
            ---
            content
        """))
        pb = load_playbook_from_md(md)
        assert pb is not None
        assert pb.version == "2.1.0"


# ---------------------------------------------------------------------------
# PlaybookRegistry — using real vector_store/playbooks/ directory
# ---------------------------------------------------------------------------


# Locate the actual playbooks directory relative to this file's repo root
_REPO_ROOT = Path(__file__).parent.parent
_PLAYBOOK_DIR = _REPO_ROOT / "vector_store" / "playbooks"


@pytest.fixture(scope="module")
def registry() -> PlaybookRegistry:
    """Shared registry loaded from the real playbooks directory."""
    return PlaybookRegistry(playbook_dir=_PLAYBOOK_DIR)


class TestPlaybookRegistryReal:
    def test_loads_at_least_five_playbooks(self, registry):
        """Registry must load at least 5 valid playbooks from vector_store/playbooks/."""
        assert registry.size() >= 5

    def test_get_bollinger_squeeze(self, registry):
        """get() returns the right playbook by ID."""
        pb = registry.get("bollinger_squeeze")
        assert pb is not None
        assert pb.playbook_id == "bollinger_squeeze"

    def test_get_rsi_extremes(self, registry):
        pb = registry.get("rsi_extremes")
        assert pb is not None
        assert pb.playbook_id == "rsi_extremes"

    def test_get_unknown_returns_none(self, registry):
        """get() returns None for an unknown playbook ID."""
        result = registry.get("nonexistent_playbook_xyz")
        assert result is None

    def test_list_all_non_empty(self, registry):
        pbs = registry.list_all()
        assert isinstance(pbs, list)
        assert len(pbs) > 0

    def test_list_all_contains_playbook_definitions(self, registry):
        pbs = registry.list_all()
        for pb in pbs:
            assert isinstance(pb, PlaybookDefinition)

    def test_list_eligible_range_returns_results(self, registry):
        """Playbooks with 'range' in eligible_regimes should be returned."""
        results = registry.list_eligible("range")
        assert isinstance(results, list)
        assert len(results) > 0

    def test_list_eligible_all_have_range(self, registry):
        """All returned playbooks must actually include 'range' in eligible_regimes."""
        results = registry.list_eligible("range")
        for pb in results:
            assert "range" in pb.regime_eligibility.eligible_regimes

    def test_list_eligible_unknown_regime_returns_empty(self, registry):
        """list_eligible() returns empty list for a regime not in any playbook."""
        results = registry.list_eligible("unknown_regime_xyz_99")
        assert results == []

    def test_size_matches_playbooks_with_id(self, registry):
        """size() should equal the number of .md files that have a playbook_id."""
        count = 0
        for path in _PLAYBOOK_DIR.glob("*.md"):
            text = path.read_text(encoding="utf-8")
            import re
            import yaml
            m = re.match(r"^---\s*\n(.*?)\n---", text, re.DOTALL)
            if m:
                try:
                    fm = yaml.safe_load(m.group(1)) or {}
                    if fm.get("playbook_id"):
                        count += 1
                except Exception:
                    pass
        assert registry.size() == count

    def test_registry_skips_files_without_playbook_id(self, tmp_path):
        """Registry silently skips .md files without playbook_id."""
        # Create a directory with one valid and one invalid playbook
        (tmp_path / "valid.md").write_text(textwrap.dedent("""\
            ---
            playbook_id: valid_pb
            regimes: [range]
            ---
            content
        """))
        (tmp_path / "no_id.md").write_text(textwrap.dedent("""\
            ---
            title: No ID
            regimes: [range]
            ---
            content
        """))
        reg = PlaybookRegistry(playbook_dir=tmp_path)
        assert reg.size() == 1
        assert reg.get("valid_pb") is not None

    def test_registry_handles_malformed_yaml_gracefully(self, tmp_path):
        """Registry does not raise on malformed YAML; it skips the file."""
        (tmp_path / "bad.md").write_text(textwrap.dedent("""\
            ---
            broken: [unclosed
            playbook_id: bad_pb
            ---
            content
        """))
        (tmp_path / "good.md").write_text(textwrap.dedent("""\
            ---
            playbook_id: good_pb
            ---
            content
        """))
        reg = PlaybookRegistry(playbook_dir=tmp_path)
        assert reg.size() == 1
        assert reg.get("good_pb") is not None

    def test_loaded_playbook_identifiers_is_list(self, registry):
        """All loaded playbooks must have identifiers as a Python list."""
        for pb in registry.list_all():
            assert isinstance(pb.identifiers, list), (
                f"playbook {pb.playbook_id} has non-list identifiers: {type(pb.identifiers)}"
            )
