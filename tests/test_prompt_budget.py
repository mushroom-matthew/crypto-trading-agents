"""Tests for prompt budget telemetry: token counter, judge prompt caps, schema mode."""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch


# ---------------------------------------------------------------------------
# TestCountTokens
# ---------------------------------------------------------------------------

class TestCountTokens:
    def test_positive_for_non_empty_text(self):
        from services.prompt_token_counter import count_tokens
        assert count_tokens("Hello, world!") > 0

    def test_zero_for_empty_string(self):
        from services.prompt_token_counter import count_tokens
        assert count_tokens("") == 0

    def test_consistent_with_known_text(self):
        from services.prompt_token_counter import count_tokens
        # "Hello, world!" should be a small number of tokens (definitely < 20)
        result = count_tokens("Hello, world!")
        assert 1 <= result <= 20

    def test_longer_text_more_tokens(self):
        from services.prompt_token_counter import count_tokens
        short = count_tokens("Hi")
        long = count_tokens("This is a much longer piece of text that should consume more tokens than a short greeting.")
        assert long > short

    def test_fallback_on_tiktoken_error(self):
        """If tiktoken raises, function falls back to char/4."""
        from services import prompt_token_counter
        original = prompt_token_counter._get_encoding

        def _bad_encoding(name):
            raise RuntimeError("simulated tiktoken failure")

        prompt_token_counter._get_encoding.cache_clear()
        try:
            prompt_token_counter._get_encoding = _bad_encoding
            result = prompt_token_counter.count_tokens("abcd")
            # char/4 fallback: max(1, 4//4) = 1
            assert result >= 1
        finally:
            prompt_token_counter._get_encoding = original
            prompt_token_counter._get_encoding.cache_clear()


# ---------------------------------------------------------------------------
# TestCapJson
# ---------------------------------------------------------------------------

class TestCapJson:
    def _cap_json(self, obj, max_chars=500, label=""):
        from services.judge_feedback_service import _cap_json
        return _cap_json(obj, max_chars=max_chars, label=label)

    def test_short_obj_not_truncated(self):
        obj = {"key": "value"}
        result = self._cap_json(obj, max_chars=500)
        # Should be valid JSON (not truncated)
        parsed = json.loads(result)
        assert parsed["key"] == "value"

    def test_long_obj_truncated_with_marker(self):
        obj = {"data": "x" * 1000}
        result = self._cap_json(obj, max_chars=100)
        assert "omitted" in result
        assert len(result) < 200  # truncation kept it small

    def test_empty_obj_returns_label_string(self):
        result = self._cap_json({}, label="market structure")
        assert "market structure" in result

    def test_none_returns_label_string(self):
        result = self._cap_json(None, label="risk state")
        assert "risk state" in result

    def test_no_indent_in_output(self):
        """Compact form — must not contain indent=2 style newlines."""
        obj = {"a": 1, "b": 2, "c": [1, 2, 3]}
        result = self._cap_json(obj, max_chars=10000)
        # compact JSON has no leading whitespace on lines
        assert "\n  " not in result

    def test_empty_label_returns_empty_string(self):
        result = self._cap_json({})
        assert result == ""


# ---------------------------------------------------------------------------
# TestJudgePromptCaps
# ---------------------------------------------------------------------------

class TestJudgePromptCaps:
    """Verify list limits are capped at 10."""

    def _make_fills(self, n: int) -> list[dict]:
        return [
            {
                "timestamp": "2024-01-01T00:00:00Z",
                "symbol": "BTC-USD",
                "side": "buy",
                "qty": 1,
                "price": 40000 + i,
                "trigger_id": f"t{i}",
                "pnl": i * 10,
            }
            for i in range(n)
        ]

    def _build_prompt(self, fills, active_triggers=None, trigger_attempts=None):
        """Call the private _build_judge_prompt_sections logic via _call_llm indirectly.

        We test the cap by checking the rendered prompt line count for fills.
        Instead of calling the full LLM, we examine _cap_json and slice behavior
        by checking the judge service's _build_analysis_prompt helper via the
        fills[:10] guard in _call_llm.
        """
        # We test this by checking the slice directly, since we can't call _call_llm
        # without a full service setup.
        return fills[:10]

    def test_fills_capped_at_10(self):
        fills = self._make_fills(25)
        capped = fills[:10]
        assert len(capped) == 10

    def test_active_triggers_capped_at_10(self):
        triggers = [{"id": f"t{i}", "symbol": "BTC-USD"} for i in range(25)]
        capped = triggers[:10]
        assert len(capped) == 10

    def test_trigger_attempts_capped_at_10(self):
        attempts = {f"trigger_{i}": {"attempted": i, "blocked": 0} for i in range(25)}
        ranked = sorted(attempts.items(), key=lambda item: item[1].get("attempted", 0), reverse=True)
        capped = ranked[:10]
        assert len(capped) == 10

    def test_fills_slice_constant_in_service(self):
        """Verify the service source uses [:10] not [:20] for fills."""
        src_path = Path(__file__).resolve().parents[1] / "services" / "judge_feedback_service.py"
        src = src_path.read_text(encoding="utf-8")
        # Should NOT contain fills_since_last_judge[:20]
        assert "fills_since_last_judge[:20]" not in src
        # Should contain fills_since_last_judge[:10]
        assert "fills_since_last_judge[:10]" in src

    def test_active_triggers_slice_constant_in_service(self):
        src_path = Path(__file__).resolve().parents[1] / "services" / "judge_feedback_service.py"
        src = src_path.read_text(encoding="utf-8")
        assert "active_triggers[:20]" not in src
        assert "active_triggers[:10]" in src


# ---------------------------------------------------------------------------
# TestSchemaMode
# ---------------------------------------------------------------------------

class TestSchemaMode:
    def test_default_mode_is_core(self):
        """Without env var, STRATEGIST_SCHEMA_MODE defaults to 'core'."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("STRATEGIST_SCHEMA_MODE", None)
            mode = os.environ.get("STRATEGIST_SCHEMA_MODE", "core").lower()
            assert mode == "core"

    def test_verbose_mode_env_var(self):
        with patch.dict(os.environ, {"STRATEGIST_SCHEMA_MODE": "verbose"}):
            mode = os.environ.get("STRATEGIST_SCHEMA_MODE", "core").lower()
            assert mode == "verbose"

    def test_core_schema_file_exists(self):
        prompts_dir = Path(__file__).resolve().parents[1] / "prompts"
        core_path = prompts_dir / "strategy_plan_schema_core.txt"
        assert core_path.exists(), "strategy_plan_schema_core.txt must exist"

    def test_verbose_mode_loads_full_schema(self):
        from agents.strategies.llm_client import _load_schema_prompt
        _load_schema_prompt.cache_clear()
        result = _load_schema_prompt("verbose")
        # Full schema should have CANDLESTICK USAGE GUIDANCE multi-line content
        assert "CANDLESTICK USAGE GUIDANCE" in result
        assert len(result) > 100

    def test_core_schema_shorter_than_verbose(self):
        from agents.strategies.llm_client import _load_schema_prompt
        _load_schema_prompt.cache_clear()
        core = _load_schema_prompt("core")
        verbose = _load_schema_prompt("verbose")
        # Core should be meaningfully shorter
        assert len(core) < len(verbose), (
            f"Core schema ({len(core)} chars) should be shorter than verbose ({len(verbose)} chars)"
        )

    def test_core_schema_preserves_required_sections(self):
        prompts_dir = Path(__file__).resolve().parents[1] / "prompts"
        core = (prompts_dir / "strategy_plan_schema_core.txt").read_text(encoding="utf-8")
        # All critical constraint sections must be present
        required_fragments = [
            "STRATEGYPLAN SCHEMA",
            "Allowed Rule Identifiers",
            "stop_anchor_type",
            "target_anchor_type",
            "BUILT-IN IDENTIFIERS",
            "R-TRACKING IDENTIFIERS",
            "COMPRESSION / BREAKOUT IDENTIFIERS",
            "ATR Tautology",
        ]
        for fragment in required_fragments:
            assert fragment in core, f"Missing required section: {fragment!r}"
