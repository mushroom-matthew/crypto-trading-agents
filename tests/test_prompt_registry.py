from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ops_api.routers.prompts import router
from ops_api.prompt_registry import current_strategist_prompt_path


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_current_strategist_prompt_path_defaults_to_simple(monkeypatch):
    monkeypatch.delenv("STRATEGIST_PROMPT", raising=False)
    assert current_strategist_prompt_path().name == "llm_strategist_simple.txt"


def test_current_strategist_prompt_path_can_switch_to_full(monkeypatch):
    monkeypatch.setenv("STRATEGIST_PROMPT", "full")
    assert current_strategist_prompt_path().name == "llm_strategist_prompt.txt"


def test_get_strategist_prompt_uses_runtime_default(monkeypatch):
    monkeypatch.delenv("STRATEGIST_PROMPT", raising=False)
    client = _client()

    response = client.get("/prompts/strategist")

    assert response.status_code == 200
    assert response.json()["file_path"].endswith("prompts/llm_strategist_simple.txt")


def test_get_strategist_prompt_uses_full_when_configured(monkeypatch):
    monkeypatch.setenv("STRATEGIST_PROMPT", "full")
    client = _client()

    response = client.get("/prompts/strategist")

    assert response.status_code == 200
    assert response.json()["file_path"].endswith("prompts/llm_strategist_prompt.txt")


def test_list_strategies_hides_legacy_templates_by_default(monkeypatch):
    monkeypatch.delenv("PROMPT_SHOW_HIDDEN_STRATEGIES", raising=False)
    client = _client()

    response = client.get("/prompts/strategies/")

    assert response.status_code == 200
    strategy_ids = {item["id"] for item in response.json()["strategies"]}
    assert "default" in strategy_ids
    assert "scalper_fast" in strategy_ids
    assert "range_long" in strategy_ids
    assert "range_short" in strategy_ids
    assert "aggressive_active" not in strategy_ids
    assert "balanced_hybrid" not in strategy_ids


def test_list_strategies_can_show_hidden_templates(monkeypatch):
    monkeypatch.setenv("PROMPT_SHOW_HIDDEN_STRATEGIES", "true")
    client = _client()

    response = client.get("/prompts/strategies/")

    assert response.status_code == 200
    strategy_ids = {item["id"] for item in response.json()["strategies"]}
    assert "aggressive_active" in strategy_ids
    assert "balanced_hybrid" in strategy_ids
