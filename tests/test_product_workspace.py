"""Product-workspace API tests: connections, uploads, approval and evidence."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import dft_forge.web as web
from dft_forge.llm import OpenAICompatProvider


client = TestClient(web.app)


@pytest.fixture
def product_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    monkeypatch.setattr(web, "WEB_WORKDIR", sessions)
    monkeypatch.setattr(web, "ENV_FILE", tmp_path / ".env")
    web.WEB_SESSION_CTX.clear()
    web.WEB_ACTIVE_RUNS.clear()
    monkeypatch.setenv("DFT_FORGE_LLM_PROVIDER", "dummy")
    monkeypatch.setenv("DFT_FORGE_LLM_VENDOR", "rules")
    monkeypatch.setenv("DFT_FORGE_EXECUTOR", "fake")
    monkeypatch.delenv("DFT_FORGE_LLM_API_KEY", raising=False)
    yield sessions
    web.WEB_SESSION_CTX.clear()
    web.WEB_ACTIVE_RUNS.clear()


def test_connection_catalog_and_compute_state(product_env):
    response = client.get("/api/config")
    assert response.status_code == 200
    data = response.json()
    assert {item["id"] for item in data["providers"]} >= {"openai", "deepseek", "qwen", "ollama", "custom"}
    assert data["compute"]["executor"] == "fake"
    assert data["provider"] == "dummy"


def test_save_model_config_persists_without_exposing_key(product_env, monkeypatch):
    monkeypatch.setattr(web.agent_loop, "reset_llm", lambda: None)
    response = client.post("/api/config", json={
        "provider": "openai", "vendor": "deepseek",
        "base_url": "https://api.deepseek.com", "model": "deepseek-chat", "api_key": "secret-key-value",
    })
    assert response.status_code == 200
    assert response.json()["api_key_hint"] != "secret-key-value"
    env_text = web.ENV_FILE.read_text()
    assert "DFT_FORGE_LLM_VENDOR=deepseek" in env_text
    assert "DFT_FORGE_LLM_MODEL=deepseek-chat" in env_text


def test_model_connection_test_uses_submitted_values(product_env, monkeypatch):
    monkeypatch.setattr(OpenAICompatProvider, "chat", lambda self, system, user: "OK")
    response = client.post("/api/config/test", json={
        "vendor": "custom", "base_url": "https://llm.example/v1",
        "model": "test-model", "api_key": "key",
    })
    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert response.json()["model"] == "test-model"


def test_compute_config_and_running_guard(product_env, monkeypatch):
    import dft_forge.tools.graph_tools as graph_tools

    monkeypatch.setattr(graph_tools, "reset_engine_cache", lambda: None)
    response = client.post("/api/config/compute", json={"executor": "fake", "scheduler": "none", "walltime": 600})
    assert response.status_code == 200
    assert response.json()["executor"] == "fake"
    web.WEB_ACTIVE_RUNS["busy"] = {"cancel": None}
    blocked = client.post("/api/config/compute", json={"executor": "local", "scheduler": "none", "walltime": 600})
    assert blocked.status_code == 409


_SI_CIF = b"""data_Si
_cell_length_a 5.43
_cell_length_b 5.43
_cell_length_c 5.43
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Si1 Si 0 0 0
Si2 Si 0.25 0.25 0.25
"""


def test_structure_upload_artifacts_and_reproducibility(product_env):
    upload = client.post(
        "/api/sessions/product-session/structure",
        files={"file": ("silicon.cif", _SI_CIF, "chemical/x-cif")},
    )
    assert upload.status_code == 200
    structure = upload.json()
    assert structure["natoms"] == 2
    assert structure["formula"] == "Si2"
    artifacts = client.get("/api/sessions/product-session/artifacts").json()["artifacts"]
    assert any(item["kind"] == "structure" for item in artifacts)
    assert all(item["download_url"].startswith("/api/sessions/product-session/") for item in artifacts)
    manifest = client.get("/api/sessions/product-session/reproducibility")
    assert manifest.status_code == 200
    assert manifest.json()["schema"] == "dft-forge/reproducibility-v1"
    assert "ssh_host" not in manifest.json()["compute"]


def test_research_mode_requires_approval(product_env, monkeypatch):
    denied = client.post("/chat", json={
        "message": "计算 Si 的能带", "session_id": "approval-session",
        "run_config": {"mode": "research", "accuracy": "balanced"}, "approved": False,
    })
    assert denied.status_code == 409

    async def fake_run(*args, **kwargs):
        return {"reply": "ok", "commands": [], "results": [], "chain": [], "ctx": {}}

    monkeypatch.setattr(web.agent_loop, "run", fake_run)
    allowed = client.post("/chat", json={
        "message": "计算 Si 的能带", "session_id": "approval-session",
        "run_config": {"mode": "research", "accuracy": "balanced"}, "approved": True,
    })
    assert allowed.status_code == 200


def test_workspace_surface_is_present(product_env):
    html = client.get("/").text
    for element_id in (
        "providerGrid", "cfgTest", "attachStructure", "runMode", "methodDlg",
        "artifactList", "exportEvidence", "retryRun", "resumeRun",
    ):
        assert f'id="{element_id}"' in html
