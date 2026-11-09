"""Per-run evidence is a read-only snapshot, scoped to one graph."""

import asyncio
import time

import httpx
import pytest

import dft_forge.web as web
from dft_forge.runtime.engine import GraphEngine
from dft_forge.runtime.graph import GraphTemplate, NodeSpec
from dft_forge.runtime.store import sqlite_connection


class Tool:
    name = "offline"

    def execute(self, node, ctx):
        directory = ctx.node_workdir(node.node_id)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "result #1.out").write_text("offline output")
        return {"energy_ry": -2.5, "curve": [1, 2, 3]}


@pytest.fixture
def evidence_env(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "WEB_WORKDIR", tmp_path / "sessions")
    monkeypatch.setattr(web, "WEB_SESSION_CTX", {})
    monkeypatch.setattr(web, "WEB_ACTIVE_RUNS", {})
    root = tmp_path / "sessions" / "sample"
    engine = GraphEngine(
        root / "graphs", tools={"offline": Tool()},
        execution_identity="exec_" + "a" * 64,
        execution_profile={"backend": "fake", "simulated": True, "walltime_limit_sec": 30,
                           "host": "private-host-fixture", "ssh_key": "/fixture/secret-key"},
    )
    engine.register_template(GraphTemplate(
        template_id="evidence", inputs={"cutoff": 30},
        nodes=[NodeSpec(id="scf", tool="offline", params={"ecutwfc": "${inputs.cutoff}"})],
    ))
    return root, engine


def request(run_id):
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web.app), base_url="http://test") as client:
            return await client.get(f"/api/sessions/sample/runs/{run_id}/evidence")
    return asyncio.run(scenario())


def test_evidence_contains_only_selected_run_and_persisted_method(evidence_env):
    root, engine = evidence_env
    selected = engine.run_template("evidence", {"cutoff": 55})
    other = engine.run_template("evidence", {"cutoff": 80})
    (root / "graphs" / selected.run_id / "other-output.out").symlink_to(
        root / "graphs" / other.run_id / "scf" / "result #1.out"
    )
    web.WEB_SESSION_CTX["sample"] = {"messages": [{"text": "unrelated private conversation"}]}
    response = request(selected.run_id)
    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "no-store"
    assert selected.run_id in response.headers["content-disposition"]
    data = response.json()
    assert data["schema"] == "dft-forge/run-evidence-v1"
    assert data["run"]["run_id"] == selected.run_id
    assert data["run"]["execution"]["origin"]["profile"]["simulated"] is True
    assert data["run"]["template_snapshot"]["inputs"]["cutoff"] == 30
    assert data["nodes"][0]["resolved_params"]["ecutwfc"] == 55
    assert data["nodes"][0]["scalar_outputs"] == {"energy_ry": -2.5}
    assert len(data["artifacts"]) == 1
    assert data["artifacts"][0]["hash_status"] == "computed"
    assert all(item["path"].startswith(f"graphs/{selected.run_id}/") for item in data["artifacts"])
    assert other.run_id not in response.text
    assert "private-host-fixture" not in response.text
    assert "secret-key" not in response.text
    assert "unrelated private conversation" not in response.text
    assert data["scientific_validation"] == "not_assessed"
    assert data["snapshot"] == {"live": False, "database_consistent": True, "artifacts_atomic": False}


def test_created_run_stays_unresolved_and_is_exported_as_live(evidence_env):
    _, engine = evidence_env
    run = engine.create("evidence")
    before = engine.status(run.run_id)
    exported = request(run.run_id).json()
    assert exported["snapshot"]["live"]
    assert "run_still_active" in exported["notices"]
    assert exported["nodes"][0]["params_resolved"] is False
    assert exported["nodes"][0]["resolved_params"] is None
    assert engine.status(run.run_id) == before


@pytest.mark.parametrize("broken", [False, True])
def test_legacy_or_corrupt_template_never_substitutes_current_template(evidence_env, broken):
    _, engine = evidence_env
    run = engine.create("evidence")
    with sqlite_connection(engine.store.path) as conn:
        if broken:
            conn.execute("UPDATE graph_run_templates SET template_json = 'broken' WHERE run_id = ?", (run.run_id,))
        else:
            conn.execute("DROP TABLE graph_run_templates")
    result = request(run.run_id)
    assert result.status_code == 200
    assert result.json()["run"]["template_snapshot"] is None
    assert result.json()["run"]["template_snapshot_status"] == ("invalid" if broken else "unavailable_legacy")


def test_paths_and_sensitive_parameters_are_not_exported(evidence_env):
    root, engine = evidence_env
    engine.get_template("evidence").inputs.update({"structure": None, "pseudo_dir": None, "api_key": None})
    run = engine.create("evidence", {"cutoff": 40, "structure": str(root / "uploads" / "input.cif"),
                                     "pseudo_dir": "/opt/private-fixture/pseudo", "api_key": "fixture-not-a-real-key"})
    result = request(run.run_id)
    assert str(root) not in result.text
    assert "private-fixture" not in result.text
    assert "fixture-not-a-real-key" not in result.text
    assert result.json()["run"]["inputs"]["structure"] == "./uploads/input.cif"
    assert "private_fields_omitted" in result.json()["notices"]


def test_ambiguous_run_id_is_rejected(evidence_env):
    root, engine = evidence_env
    run = engine.create("evidence")
    other = GraphEngine(root / "workflows" / "other" / "graphs", tools={"offline": Tool()})
    other.store.save_run(run, engine.get_template("evidence"))
    assert request(run.run_id).status_code == 409


def test_nonportable_identity_is_explicitly_marked_as_omitted(evidence_env):
    _, engine = evidence_env
    run = engine.create("evidence")
    run.metadata["execution"]["origin"]["identity"] = "private-host-fixture.local"
    engine.store.save_run(run)
    response = request(run.run_id)
    assert response.status_code == 200
    assert "private-host-fixture.local" not in response.text
    origin = response.json()["run"]["execution"]["origin"]
    assert origin["identity"] is None
    assert origin["identity_status"] == "omitted_nonportable"
    assert "execution_identity_omitted" in response.json()["notices"]


def test_hash_budget_and_large_file_status_are_explicit(tmp_path):
    path = tmp_path / "large.out"
    with path.open("wb") as stream:
        stream.truncate(32 * 1024 * 1024 + 1)
    assert web._artifact_digest(path, path.stat(), time.monotonic() + 2) == (None, "omitted_large")
    small = tmp_path / "small.out"
    small.write_text("data")
    assert web._artifact_digest(small, small.stat(), time.monotonic() - 1) == (None, "time_budget_exceeded")
    before = small.stat()
    small.write_text("data changed while preparing the snapshot")
    assert web._artifact_digest(small, before, time.monotonic() + 2) == (None, "changed_during_read")
    before = small.stat()
    small.unlink()
    assert web._artifact_digest(small, before, time.monotonic() + 2) == (None, "read_error")
