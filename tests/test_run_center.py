"""Run-center regressions use temporary state and cooperative offline tools."""

import asyncio
import threading
from pathlib import Path

import httpx
import pytest

import dft_forge.web as web
import dft_forge.tools.graph_tools as graph_tools
from dft_forge.runtime.engine import GraphEngine
from dft_forge.runtime.graph import GraphTemplate, NodeSpec
from dft_forge.runtime.run import ToolError


class Tool:
    name = "test"

    def __init__(self, fail=False, block=False):
        self.fail, self.block = fail, block
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def execute(self, node, ctx):
        self.calls += 1
        self.started.set()
        if self.block and not self.release.wait(5):
            raise RuntimeError("test timeout")
        if self.fail:
            raise ToolError("SCF did not converge")
        return {"energy_ry": -10.2, "large_curve": [1, 2, 3]}

    def cancel_all(self):
        self.release.set()
        return 1


@pytest.fixture
def run_env(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "WEB_WORKDIR", tmp_path / "sessions")
    monkeypatch.setattr(web, "ENV_FILE", tmp_path / ".env")
    monkeypatch.setattr(web, "WEB_SESSION_CTX", {})
    monkeypatch.setattr(web, "WEB_ACTIVE_RUNS", {})
    monkeypatch.setattr(graph_tools, "_ENGINE_CACHE", {})
    monkeypatch.setenv("DFT_FORGE_EXECUTOR", "fake")
    return tmp_path / "sessions"


def make_run(root: Path, *, session="sample", nested=False, fail=False):
    base = root / session
    base = base / "workflows" / "convergence" / "graphs" if nested else base / "graphs"
    tool = Tool(fail=fail)
    engine = GraphEngine(base, tools={tool.name: tool})
    engine.register_template(GraphTemplate(
        template_id="dynamic_test",
        nodes=[NodeSpec(id="scf", tool=tool.name, max_repair_attempts=0)],
    ))
    run = engine.run_template("dynamic_test")
    graph_tools._ENGINE_CACHE[str(base)] = engine
    return engine, run


def test_nested_runs_and_scalar_details_are_readable(run_env):
    async def scenario():
        _, run = make_run(run_env, nested=True)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web.app), base_url="http://test") as client:
            listing = (await client.get("/api/sessions/sample/runs")).json()["runs"]
            assert listing[0]["run_id"] == run.run_id
            assert listing[0]["workspace"] == "workflows/convergence/graphs"
            detail = (await client.get(f"/api/sessions/sample/runs/{run.run_id}")).json()
            assert detail["nodes"]["scf"]["outputs"] == {"energy_ry": -10.2}
            assert detail["actions"] == {"retry": False, "resume": False}
            assert (await client.get(f"/api/sessions/other/runs/{run.run_id}")).status_code == 404
            assert not (run_env / "other").exists()
    asyncio.run(scenario())


def test_recovery_is_owned_cancellable_and_isolated(run_env):
    async def scenario():
        engine, run = make_run(run_env, fail=True)
        blocker = Tool(block=True)
        engine.tools["test"] = blocker
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web.app), base_url="http://test") as client:
            response = await client.post("/api/sessions/sample/runs/action", json={
                "action": "retry", "run_id": run.run_id, "background": True,
            })
            assert response.status_code == 202, response.text
            entry = web.WEB_ACTIVE_RUNS["sample"]
            try:
                assert await asyncio.to_thread(blocker.started.wait, 2)
                active = (await client.get("/api/sessions/sample/running")).json()
                assert active["running"] and active["run_id"] == run.run_id
                assert (await client.post("/chat", json={"session_id": "sample", "message": "help"})).status_code == 409
                assert (await client.post("/chat/stream", json={"session_id": "sample", "message": "help"})).status_code == 409
                assert (await client.delete("/api/sessions/sample")).status_code == 409
                assert (await client.post("/api/config", json={"provider": "dummy"})).status_code == 409
                assert (await client.post("/api/config/compute", json={"executor": "fake"})).status_code == 409
                assert (await client.post("/api/sessions/sample/runs/action", json={"action": "retry", "run_id": run.run_id})).status_code == 409
                assert not (await client.get("/api/sessions/other/running")).json()["running"]
                assert (await client.post("/api/sessions/other/cancel")).json()["cancelled"] is False
                stopped = await client.post("/api/sessions/sample/cancel")
                assert stopped.json()["cancelled"] and entry["cancel"].is_set()
                await asyncio.wait_for(entry["task"], timeout=3)
                assert not (await client.get("/api/sessions/sample/running")).json()["running"]
                detail = (await client.get(f"/api/sessions/sample/runs/{run.run_id}")).json()
                assert detail["state"] == "paused"
                assert detail["actions"]["resume"]
            finally:
                entry["cancel"].set()
                blocker.release.set()
                await asyncio.wait_for(entry["task"], timeout=3)
    asyncio.run(scenario())


def test_sync_recovery_compatibility_and_explicit_run_target(run_env):
    async def scenario():
        engine, first = make_run(run_env, fail=True)
        second = engine.run_template("dynamic_test")
        engine.tools["test"] = Tool()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web.app), base_url="http://test") as client:
            response = await client.post("/api/sessions/sample/runs/action", json={"action": "retry", "run_id": first.run_id})
            assert response.status_code == 200, response.text
            assert response.json()["state"] == "succeeded"
            assert engine.status(second.run_id)["state"] == "failed"
            rejected = await client.post("/api/sessions/sample/runs/action", json={"action": "retry", "run_id": first.run_id})
            assert rejected.status_code == 409
    asyncio.run(scenario())


def test_old_dynamic_graph_without_snapshot_is_inspectable(run_env):
    import sqlite3
    from contextlib import closing

    engine, run = make_run(run_env, fail=True)
    with closing(sqlite3.connect(engine.store.path)) as conn, conn:
        conn.execute("DROP TABLE IF EXISTS graph_run_templates")
    detail = web.inspect_session_run("sample", run.run_id)
    assert detail["nodes"]["scf"]["error"]
    assert not any(detail["actions"].values())
    assert "模板" in detail["action_reason"]


def test_nonstream_chat_ownership_and_request_survives_reload(run_env, monkeypatch):
    async def scenario():
        started, release = asyncio.Event(), asyncio.Event()

        async def fake_run(*args, ctx, **kwargs):
            started.set()
            await release.wait()
            return {"reply": "ok", "commands": [], "results": []}

        monkeypatch.setattr(web.agent_loop, "run", fake_run)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web.app), base_url="http://test") as client:
            request = asyncio.create_task(client.post("/chat", json={"session_id": "sample", "message": "help"}))
            await asyncio.wait_for(started.wait(), 2)
            try:
                assert (await client.get("/api/sessions/sample/running")).json()["running"]
                session = await client.get("/api/sessions/sample")
                assert session.status_code == 200
                assert session.json()["messages"][0]["text"] == "help"
                assert "cancel_event" not in session.json()["ctx"]
                assert (await client.post("/chat", json={"session_id": "sample", "message": "again"})).status_code == 409
            finally:
                release.set()
                assert (await request).status_code == 200
            assert not web.WEB_ACTIVE_RUNS
    asyncio.run(scenario())


def test_old_owner_cannot_release_a_new_operation(run_env):
    first = web._claim_operation("sample", "chat")
    web._release_operation("sample", first)
    second = web._claim_operation("sample", "retry", "run_new")
    web._release_operation("sample", first)
    assert web.WEB_ACTIVE_RUNS["sample"] is second


def test_listing_opens_one_connection_per_database_not_per_run(run_env, monkeypatch):
    import dft_forge.web.run_views as views

    engine, _ = make_run(run_env, fail=True)
    for _ in range(15):
        engine.create("dynamic_test")
    original = views._connect
    opened = []

    def counted(path):
        opened.append(path)
        return original(path)

    monkeypatch.setattr(views, "_connect", counted)
    result = web.list_session_runs("sample")["runs"]
    assert len(result) == 16
    assert len(opened) == 1
    assert all(item["has_template_snapshot"] for item in result)


def test_failed_recovery_preflight_releases_session_ownership(run_env):
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web.app), base_url="http://test") as client:
            response = await client.post("/api/sessions/missing/runs/action", json={"action": "resume"})
            assert response.status_code == 404
            assert not web.WEB_ACTIVE_RUNS
    asyncio.run(scenario())


def test_workflow_cancel_handle_and_events_reach_graph(run_env, monkeypatch):
    from dft_forge.workflows import run_convergence
    from dft_forge.agent_loop.step_prep import prepare_step_args

    root = run_env / "sample"
    tool = Tool(block=True)
    engine = GraphEngine(root / "workflows" / "workflow_convergence" / "graphs", tools={"qe": tool}, max_concurrency=1)
    graph_tools._ENGINE_CACHE[str(engine.base_dir)] = engine
    cancel = threading.Event()
    cancel.set()  # A stopped macro must not dispatch even its first node.
    events = []
    args = {"material": "Si", "ecutwfc_values": [20, 30]}
    prepare_step_args(
        "workflow.convergence", args, session_dir=root, ctx={"cancel_event": cancel},
        built_structure=None, scf_energies=[], node_log=[], emit=events.append, idx=0,
    )
    result = run_convergence(args)
    assert result["state"] == "paused"
    assert tool.calls == 0


def test_disconnected_chat_keeps_ownership_until_worker_finishes(run_env, monkeypatch):
    async def scenario():
        started, release = asyncio.Event(), asyncio.Event()

        async def fake_run(*args, **kwargs):
            started.set()
            await release.wait()
            return {"reply": "finished", "commands": [], "results": []}

        monkeypatch.setattr(web.agent_loop, "run", fake_run)
        request = asyncio.create_task(web.chat(web.ChatRequest(message="help", session_id="sample")))
        await started.wait()
        entry = web.WEB_ACTIVE_RUNS["sample"]
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        assert web.WEB_ACTIVE_RUNS["sample"] is entry
        release.set()
        await entry["task"]
        assert not web.WEB_ACTIVE_RUNS
        assert web.get_session("sample")["messages"][-1]["text"] == "finished"
    asyncio.run(scenario())


def test_stream_error_is_saved_and_finishes_without_stuck_owner(run_env, monkeypatch):
    async def fail(*args, **kwargs):
        raise RuntimeError("offline test failure")

    monkeypatch.setattr(web.agent_loop, "run", fail)

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web.app), base_url="http://test") as client:
            response = await client.post("/chat/stream", json={"session_id": "sample", "message": "help"})
            assert response.status_code == 200
            assert '"type": "error"' in response.text
            assert '"type": "done"' in response.text
            assert not web.WEB_ACTIVE_RUNS
            session = (await client.get("/api/sessions/sample")).json()
            assert "offline test failure" in session["messages"][-1]["text"]
    asyncio.run(scenario())


def test_empty_chat_is_rejected_without_creating_session(run_env):
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web.app), base_url="http://test") as client:
            assert (await client.post("/chat", json={"session_id": "empty", "message": "  "})).status_code == 400
            assert not (run_env / "empty").exists()
    asyncio.run(scenario())


def test_uploaded_structure_preview_survives_session_reload(run_env):
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web.app), base_url="http://test") as client:
            source = Path(__file__).parent / "fixtures" / "Si.cif"
            uploaded = await client.post("/api/sessions/sample/structure", files={
                "file": ("silicon.cif", source.read_bytes(), "chemical/x-cif"),
            })
            assert uploaded.status_code == 200
            original = uploaded.json()
            web.WEB_SESSION_CTX.clear()
            restored = await client.get("/api/sessions/sample/structure")
            assert restored.status_code == 200
            assert restored.json()["cif"] == original["cif"]
            assert restored.json()["path"] == original["path"]
            assert restored.json()["natoms"] == original["natoms"]
    asyncio.run(scenario())


def test_corrupt_snapshot_refuses_recovery_but_keeps_details(run_env):
    import sqlite3
    from contextlib import closing

    engine, run = make_run(run_env, fail=True)
    with closing(sqlite3.connect(engine.store.path)) as conn, conn:
        conn.execute("UPDATE graph_run_templates SET template_json = ? WHERE run_id = ?", ("broken", run.run_id))

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web.app), base_url="http://test") as client:
            assert (await client.get(f"/api/sessions/sample/runs/{run.run_id}")).status_code == 200
            rejected = await client.post("/api/sessions/sample/runs/action", json={
                "action": "retry", "run_id": run.run_id, "background": True,
            })
            assert rejected.status_code == 409
            assert "snapshot" in rejected.json()["detail"]
            assert not web.WEB_ACTIVE_RUNS
    asyncio.run(scenario())
