"""Product modes guard actual tools; text wording cannot grant approval."""

import asyncio

import httpx
import pytest

import dft_forge.web as web
from dft_forge.agent_loop.execution_policy import ExecutionPolicyError, check_execution_policy
from dft_forge.agent_loop.executor import run_session
from dft_forge.tools.models import ToolEntry
from dft_forge.tools.registry import ToolRegistry


class Planner:
    last_plan_ir = None

    def __init__(self, steps):
        self.steps = steps

    def plan(self, *args):
        return self.steps, None

    def _get_provider(self):
        return None


@pytest.fixture
def mode_env(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "WEB_WORKDIR", tmp_path / "sessions")
    monkeypatch.setattr(web, "WEB_SESSION_CTX", {})
    monkeypatch.setattr(web, "WEB_ACTIVE_RUNS", {})
    monkeypatch.setenv("DFT_FORGE_EXECUTOR", "fake")
    monkeypatch.setenv("DFT_FORGE_LLM_PROVIDER", "dummy")
    return tmp_path


def test_missing_approval_blocks_entire_plan_before_preparation(mode_env):
    calls = []
    registry = ToolRegistry()
    for tool_id, effect in [("structure.generate", "workspace_write"), ("graph.run", "external_execution")]:
        registry.register(ToolEntry(
            id=tool_id, name=tool_id, description="test", category="test",
            side_effect=effect, input_schema={}, execute_fn=lambda args: calls.append(args) or {},
        ))
    steps = [
        {"tool": "structure.generate", "args": {"source": "Si"}},
        {"tool": "graph.run", "args": {"template_id": "t0_scf", "inputs": {"material": "Si"}}},
    ]
    planner = Planner(steps)
    with pytest.raises(ExecutionPolicyError, match="确认计算方法"):
        asyncio.run(run_session(
            "算这个", mode_env, ctx={"run_config": {"mode": "research", "approved": False}},
            registry=registry, llm_planner=planner, rule_planner=planner,
        ))
    assert calls == []
    assert not (mode_env / "generated.cif").exists()


@pytest.mark.parametrize("config", [
    {"mode": "typo"}, {"accuracy": "unbounded"}, {"ecutwfc": -1}, {"ecutwfc": True},
    {"ecutwfc": "30"}, {"nbnd": 1.5}, {"nbnd": True}, {"nkpoints_bands": 2},
    {"kpoints": [4, 4, 4]}, {"kpoints": [4, 0, 4, 1, 1, 1]},
    {"kpoints": [4, 4, 4, 2, 1, 1]}, {"kpoints": [True, 4, 4, 1, 1, 1]},
    {"undeclared_parameter": 5},
])
def test_invalid_method_rejected_before_session_mutation(mode_env, config):
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web.app), base_url="http://test") as client:
            response = await client.post("/chat", json={"message": "help", "session_id": "invalid", "run_config": config})
            assert response.status_code == 422, response.text
            assert "invalid" not in web.WEB_SESSION_CTX
            assert not (web.WEB_WORKDIR / "invalid").exists()
    asyncio.run(scenario())


def test_research_fake_and_demo_real_are_rejected_but_help_is_allowed(mode_env, monkeypatch):
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web.app), base_url="http://test") as client:
            common = {"message": "算 Si 的能带", "session_id": "mode-test"}
            research = await client.post("/chat", json={**common, "run_config": {"mode": "research"}, "approved": True})
            assert research.status_code == 409
            assert "模拟后端" in research.json()["detail"]
            monkeypatch.setenv("DFT_FORGE_EXECUTOR", "local")
            demo = await client.post("/chat", json={**common, "run_config": {"mode": "demo"}})
            assert demo.status_code == 409
            assert "演示模式" in demo.json()["detail"]
            help_response = await client.post("/chat", json={"message": "help", "run_config": {"mode": "research"}, "approved": False})
            assert help_response.status_code == 200
            assert help_response.json()["commands"][0]["command"] == "help"
            assert not (web.WEB_WORKDIR / "mode-test" / "graphs").exists()
    asyncio.run(scenario())


def test_plan_only_flag_and_method_parameters_do_not_leak_into_next_turn(mode_env):
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web.app), base_url="http://test") as client:
            first = await client.post("/chat", json={
                "message": "生成 Si 结构", "session_id": "modes", "execution_mode": "plan",
                "run_config": {"mode": "research", "ecutwfc": 80},
            })
            assert first.status_code == 200
            assert first.json()["commands"] == []
            second = await client.post("/chat", json={"message": "生成 Si 结构", "session_id": "modes"})
            assert second.status_code == 200
            assert second.json()["commands"]
            state = (await client.get("/api/sessions/modes")).json()["ctx"]
            assert "requested_execution_mode" not in state
            assert "ecutwfc" not in state["run_config"]
            assert state["run_config"]["approved"] is False
    asyncio.run(scenario())


def test_demo_cannot_reach_legacy_real_job_submit(mode_env):
    registry = ToolRegistry()
    registry.register(ToolEntry(
        id="job.submit", name="job", description="job", category="test",
        side_effect="external_execution", input_schema={}, execute_fn=lambda args: {},
    ))
    with pytest.raises(ExecutionPolicyError, match="不一致"):
        check_execution_policy([{"tool": "job.submit", "args": {}}], {"run_config": {"mode": "demo"}}, registry)


@pytest.mark.parametrize("number", ["NaN", "Infinity", "1e999"])
def test_nonfinite_method_values_return_valid_json_errors(mode_env, number):
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web.app), base_url="http://test") as client:
            response = await client.post(
                "/chat", headers={"Content-Type": "application/json"},
                content='{"message":"help","run_config":{"ecutwfc":' + number + '}}',
            )
            assert response.status_code == 422
            assert response.json()["detail"][0]["loc"][-1] == "ecutwfc"
    asyncio.run(scenario())


def test_web_omitting_mode_cannot_bypass_approval(mode_env):
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web.app), base_url="http://test") as client:
            request = {"message": "算 Si 的能带", "session_id": "unapproved"}
            response = await client.post("/chat", json=request)
            assert response.status_code == 409
            assert response.headers["X-DFT-Policy"] == "approval_required"
            stream = await client.post("/chat/stream", json=request)
            assert '"code": "approval_required"' in stream.text
            assert not (web.WEB_WORKDIR / "unapproved" / "graphs").exists()
    asyncio.run(scenario())
