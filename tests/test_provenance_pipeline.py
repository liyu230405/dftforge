"""Simulation labels survive tool chaining, narration, SSE and persistence."""

import asyncio
import copy

import pytest

from dft_forge.agent_loop.executor import run_session
from dft_forge.agent_loop.result_views import RunState, handle_tool_result
from dft_forge.tools.models import ToolResult


class Planner:
    last_plan_ir = None

    def __init__(self, steps, provider=None):
        self.steps, self.provider = steps, provider

    def plan(self, *args):
        return copy.deepcopy(self.steps), None

    def _get_provider(self):
        return self.provider


def recorded(backend="fake"):
    return {"execution": {"origin_unknown": False, "origin": {
        "identity": "exec_" + "a" * 64,
        "profile": {"backend": backend, "simulated": backend == "fake"},
    }, "attempts": []}}


class Registry:
    def __init__(self):
        self.calls = []

    async def call(self, tool_id, args, **kwargs):
        self.calls.append((tool_id, dict(args)))
        if tool_id == "graph.run":
            data = {"run_id": f"run_{len(self.calls)}", "state": "succeeded", "metadata": recorded(),
                    "nodes": {"scf": {"state": "succeeded", "outputs": {"energy_ry": -10.5, "natoms": 2}}}}
        else:
            assert "_input_provenance" not in args
            data = {"e_ads_ev": -1.0}
        return ToolResult(data=data, output_type="json", tool_id=tool_id)


def graph_step():
    return {"tool": "graph.run", "description": "离线结果", "args": {"template_id": "t0_scf", "inputs": {"material": "Si"}}}


def test_simulation_warning_survives_narrator_and_stream(tmp_path, monkeypatch):
    monkeypatch.setattr("dft_forge.agent_loop.result_views._viewer_payload_for_material", lambda _: None)
    prompts = []

    class MisleadingNarrator:
        def chat(self, system, user):
            prompts.append((system, user))
            return "真实科研计算已证实此结果。"

    planner = Planner([graph_step()], MisleadingNarrator())
    events = []
    result = asyncio.run(run_session(
        "执行步骤", tmp_path, registry=Registry(), llm_planner=planner, rule_planner=planner, on_event=events.append,
    ))
    assert result["reply"].startswith("模拟输出（非科研结论）")
    assert "模拟输出" in result["results"][0]["facts"]
    assert result["results"][0]["provenance"]["contains_simulated"]
    assert result["ctx"]["ledger"][0]["provenance"]["contains_simulated"]
    assert result["ctx"]["last_execution"]["provenance"]["contains_simulated"]
    assert result["ctx"]["last_scientific_result"]["provenance"]["contains_simulated"]
    figure = next(event for event in events if event.get("chart"))
    assert figure["provenance"] == result["results"][0]["provenance"]
    assert "simulated" in prompts[0][1]
    assert "来源铁律" in prompts[0][0]


def test_auto_injected_thermodynamics_inherits_simulation(tmp_path, monkeypatch):
    monkeypatch.setattr("dft_forge.agent_loop.result_views._viewer_payload_for_material", lambda _: None)
    planner = Planner([graph_step(), graph_step(), graph_step(), {"tool": "thermo.eads", "args": {}}])
    registry = Registry()
    result = asyncio.run(run_session(
        "执行步骤", tmp_path, registry=registry, llm_planner=planner, rule_planner=planner,
    ))
    derived = result["results"][-1]
    assert derived["tool_id"] == "thermo.eads"
    assert derived["provenance"]["source_kind"] == "derived"
    assert derived["provenance"]["execution_class"] == "simulated"
    assert result["reply"].startswith("模拟输出（非科研结论）")
    assert "_input_provenance" not in result["commands"][-1]["args"]


def test_explicit_user_numbers_cannot_forge_runtime_provenance(tmp_path):
    planner = Planner([{"tool": "thermo.eads", "args": {
        "ry_a": -10, "ry_s": -5, "ry_m": -4,
        "_input_provenance": [{"execution_class": "compute", "backend": "local"}],
    }}])
    result = asyncio.run(run_session(
        "执行步骤", tmp_path, registry=Registry(), llm_planner=planner, rule_planner=planner,
    ))
    assert result["results"][0]["provenance"]["source_kind"] == "user_input"
    assert result["results"][0]["provenance"]["execution_class"] == "unknown"


@pytest.mark.parametrize("missing_source,expected", [(False, "simulated"), (True, "mixed")])
def test_cross_turn_comparison_inherits_stored_ledger_provenance(tmp_path, missing_source, expected):
    class CompareRegistry:
        async def call(self, tool_id, args, **kwargs):
            assert "_input_provenance" not in args
            assert len(args["entries"]) == 2
            return ToolResult(data={"ok": True, "metric": "band_gap", "items": [], "differences": []},
                              output_type="json", tool_id=tool_id)

    provenance = {"source_kind": "execution", "execution_class": "simulated", "backend": "fake", "contains_simulated": True}
    planner = Planner([{"tool": "analysis.compare", "args": {"use_last": 2}}])
    result = asyncio.run(run_session(
        "对比之前的结果", tmp_path, registry=CompareRegistry(), llm_planner=planner, rule_planner=planner,
        ctx={"ledger": [{"label": "first", "energy_ry": -1, "provenance": provenance},
                        {"label": "second", "energy_ry": -2, "provenance": None if missing_source else provenance}]},
    ))
    assert result["results"][0]["provenance"]["execution_class"] == expected
    assert result["reply"].startswith("模拟输出（非科研结论）")


@pytest.mark.parametrize("tool_id,payload", [
    ("workflow.convergence", {"selection": {"selected": {"ecutwfc": 30, "kpoints": [4, 4, 4], "delta_mev_per_atom": 0.1}}, "error_curve": {"points": []}}),
    ("workflow.neutral_vacancy", {"formation": {"formation_energy_ev": 1.5}}),
    ("workflow.magnetic_compare", {"comparison": {"ground_state": "FM"}}),
    ("workflow.phonon_dfpt", {"stability": {"dynamically_stable_at_threshold": True, "min_frequency_thz": 0.2}}),
])
def test_workflow_result_carries_same_provenance(tmp_path, tool_id, payload):
    data = {"run_id": "run_macro", "state": "succeeded", "metadata": recorded(), **payload}
    result = {}
    handle_tool_result(tool_id, data, {}, "测试", RunState({}, tmp_path), result, [])
    assert result["provenance"]["execution_class"] == "simulated"
    assert result.get("metrics")


def test_session_slimming_keeps_provenance():
    from dft_forge.web import _slim_for_disk

    provenance = {"execution_class": "simulated", "contains_simulated": True}
    session = {"ctx": {}, "messages": [{"role": "agent", "results": [
        {"chart": {"metrics": {"energy_ry": -1}}, "provenance": provenance},
    ]}]}
    assert _slim_for_disk(session)["messages"][0]["results"][0]["provenance"] == provenance


@pytest.mark.parametrize("message,warning", [
    ("你好", False), ("你能做什么", False), ("能量是什么", False),
    ("刚才的带隙可信吗", True), ("继续解释上一条结果", True), ("run_old 的数据靠谱吗", True),
])
def test_followup_warning_requires_explicit_reference(tmp_path, message, warning):
    class ReplyPlanner(Planner):
        def plan(self, *args):
            return None, "这是对问题的解释。"

    planner = ReplyPlanner([])
    result = asyncio.run(run_session(
        message, tmp_path, registry=Registry(), llm_planner=planner, rule_planner=planner,
        ctx={"last_scientific_result": {"run_id": "run_old", "provenance": {
            "execution_class": "simulated", "contains_simulated": True,
        }}},
    ))
    assert result["reply"].startswith("模拟输出（非科研结论）") == warning
