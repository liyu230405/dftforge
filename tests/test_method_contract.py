"""Method confirmation follows the actual plan, before any tool side effect."""

import asyncio
import copy

import httpx
import pytest

import dft_forge.web as web
from dft_forge.agent_loop.executor import run_session
from dft_forge.agent_loop.method_contract import MethodContractError, build_method_contract, check_method_contract
from dft_forge.runtime.engine import GraphEngine
from dft_forge.runtime.graph import GraphTemplate, NodeSpec
from dft_forge.tools.models import ToolEntry
from dft_forge.tools.registry import ToolRegistry


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("DFT_FORGE_EXECUTOR", "fake")
    monkeypatch.setenv("DFT_FORGE_LLM_PROVIDER", "dummy")


def step(tool="workflow.phonon_dfpt", **args):
    return {"tool": tool, "args": args}


def config(**values):
    return {"mode": "demo", "accuracy": "balanced", **values}


def test_precise_phonon_applies_mesh_and_discloses_irrelevant_band_path():
    contract = build_method_contract([step(material="MgO")], config(accuracy="precise", kpoints=[8, 8, 8, 1, 1, 1], nkpoints_bands=160))
    method = contract["steps"][0]
    assert method["applied"] == {"kpoints": [8, 8, 8, 1, 1, 1]}
    assert method["effective"]["q_grid"] == [2, 2, 2]
    assert "nkpoints_bands" in method["not_applicable"]
    assert contract["requires_confirmation"]


@pytest.mark.parametrize("tool,args", [("workflow.phonon_dfpt", {}), ("graph.run", {"template_id": "t0_scf", "inputs": {"material": "Si"}})])
def test_quick_preset_does_not_claim_to_accelerate_scf_or_phonons(tool, args):
    method = build_method_contract([step(tool, **args)], config(accuracy="quick", nkpoints_bands=60))["steps"][0]
    assert method["applied"] == {}
    assert any("不承诺加速" in note for note in method["notes"])


class Planner:
    last_plan_ir = None

    def __init__(self, steps):
        self.steps = steps

    def plan(self, *args):
        return copy.deepcopy(self.steps), None

    def _get_provider(self):
        return None


def registry_for(steps, calls):
    registry = ToolRegistry()
    for item in steps:
        tool = item["tool"]
        registry.register(ToolEntry(id=tool, name=tool, description="fixture", category="test", input_schema={},
                                    side_effect="external_execution" if tool.startswith("workflow.") or tool == "graph.run" else "workspace_write",
                                    execute_fn=lambda args, tool=tool: calls.append((tool, copy.copy(args))) or {}))
    return registry


def execute(steps, root, ctx, calls):
    planner = Planner(steps)
    return asyncio.run(run_session("执行步骤", root, ctx=ctx, registry=registry_for(steps, calls), llm_planner=planner, rule_planner=planner))


def test_unsupported_custom_field_blocks_preparation_prefix(tmp_path):
    steps = [step("structure.generate", source="Si"), step("workflow.neutral_vacancy", source="Si", element="Si")]
    calls = []
    with pytest.raises(MethodContractError) as denied:
        execute(steps, tmp_path, {"run_config": config(accuracy="custom", nbnd=20)}, calls)
    assert denied.value.code == "method_unsupported"
    assert denied.value.method_token is None
    assert not calls
    assert list(tmp_path.iterdir()) == []


def test_confirmed_macro_receives_all_bindings_and_private_metadata(tmp_path):
    steps = [step("workflow.magnetic_compare", material="Fe", _method_contract={"forged": True})]
    ctx = {"run_config": config(accuracy="custom", ecutwfc=70, kpoints=[6, 6, 6, 0, 0, 0])}
    calls = []
    with pytest.raises(MethodContractError) as denied:
        execute(steps, tmp_path, ctx, calls)
    assert not calls
    assert not list(tmp_path.iterdir())
    result = execute(steps, tmp_path, {**ctx, "requested_method_token": denied.value.method_token}, calls)
    assert len(calls) == 1
    assert calls[0][1]["ecutwfc"] == 70
    assert calls[0][1]["kpoints"] == [6, 6, 6, 0, 0, 0]
    assert calls[0][1]["_method_contract"]["confirmation"] == "effective_contract"
    assert "_method_contract" not in result["commands"][0]["args"]


def test_convergence_confirmation_uses_exact_grid_not_single_value(tmp_path):
    from dft_forge.workflows import convergence_grid

    steps = [step("workflow.convergence", material="Si")]
    ctx = {"run_config": config(accuracy="custom", ecutwfc=80, kpoints=[8, 8, 8, 1, 1, 1])}
    with pytest.raises(MethodContractError) as denied:
        check_method_contract(steps, ctx, tmp_path)
    row = denied.value.method_contract["steps"][0]
    assert row["effective"]["scan_points"] == convergence_grid()
    assert row["effective"]["job_count"] == 12
    assert set(row["conflicts"]) == {"ecutwfc", "kpoints"}
    assert row["applied"] == {}
    check_method_contract(steps, {**ctx, "requested_method_token": denied.value.method_token}, tmp_path)
    assert not list(tmp_path.iterdir())


def test_confirmation_invalidates_when_plan_file_or_backend_changes(tmp_path, monkeypatch):
    structure = tmp_path / "uploaded.cif"
    structure.write_text("first fixture")
    steps = [step("graph.run", template_id="t0_scf", inputs={"material": "Si"})]
    ctx = {"run_config": config(), "last_structure_source": str(structure), "session_id": "sample"}
    first = build_method_contract(steps, ctx["run_config"], ctx, tmp_path)
    ctx["requested_method_token"] = first["digest"]
    assert check_method_contract(steps, ctx, tmp_path)["digest"] == first["digest"]
    structure.write_text("second fixture")
    with pytest.raises(MethodContractError, match="实际方法"):
        check_method_contract(steps, ctx, tmp_path)
    structure.write_text("first fixture")
    monkeypatch.setenv("DFT_FORGE_WALLTIME", "3600")
    with pytest.raises(MethodContractError, match="实际方法"):
        check_method_contract(steps, ctx, tmp_path)
    monkeypatch.delenv("DFT_FORGE_WALLTIME")
    changed = copy.deepcopy(steps)
    changed[0]["args"]["inputs"]["material"] = "MgO"
    with pytest.raises(MethodContractError, match="实际方法"):
        check_method_contract(changed, ctx, tmp_path)


def test_multistep_records_each_method_and_structural_lineage(tmp_path):
    steps = [step("structure.generate", source="Si"), step("graph.run", template_id="t0_scf", inputs={"material": "Si"}),
             step("structure.dope", source="Si", element="P"), step("graph.run", template_id="t2_bands", inputs={"material": "Si"})]
    contract = build_method_contract(steps, config(ecutwfc=70), session_dir=tmp_path)
    assert [row["index"] for row in contract["steps"]] == [1, 3]
    assert [row["structure_input"]["from_step"] for row in contract["steps"]] == [0, 2]
    assert all(row["applied"]["ecutwfc"] == 70 for row in contract["steps"])


def test_historical_contract_cannot_grant_a_new_request(tmp_path):
    steps = [step(material="MgO")]
    ctx = {"run_config": config(), "effective_method_contract": {"forged": True}, "last_method_contract": {"forged": True}}
    with pytest.raises(MethodContractError):
        check_method_contract(steps, ctx, tmp_path)
    assert "effective_method_contract" not in ctx
    web._apply_request_context(web.ChatRequest(message="help"), tmp_path, ctx)
    assert "requested_method_token" not in ctx


def test_direct_agent_without_product_config_keeps_api(tmp_path):
    calls = []
    execute([step(material="MgO")], tmp_path, {}, calls)
    assert len(calls) == 1


def test_actual_graph_saves_confirmed_method_in_evidence(tmp_path, monkeypatch):
    from dft_forge.tools.graph_tools import _graph_run, _ENGINE_CACHE, DEFAULT_TEMPLATES
    from dft_forge.web.evidence import build_evidence
    from dft_forge.web.run_views import evidence_snapshot
    from dft_forge.agent_loop.method_contract import method_for_step

    class FixtureTool:
        name = "qe"
        def execute(self, node, ctx):
            return {"energy_ry": -10, "natoms": 2}

    engine = GraphEngine(tmp_path / "graphs", tools={"qe": FixtureTool()}, templates_dir=DEFAULT_TEMPLATES,
                         execution_identity="fixture", execution_profile={"backend": "fake", "simulated": True})
    monkeypatch.setitem(_ENGINE_CACHE, str(tmp_path / "graphs"), engine)

    steps = [step("graph.run", template_id="t0_scf", inputs={"material": "Si"})]
    method = check_method_contract(steps, {"run_config": config(accuracy="custom", ecutwfc=67, kpoints=[3, 3, 3, 0, 0, 0])}, tmp_path)
    args = {"template_id": "t0_scf", "inputs": {"material": "Si", **method["steps"][0]["applied"]},
            "workdir": str(tmp_path / "graphs"), "_method_contract": method_for_step(method, 0)}
    result = _graph_run(args)
    assert result["state"] == "succeeded", result
    snapshot = evidence_snapshot(tmp_path, result["run_id"])
    evidence = build_evidence(snapshot, tmp_path, [])
    assert evidence["run"]["method_contract"]["step"]["effective"]["ecutwfc"] == 67
    assert evidence["nodes"][0]["resolved_params"]["ecutwfc"] == 67
    assert evidence["run"]["method_contract"]["scope"] == "requested_inputs_not_scientific_validation"


def test_nan_planner_parameter_is_reported_as_method_error(tmp_path):
    with pytest.raises(MethodContractError) as denied:
        check_method_contract([step("graph.run", template_id="t0_scf", inputs={"ecutwfc": float("nan")})], {"run_config": config()}, tmp_path)
    assert denied.value.code == "method_invalid"
    assert not list(tmp_path.iterdir())


def test_method_metadata_exists_at_first_persistence(tmp_path, monkeypatch):
    engine = GraphEngine(tmp_path / "graphs", tools={})
    class Tool:
        name = "test"
        def execute(self, node, ctx):
            return {}
    engine.register_tool(Tool())
    engine.register_template(GraphTemplate(template_id="opaque", nodes=[NodeSpec(id="n", tool="test")]))
    writes = []
    save = engine.store.save_run
    def capture(run, *args):
        writes.append(copy.deepcopy(run.metadata))
        return save(run, *args)
    monkeypatch.setattr(engine.store, "save_run", capture)
    metadata = {"method_contract": {"schema": "opaque", "steps": [1]}}
    run = engine.run_template("opaque", run_metadata=metadata)
    metadata["method_contract"]["steps"].append(2)
    assert writes[0]["method_contract"]["steps"] == [1]
    assert engine.status(run.run_id)["metadata"]["method_contract"]["steps"] == [1]
    with pytest.raises(ValueError, match="reserved"):
        engine.create("opaque", run_metadata={"execution": {}})
    with pytest.raises(ValueError, match="finite JSON"):
        engine.create("opaque", run_metadata={"bad": float("nan")})
    assert len(engine.list_runs()) == 1


def test_web_returns_exact_contract_over_sync_and_sse(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "WEB_WORKDIR", tmp_path / "sessions")
    monkeypatch.setattr(web, "WEB_SESSION_CTX", {})
    monkeypatch.setattr(web, "WEB_ACTIVE_RUNS", {})
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web.app), base_url="http://test") as client:
            body = {"session_id": "method-preview", "message": "做 Si 的收敛扫描", "run_config": config()}
            response = await client.post("/chat", json=body)
            assert response.status_code == 409, response.text
            assert response.json()["code"] == "method_confirmation_required"
            assert response.json()["method_contract"]["steps"][0]["effective"]["job_count"] == 12
            stream = await client.post("/chat/stream", json=body)
            assert '"method_confirmation_required"' in stream.text
            assert '"method_token": "' in stream.text
            assert not (web.WEB_WORKDIR / "method-preview" / "workflows").exists()
    asyncio.run(scenario())


def test_web_confirmation_reuses_original_plan_once_without_history_pollution(tmp_path, monkeypatch):
    import json

    monkeypatch.setattr(web, "WEB_WORKDIR", tmp_path / "sessions")
    monkeypatch.setattr(web, "WEB_SESSION_CTX", {})
    monkeypatch.setattr(web, "WEB_ACTIVE_RUNS", {})
    monkeypatch.setattr(web, "WEB_PENDING_METHOD_PLANS", {})
    planned, calls = [], []
    steps = [step("workflow.phonon_dfpt", material="MgO")]
    planner = Planner(steps)
    original_plan = planner.plan
    def plan(*args):
        planned.append(args)
        return original_plan(*args)
    planner.plan = plan
    async def agent(message, root, **kwargs):
        return await run_session(message, root, registry=registry_for(steps, calls), llm_planner=planner, rule_planner=planner, **kwargs)
    monkeypatch.setattr(web.agent_loop, "run", agent)

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web.app), base_url="http://test") as client:
            body = {"session_id": "freeze", "message": "执行步骤", "run_config": config()}
            first = await client.post("/chat", json=body)
            assert first.status_code == 409, first.text
            token = first.json()["method_token"]
            session = (await client.get("/api/sessions/freeze")).json()
            assert len(session["messages"]) == 1
            assert "pending_plan" not in json.dumps(session)
            assert "confirmed_method_plan" not in (web.WEB_WORKDIR / "freeze" / "session.json").read_text()
            # If replanning occurred this would return a materially different plan.
            planner.steps = [step("workflow.magnetic_compare", material="Fe")]
            second = await client.post("/chat", json={**body, "method_token": token})
            assert second.status_code == 200, second.text
            assert len(planned) == 1
            assert calls[0][0] == "workflow.phonon_dfpt"
            session = (await client.get("/api/sessions/freeze")).json()
            assert [entry["role"] for entry in session["messages"]] == ["user", "agent"]
            assert "freeze" not in web.WEB_PENDING_METHOD_PLANS
            replay = await client.post("/chat", json={**body, "method_token": token})
            assert replay.status_code == 409
            assert replay.headers["X-DFT-Policy"] == "method_confirmation_expired"
            assert len(calls) == 1
    asyncio.run(scenario())


@pytest.mark.parametrize("missing_file", [False, True])
def test_failed_or_missing_builder_cannot_fall_back_to_previous_structure(tmp_path, missing_file):
    steps = [step("structure.generate", source="Si"), step("structure.generate", source="MgO"),
             step("graph.run", template_id="t0_scf", inputs={"material": "Si"})]
    ctx = {"run_config": config()}
    contract = build_method_contract(steps, ctx["run_config"], ctx, tmp_path)
    ctx["requested_method_token"] = contract["digest"]
    registry = registry_for(steps, [])
    generated = []
    def builder(args):
        index = len(generated)
        generated.append(index)
        output = tmp_path / f"structure_{index}.cif"
        if index == 0:
            output.write_text("fixture")
        elif not missing_file:
            raise ValueError("builder fixture failed")
        return {"output": str(output), "formula": "Si", "species": []}
    registry.get("structure.generate").execute_fn = builder
    graph_calls = []
    registry.get("graph.run").execute_fn = lambda args: graph_calls.append(args) or {}
    planner = Planner(steps)
    result = asyncio.run(run_session("执行步骤", tmp_path, ctx=ctx, registry=registry, llm_planner=planner, rule_planner=planner))
    assert graph_calls == []
    assert "拒绝回退" in result["reply"]


def test_self_doping_blocks_product_plan_before_preview(tmp_path):
    calls = []
    steps = [step("structure.dope", source="Si", element="Si"), step("graph.run", template_id="t0_scf", inputs={"material": "Si"})]
    result = execute(steps, tmp_path, {"run_config": config()}, calls)
    assert not calls
    assert not list(tmp_path.iterdir())
    assert "本体元素相同" in result["reply"]


def test_fifth_template_is_resolved_and_its_changes_invalidate_confirmation(tmp_path, monkeypatch):
    import json
    from dft_forge.tools import graph_tools

    templates = tmp_path / "templates"
    templates.mkdir()
    template = GraphTemplate(template_id="new_scf", inputs={"ecutwfc": 40},
                             nodes=[NodeSpec(id="scf", tool="qe", params={"ecutwfc": "${inputs.ecutwfc}"})])
    path = templates / "new_scf.json"
    path.write_text(json.dumps(template.to_dict()))
    monkeypatch.setattr(graph_tools, "DEFAULT_TEMPLATES", templates)
    steps = [step("graph.run", template_id="new_scf", inputs={"ecutwfc": 70})]
    ctx = {"run_config": config()}
    contract = build_method_contract(steps, ctx["run_config"], ctx, tmp_path)
    assert contract["steps"][0]["template_digest"]
    template.node("scf").params["changed"] = True
    path.write_text(json.dumps(template.to_dict()))
    with pytest.raises(MethodContractError) as denied:
        check_method_contract(steps, {**ctx, "requested_method_token": contract["digest"]}, tmp_path)
    assert denied.value.code == "method_confirmation_required"


@pytest.mark.parametrize("template_id", ["../outside", "/tmp/unknown", "missing_template"])
def test_unresolvable_web_template_is_blocked_before_any_tool(tmp_path, template_id):
    calls = []
    with pytest.raises(MethodContractError) as denied:
        execute([step("structure.generate", source="Si"), step("graph.run", template_id=template_id)], tmp_path, {"run_config": config()}, calls)
    assert denied.value.code == "method_invalid"
    assert not calls
    assert not list(tmp_path.iterdir())
