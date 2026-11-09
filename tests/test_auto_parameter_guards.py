"""Integration guards around selection, source identity and plan confirmation."""

import asyncio
import copy
import hashlib
import json
from pathlib import Path
import threading

import httpx
import pytest

import dft_forge.web as web
from dft_forge.agent_loop import executor as agent_executor
from dft_forge.agent_loop.llm_planner import LLMPlanner
from dft_forge.agent_loop.method_contract import MethodContractError, build_method_contract, method_for_step
from dft_forge.compiler import DEFAULT_PSEUDO_DIR
from dft_forge.compiler.pseudopotentials import read_pseudo_metadata
from dft_forge.engines.qe import QECalcTool
from dft_forge.executor import FakeExecutor
from dft_forge.parser import RY_TO_EV
from dft_forge.runtime.engine import GraphEngine
from dft_forge.runtime.run import ExecutionContext, NodeRun, ToolError
from dft_forge.tools import graph_tools


def _config(**values):
    return {"mode": "demo", "accuracy": "balanced", "auto_parameters": True, **values}


def _steps():
    return [{"tool": "graph.run", "args": {"template_id": "t0_scf", "inputs": {"material": "Si"}}}]


class CompilingFixtureQE(QECalcTool):
    """Compile real QE inputs and replace only execution with synthetic data."""

    def __init__(self):
        super().__init__(executor=FakeExecutor())
        self.compiled = []
        self.fail_scan = False
        self.cancel_scan = None
        self.plateau = True

    def _run_scf(self, material, prefix, params, workdir, atoms=None, timeout=None):
        output = workdir / f"{prefix}.in"
        self.compiler.compile_scf(
            material, output, atoms=atoms, prefix=prefix,
            **{key: params[key] for key in ("ecutwfc", "ecutrho", "kpoints", "nbnd", "occupations", "smearing", "degauss")
               if params.get(key) is not None},
        )
        self.compiled.append((prefix, output, copy.deepcopy(params)))
        delta = 0.0
        if prefix.startswith("scan_"):
            if self.fail_scan:
                raise ToolError("controlled SCF failure", repairable=True)
            if self.cancel_scan is not None:
                self.cancel_scan.set()
            ci, ki = int(prefix[6]), int(prefix[9])
            grid = [[5, 5, 5], [5, 0.4, 0.2], [5, 0.1, 0]] if self.plateau else [[9, 8, 7], [6, 5, 4], [3, 2, 0]]
            delta = grid[ci][ki]
        stdout = output.with_suffix(".out")
        stdout.write_text("synthetic integration fixture; no QE process was launched")
        return {"energy_ry": -10 + delta * 2 / (RY_TO_EV * 1000),
                "natoms": len(atoms) if atoms is not None else 2, "stdout_file": str(stdout)}


def _engines(monkeypatch):
    tool, engines = CompilingFixtureQE(), {}

    def get_engine(workdir, **kwargs):
        key = str(workdir)
        if key not in engines:
            engines[key] = GraphEngine(
                Path(workdir), tools={"qe": tool}, templates_dir=graph_tools.DEFAULT_TEMPLATES,
                max_concurrency=1, execution_identity="guard-fixture-fake",
                execution_profile={"backend": "fake", "simulated": True},
            )
        return engines[key]

    monkeypatch.setattr(graph_tools, "_get_engine", get_engine)
    return tool, engines


def _scan_request(tmp_path):
    contract = build_method_contract(_steps(), _config(converge_first=True), session_dir=tmp_path)
    contract["confirmation"] = "effective_contract"
    method = method_for_step(contract, 0)
    return {"template_id": "t0_scf", "workdir": str(tmp_path / "graphs"),
            "inputs": {"material": "Si", **method["step"]["applied"]}, "_method_contract": method}


def test_graph_sweep_selection_reaches_compiled_target_and_persisted_method(tmp_path, monkeypatch):
    monkeypatch.setenv("DFT_FORGE_EXECUTOR", "fake")
    tool, engines = _engines(monkeypatch)
    result = graph_tools._graph_run(_scan_request(tmp_path))
    assert result.get("state") == "succeeded", result
    assert len(tool.compiled) == 10
    assert all(prefix.startswith("scan_") for prefix, _, _ in tool.compiled[:9])
    assert tool.compiled[-1][0] == "si"
    method = result["metadata"]["method_contract"]["step"]
    evidence = method["convergence_evidence"]
    assert evidence["selected"]["ecutwfc"] == 31.25
    assert evidence["selected"]["kpoints"] == [6, 6, 6, 1, 1, 1]
    assert evidence["validation"] == "simulated_bounded_energy_plateau"
    target_text = tool.compiled[-1][1].read_text()
    assert "ecutwfc = 31.25" in target_text and "ecutrho = 300" in target_text
    assert "nbnd = 12" in target_text and "6 6 6  1 1 1" in target_text
    assert method["effective"]["ecutwfc"] == method["applied"]["ecutwfc"] == 31.25
    target = engines[str(tmp_path / "graphs")]
    stored = target.store.load_run(result["run_id"], target.get_template("t0_scf"))
    assert stored.nodes["scf"].params["ecutwfc"] == 31.25
    assert stored.nodes["scf"].params["ecutrho"] == 300
    assert stored.metadata["method_contract"]["step"]["convergence_evidence"] == evidence
    assert not tool.executor.submitted


@pytest.mark.parametrize("outcome", ["failed", "cancelled"])
def test_failed_or_cancelled_sweep_never_creates_target_run(tmp_path, monkeypatch, outcome):
    monkeypatch.setenv("DFT_FORGE_EXECUTOR", "fake")
    tool, engines = _engines(monkeypatch)
    request = _scan_request(tmp_path)
    if outcome == "failed":
        tool.fail_scan = True
    else:
        tool.cancel_scan = threading.Event()
        request["_cancel_event"] = tool.cancel_scan
    result = graph_tools._graph_run(request)
    assert "error" in result
    assert engines[str(tmp_path / "graphs")].list_runs() == []
    assert all(prefix.startswith("scan_") for prefix, _, _ in tool.compiled)


@pytest.mark.parametrize("changed", ["pseudo", "structure"])
def test_changed_auto_sources_stop_before_qe_execution(tmp_path, monkeypatch, changed):
    pseudo = tmp_path / "Si.upf"
    pseudo.write_text('<UPF><PP_HEADER z_valence="4" pseudo_type="NC"/></UPF>')
    structure = tmp_path / "uploaded.cif"
    structure.write_text("initial structure bytes")
    lock = {"pseudopotentials": [read_pseudo_metadata(pseudo)],
            "structure_sha256": hashlib.sha256(structure.read_bytes()).hexdigest()}
    tool = QECalcTool(executor=FakeExecutor(), pseudo_dir=tmp_path)
    entered = []
    monkeypatch.setattr(tool, "_execute", lambda *args: entered.append(True))
    if changed == "pseudo":
        pseudo.write_text('<UPF><PP_HEADER z_valence="4" pseudo_type="NC"/>changed</UPF>')
    else:
        structure.write_text("changed structure bytes")
    node = NodeRun(node_id="scf", params={"material": "Si", "structure": str(structure), "_source_lock": lock})
    with pytest.raises(ToolError, match="已变化") as denied:
        tool.execute(node, ExecutionContext(tmp_path / "run", "fixture"))
    assert denied.value.category == "method_source" and denied.value.repairable is False
    assert entered == [] and not tool.executor.submitted


@pytest.fixture
def guarded_web(tmp_path, monkeypatch):
    monkeypatch.setenv("DFT_FORGE_EXECUTOR", "fake")
    monkeypatch.setenv("DFT_FORGE_LLM_PROVIDER", "dummy")
    monkeypatch.setattr(web, "WEB_WORKDIR", tmp_path / "sessions")
    for name in ("WEB_SESSION_CTX", "WEB_ACTIVE_RUNS", "WEB_PENDING_METHOD_PLANS"):
        monkeypatch.setattr(web, name, {})
    tool, engines = _engines(monkeypatch)

    class Provider:
        def __init__(self):
            self.calls = []

        def chat(self, system, user):
            self.calls.append((system, user))
            return json.dumps({"goal": "计算 Si 总能", "execution_mode": "execute",
                               "support_status": "complete", "steps": _steps()})

    provider = Provider()
    planner = LLMPlanner(web.agent_loop.registry)
    planner._provider, planner._checked = provider, True
    monkeypatch.setattr(web.agent_loop, "llm_planner", planner)
    # Isolate planning calls from the optional, separate narration phase.
    monkeypatch.setattr(agent_executor, "narrate", lambda *args: "")
    return provider, tool, engines


def test_llm_context_confirmation_reuse_and_bad_request_preserve_pending(guarded_web, tmp_path):
    provider, tool, _ = guarded_web
    web._load_session("guarded")["ctx"]["last_input_file"] = str(tmp_path / "private-session-input.in")

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web.app), base_url="http://test") as client:
            body = {"message": "算 Si 的总能", "session_id": "guarded", "run_config": _config()}
            preview = await client.post("/chat", json=body)
            assert preview.status_code == 409, preview.text
            token = preview.json()["method_token"]
            assert len(provider.calls) == 1 and not tool.compiled
            _, prompt = provider.calls[0]
            context = json.loads(prompt.split("Context: ", 1)[1].split("\n", 1)[0])
            assert context["method_preferences"]["auto_parameters"] is True
            assert context["method_candidates"][0]["pseudopotentials"][0]["z_valence"] == 4
            assert context["last_input_file"] == "current-session-file"
            assert str(tmp_path) not in prompt and str(DEFAULT_PSEUDO_DIR) not in prompt
            pending = copy.deepcopy(web.WEB_PENDING_METHOD_PLANS["guarded"])
            for bad_request in (
                {**body, "method_token": "0" * 64, "approved": True},
                {**body, "method_token": token, "approved": True, "run_config": _config(accuracy="precise")},
            ):
                denied = await client.post("/chat", json=bad_request)
                assert denied.status_code == 409
                assert web.WEB_PENDING_METHOD_PLANS["guarded"] == pending
            accepted = await client.post("/chat", json={**body, "method_token": token, "approved": True})
            assert accepted.status_code == 200, accepted.text
            assert len(provider.calls) == 1 and len(tool.compiled) == 1
            repeated = await client.post("/chat", json={**body, "method_token": token, "approved": True})
            assert repeated.status_code == 409
            assert len(provider.calls) == 1 and len(tool.compiled) == 1

    asyncio.run(scenario())


def test_expired_confirmation_cannot_replan_or_execute(guarded_web):
    provider, tool, _ = guarded_web

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web.app), base_url="http://test") as client:
            body = {"message": "算 Si 的总能", "session_id": "expired", "run_config": _config()}
            preview = await client.post("/chat", json=body)
            assert preview.status_code == 409, preview.text
            web.WEB_PENDING_METHOD_PLANS["expired"]["created_at"] -= 601
            denied = await client.post("/chat", json={**body, "method_token": preview.json()["method_token"], "approved": True})
            assert denied.status_code == 409
            assert denied.headers["X-DFT-Policy"] == "method_confirmation_expired"
            assert "expired" not in web.WEB_PENDING_METHOD_PLANS
            assert len(provider.calls) == 1 and not tool.compiled

    asyncio.run(scenario())


def test_auto_parameters_leave_non_qe_graphs_outside_qe_resolution(tmp_path, monkeypatch):
    import dft_forge.compiler.qe_parameters as resolver
    from dft_forge.runtime.graph import GraphTemplate, NodeSpec

    templates = tmp_path / "templates"
    templates.mkdir()
    template = GraphTemplate(template_id="local_analysis", nodes=[
        NodeSpec(id="analysis", tool="offline_analysis", params={"value": 2}),
    ])
    (templates / "local_analysis.json").write_text(json.dumps(template.to_dict()))
    monkeypatch.setattr(graph_tools, "DEFAULT_TEMPLATES", templates)

    def forbidden_resolution(**kwargs):
        pytest.fail("A non-QE graph must not resolve QE materials or parameters")

    monkeypatch.setattr(resolver, "resolve_qe_parameters", forbidden_resolution)
    contract = build_method_contract(
        [{"tool": "graph.run", "args": {"template_id": "local_analysis", "inputs": {}}}],
        _config(), session_dir=tmp_path,
    )
    assert contract["steps"][0]["template_id"] == "local_analysis"
    assert not contract["steps"][0].get("convergence_plan")


def test_planner_candidates_respect_manual_method_cutoff():
    from dft_forge.compiler.qe_parameters import planner_parameter_context

    context = planner_parameter_context("算 Si 的单点总能", {"run_config": _config(ecutwfc=65)})
    candidate = next(item for item in context["method_candidates"] if item["material"] == "Si")
    assert context["method_preferences"]["ecutwfc"] == 65
    assert candidate["parameters"]["ecutwfc"] == 65
    assert candidate["parameters"]["ecutrho"] == 520


@pytest.mark.parametrize("outcome", ["failed", "no_plateau"])
def test_unsuccessful_scan_exports_its_convergence_evidence(tmp_path, monkeypatch, outcome):
    from dft_forge.web.evidence import build_evidence
    from dft_forge.web.run_views import evidence_snapshot, list_runs

    monkeypatch.setenv("DFT_FORGE_EXECUTOR", "fake")
    tool, engines = _engines(monkeypatch)
    tool.fail_scan = outcome == "failed"
    tool.plateau = outcome != "no_plateau"
    result = graph_tools._graph_run(_scan_request(tmp_path))
    assert "error" in result
    assert not engines[str(tmp_path / "graphs")].list_runs()
    runs = list_runs(tmp_path)
    assert len(runs) == 1 and runs[0]["template_id"] == "parameter_sweep"
    snapshot = evidence_snapshot(tmp_path, runs[0]["run_id"])
    payload = build_evidence(snapshot, tmp_path, [])
    exported = json.loads(json.dumps(payload, allow_nan=False))
    evidence = exported["run"]["convergence_evidence"]
    assert evidence["run_id"] == runs[0]["run_id"]
    assert evidence["validation"] == "not_validated" and evidence["selected"] is None
    assert len(evidence["points"]) == 9 and evidence["error"]
    if outcome == "no_plateau":
        assert evidence["reference"]["node_id"] == "scan_c2_k2"
        assert all(point["state"] == "succeeded" for point in evidence["points"])
    else:
        assert any(point["state"] == "failed" for point in evidence["points"])


def test_auto_mode_directs_legacy_convergence_macro_to_confirmed_scan(tmp_path):
    with pytest.raises(MethodContractError, match="勾选.*先检查收敛") as denied:
        build_method_contract(
            [{"tool": "workflow.convergence", "args": {"material": "Si"}}],
            _config(), session_dir=tmp_path,
        )
    assert denied.value.code == "method_unsupported"
    assert not list(tmp_path.iterdir())


def test_converge_first_budget_does_not_multiply_across_two_graphs(tmp_path):
    steps = [*_steps(), {"tool": "graph.run", "args": {
        "template_id": "t0_scf", "inputs": {"material": "Al"},
    }}]
    with pytest.raises(MethodContractError, match="最多 9 点") as denied:
        build_method_contract(steps, _config(converge_first=True), session_dir=tmp_path)
    assert denied.value.code == "method_unsupported"
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("manual_cutoff", [None, 70])
def test_deferred_phonon_macro_discards_planner_numbers_before_registry_call(tmp_path, manual_cutoff):
    from dft_forge.agent_loop.step_prep import prepare_step_args
    from dft_forge.tools.models import ToolEntry
    from dft_forge.tools.registry import ToolRegistry

    guessed = {"material": "Si", "ecutwfc": 1, "ecutrho": 4, "kpoints": [1, 1, 1, 0, 0, 0]}
    steps = [{"tool": "workflow.phonon_dfpt", "args": guessed}]
    config = _config(**({"ecutwfc": manual_cutoff} if manual_cutoff is not None else {}))
    contract = build_method_contract(steps, config, session_dir=tmp_path)
    row = contract["steps"][0]
    assert row["recommendation"]["status"] == "deferred"
    assert row["effective"].get("ecutrho") is None
    assert row["effective"].get("kpoints") is None
    if manual_cutoff is None:
        assert row["effective"].get("ecutwfc") is None
        assert "ecutwfc" not in row["applied"]
    else:
        assert row["effective"]["ecutwfc"] == row["applied"]["ecutwfc"] == 70
    contract["confirmation"] = "effective_contract"
    prepared = copy.deepcopy(guessed)
    assert prepare_step_args(
        "workflow.phonon_dfpt", prepared, session_dir=tmp_path, ctx={}, built_structure=None,
        scf_energies=[], node_log=[], emit=lambda _: None, idx=0,
        method=method_for_step(contract, 0),
    ) is None
    received = []
    registry = ToolRegistry()
    registry.register(ToolEntry(
        id="workflow.phonon_dfpt", name="phonon fixture", description="capture the actual macro boundary",
        side_effect="external_execution", execute_fn=lambda args: received.append(copy.deepcopy(args)) or {},
    ))
    result = asyncio.run(registry.call("workflow.phonon_dfpt", prepared))
    assert result.error is None and len(received) == 1
    assert "ecutrho" not in received[0] and "kpoints" not in received[0]
    if manual_cutoff is None:
        assert "ecutwfc" not in received[0]
    else:
        assert received[0]["ecutwfc"] == 70
        assert received[0]["_method_contract"]["step"]["effective"]["ecutwfc"] == 70
