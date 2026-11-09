"""Task-aware preview, explicit confirmation, and actual input lineage."""

import asyncio
import copy
import json
from pathlib import Path

import httpx
import pytest

import dft_forge.web as web
from dft_forge.agent_loop.method_contract import (
    MethodContractError, build_method_contract, check_method_contract, method_for_step,
)
from dft_forge.agent_loop.step_prep import prepare_step_args
from dft_forge.compiler import QECompiler, DEFAULT_PSEUDO_DIR, build_atoms
from dft_forge.compiler.qe_parameters import resolve_qe_parameters, planner_parameter_context


def config(**kwargs):
    return {"mode": "demo", "accuracy": "balanced", "auto_parameters": True, **kwargs}


def plan(template="t2_bands", **inputs):
    return [{"tool": "graph.run", "args": {"template_id": template, "inputs": {"material": "Si", **inputs}}}]


def test_si_uses_actual_upf_and_exact_initial_values(tmp_path):
    result = resolve_qe_parameters(material="Si")
    assert result["parameters"] == {"ecutwfc": 25, "ecutrho": 200, "kpoints": [4, 4, 4, 1, 1, 1], "nbnd": 12, "nkpoints_bands": 100}
    assert result["pseudopotentials"][0]["z_valence"] == 4
    assert result["structure"]["electron_count"] == 8
    assert result["validation"] == "not_tested"
    assert len(result["resolution_digest"]) == 64
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("overrides", [{"ecutwfc": float("nan")}, {"ecutrho": 10}, {"nbnd": 2}, {"nbnd": True}, {"kpoints": [0, 1, 1, 0, 0, 0]}, {"nkpoints_bands": 2}])
def test_invalid_values_rejected(overrides):
    with pytest.raises(ValueError):
        resolve_qe_parameters(material="Si", overrides=overrides)


def test_manual_cutoff_updates_rho_and_source():
    rec = resolve_qe_parameters(material="Si", overrides={"ecutwfc": 60, "nbnd": 20})
    assert rec["parameters"]["ecutrho"] == 480
    assert rec["sources"]["ecutwfc"] == "user"
    assert rec["sources"]["ecutrho"] == "deterministic"
    assert rec["parameters"]["nbnd"] == 20


def test_2d_mesh_keeps_vacuum_axis_unshifted(tmp_path):
    from ase.io import write
    from dft_forge.structure_builder import build_monolayer

    path = tmp_path / "graphene.cif"
    write(path, build_monolayer("graphene"))
    rec = resolve_qe_parameters(structure=path)
    assert rec["structure"]["pbc"] == [True, True, False]
    assert rec["parameters"]["kpoints"][2] == 1
    assert rec["parameters"]["kpoints"][5] == 0


def test_automatic_contract_overrides_planner_guess_and_binds_all_values(tmp_path):
    contract = build_method_contract(plan(ecutwfc=1, nbnd=999), config(), session_dir=tmp_path)
    row = contract["steps"][0]
    assert contract["requires_confirmation"]
    assert row["effective"]["ecutwfc"] == 25
    assert row["effective"]["nbnd"] == 12
    assert row["effective"]["ecutrho"] == 200
    assert row["automatic_defaults"] == []
    assert not list(tmp_path.iterdir())
    contract = build_method_contract(plan(ecutwfc=1), config(ecutwfc=65), session_dir=tmp_path)
    assert contract["steps"][0]["effective"]["ecutrho"] == 520


def test_deferred_macro_strips_planner_qe_guesses_but_keeps_user_overrides(tmp_path):
    steps = [{
        "tool": "workflow.phonon_dfpt",
        "args": {"material": "MgO", "ecutwfc": 1, "ecutrho": 2, "kpoints": [1, 1, 1, 0, 0, 0]},
    }]
    contract = build_method_contract(steps, config(), session_dir=tmp_path)
    row = contract["steps"][0]
    assert row["recommendation"]["status"] == "deferred"
    assert row["strip_planner_parameters"] == ["ecutwfc", "ecutrho", "kpoints"]
    assert set(row["conflicts"]) == {"ecutwfc", "ecutrho", "kpoints"}
    assert all(item["source"] == "planner" for item in row["conflicts"].values())

    contract["confirmation"] = "effective_contract"
    args = copy.deepcopy(steps[0]["args"])
    assert prepare_step_args(
        "workflow.phonon_dfpt", args, session_dir=tmp_path, ctx={}, built_structure=None,
        scf_energies=[], node_log=[], emit=lambda _: None, idx=0,
        method=method_for_step(contract, 0), structure_outputs={},
    ) is None
    assert all(key not in args for key in ("ecutwfc", "ecutrho", "kpoints"))

    explicit = build_method_contract(steps, config(ecutwfc=65, kpoints=[4, 4, 4, 1, 1, 1]), session_dir=tmp_path)
    explicit["confirmation"] = "effective_contract"
    explicit_args = copy.deepcopy(steps[0]["args"])
    assert prepare_step_args(
        "workflow.phonon_dfpt", explicit_args, session_dir=tmp_path, ctx={}, built_structure=None,
        scf_energies=[], node_log=[], emit=lambda _: None, idx=0,
        method=method_for_step(explicit, 0), structure_outputs={},
    ) is None
    assert explicit_args["ecutwfc"] == 65
    assert explicit_args["kpoints"] == [4, 4, 4, 1, 1, 1]
    assert "ecutrho" not in explicit_args


def test_pseudo_content_change_invalidates_confirmation(tmp_path, monkeypatch):
    import dft_forge.compiler.qe_parameters as resolver

    pseudo = tmp_path / "Si_r.upf"
    pseudo.write_text('<UPF><PP_HEADER z_valence="4" pseudo_type="NC"/></UPF>')
    monkeypatch.setattr(resolver, "DEFAULT_PSEUDO_DIR", tmp_path)
    ctx = {"run_config": config()}
    first = build_method_contract(plan(), ctx["run_config"], ctx, tmp_path)
    ctx["requested_method_token"] = first["digest"]
    check_method_contract(plan(), ctx, tmp_path)
    pseudo.write_text('<UPF><PP_HEADER z_valence="4" pseudo_type="NC"/>changed</UPF>')
    with pytest.raises(MethodContractError) as exc:
        check_method_contract(plan(), ctx, tmp_path)
    assert exc.value.code == "method_confirmation_required"


def test_sweep_contract_fixes_rho_and_budget(tmp_path):
    contract = build_method_contract(plan(), config(converge_first=True), session_dir=tmp_path)
    row = contract["steps"][0]
    assert row["convergence_plan"]["job_count"] == 9
    assert row["convergence_plan"]["fixed_ecutrho"] == 300
    assert row["effective"]["ecutrho"] == 300
    with pytest.raises(MethodContractError):
        build_method_contract(plan(), config(converge_first=True, ecutrho=200), session_dir=tmp_path)
    explicit = build_method_contract(plan(), config(converge_first=True, ecutrho=400), session_dir=tmp_path)
    assert explicit["steps"][0]["convergence_plan"]["fixed_ecutrho"] == 400


def test_builder_preview_does_not_guess_si_and_resolves_actual_file(tmp_path):
    from ase.io import write

    steps = [{"tool": "structure.generate", "args": {"source": "Al"}}, *plan()]
    contract = build_method_contract(steps, config(), session_dir=tmp_path)
    assert contract["steps"][0]["recommendation"]["status"] == "deferred"
    contract["confirmation"] = "effective_contract"
    method = method_for_step(contract, 1)
    source = tmp_path / "Al.cif"
    write(source, build_atoms("Al"))
    args = copy.deepcopy(steps[1]["args"])
    assert prepare_step_args("graph.run", args, session_dir=tmp_path, ctx={}, built_structure=None,
                             scf_energies=[], node_log=[], emit=lambda _: None, idx=1,
                             method=method, structure_outputs={0: str(source)}) is None
    assert args["inputs"]["structure"] == str(source)
    assert "material" not in args["inputs"]
    assert args["_method_contract"]["step"]["recommendation"]["structure"]["formula"] == "Al"
    assert args["inputs"]["nbnd"] == resolve_qe_parameters(structure=source)["parameters"]["nbnd"]


def test_llm_receives_preferences_and_safe_parameter_evidence():
    context = planner_parameter_context("算 Si 的能带", {"run_config": config(accuracy="precise")})
    assert context["method_preferences"]["accuracy"] == "precise"
    assert context["method_candidates"][0]["pseudopotentials"][0]["z_valence"] == 4
    assert str(DEFAULT_PSEUDO_DIR) not in json.dumps(context)


@pytest.fixture
def offline_web(tmp_path, monkeypatch):
    monkeypatch.setenv("DFT_FORGE_EXECUTOR", "fake")
    monkeypatch.setenv("DFT_FORGE_LLM_PROVIDER", "dummy")
    monkeypatch.setattr(web, "WEB_WORKDIR", tmp_path / "sessions")
    for key in ("WEB_SESSION_CTX", "WEB_ACTIVE_RUNS", "WEB_PENDING_METHOD_PLANS"):
        monkeypatch.setattr(web, key, {})
    return tmp_path


def test_actual_web_preview_confirm_and_compiled_evidence(offline_web, monkeypatch):
    from dft_forge.runtime.engine import GraphEngine
    from dft_forge.tools.graph_tools import DEFAULT_TEMPLATES, _ENGINE_CACHE

    class FixtureQE:
        name = "qe"
        def execute(self, node, ctx):
            p = node.params
            if p["calc"] == "bands":
                return {"n_bands": 12, "band_gap_ev": 0.5}
            output = ctx.node_workdir(node.node_id) / f"si_{p['calc']}.in"
            QECompiler(DEFAULT_PSEUDO_DIR).compile_scf("Si", output, calculation=p["calc"],
                **{key: p[key] for key in ("ecutwfc", "ecutrho", "nbnd")}, kpoints=p.get("kpoints"))
            stdout = output.with_suffix(".out")
            stdout.write_text("offline test fixture")
            return {"energy_ry": -15, "natoms": 2, "stdout_file": str(stdout)}

    base = web.WEB_WORKDIR / "auto" / "graphs"
    engine = GraphEngine(base, tools={"qe": FixtureQE()}, templates_dir=DEFAULT_TEMPLATES,
                         execution_identity="fixture", execution_profile={"backend": "fake", "simulated": True})
    monkeypatch.setitem(_ENGINE_CACHE, str(base), engine)
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web.app), base_url="http://test") as client:
            body = {"message": "算 Si 的能带", "session_id": "auto", "run_config": config()}
            response = await client.post("/chat", json=body)
            assert response.status_code == 409, response.text
            preview = response.json()
            assert preview["code"] == "method_confirmation_required"
            assert preview["method_contract"]["steps"][0]["effective"]["nbnd"] == 12
            assert engine.list_runs() == []
            wrong = await client.post("/chat", json={**body, "method_token": "0" * 64, "approved": True})
            assert wrong.status_code == 409
            assert "auto" in web.WEB_PENDING_METHOD_PLANS
            # Token validation must preserve a valid card after a bad request.
            response = await client.post("/chat", json={**body, "method_token": preview["method_token"], "approved": True})
            assert response.status_code == 200, response.text
            runs = (await client.get("/api/sessions/auto/runs")).json()["runs"]
            assert len(runs) == 1
            evidence = (await client.get(f"/api/sessions/auto/runs/{runs[0]['run_id']}/evidence")).json()
            assert evidence["run"]["method_contract"]["step"]["effective"]["nbnd"] == 12
            for name in ("scf", "nscf"):
                node = next(n for n in evidence["nodes"] if n.get("node_id", n.get("id")) == name)
                assert node["resolved_params"]["ecutrho"] == 200
                assert node["resolved_params"]["nbnd"] == 12
            inputs = list((web.WEB_WORKDIR / "auto" / "graphs").rglob("*_scf.in"))
            assert inputs and "nbnd = 12" in inputs[0].read_text()
            assert "ecutrho = 200" in inputs[0].read_text()
            session = (await client.get("/api/sessions/auto")).json()
            assert sum(m["role"] == "user" for m in session["messages"]) == 1
            repeated = await client.post("/chat", json={**body, "method_token": preview["method_token"], "approved": True})
            assert repeated.status_code == 409
    asyncio.run(scenario())


def test_research_preview_works_on_fake_but_confirmation_cannot_compute(offline_web):
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web.app), base_url="http://test") as client:
            body = {"message": "算 Si 的能带", "session_id": "research", "run_config": config(mode="research")}
            first = await client.post("/chat", json=body)
            assert first.json()["code"] == "method_confirmation_required"
            token = first.json()["method_token"]
            denied = await client.post("/chat", json={**body, "method_token": token, "approved": True})
            assert denied.status_code == 409 and "模拟后端" in denied.json()["detail"]
            assert not (web.WEB_WORKDIR / "research" / "graphs").exists()
    asyncio.run(scenario())


def test_research_parameter_lock_keeps_failed_params_unchanged(tmp_path, monkeypatch):
    from dft_forge.runtime.engine import GraphEngine
    from dft_forge.runtime.run import ToolError
    from dft_forge.engines.qe import QEParamRepairer
    from dft_forge.tools.graph_tools import DEFAULT_TEMPLATES, _ENGINE_CACHE, _graph_run

    class Failing:
        name = "qe"
        def execute(self, node, ctx):
            raise ToolError("scf not converged", category="scf_not_converged", repairable=True)

    base = tmp_path / "graphs"
    engine = GraphEngine(base, tools={"qe": Failing()}, templates_dir=DEFAULT_TEMPLATES,
                         repairers={"qe_param_bump": QEParamRepairer()})
    monkeypatch.setitem(_ENGINE_CACHE, str(base), engine)
    contract = build_method_contract(plan("t0_scf"), config(mode="research"), session_dir=tmp_path)
    contract["confirmation"] = "effective_contract"
    method = method_for_step(contract, 0)
    result = _graph_run({"template_id": "t0_scf", "workdir": str(base),
                         "inputs": {"material": "Si", **method["step"]["applied"]}, "_method_contract": method})
    assert result["state"] == "failed"
    node = engine.store.load_run(result["run_id"], engine.get_template("t0_scf")).nodes["scf"]
    assert node.params["ecutwfc"] == 25
    assert node.attempt == 1
    assert "重新" in node.error
