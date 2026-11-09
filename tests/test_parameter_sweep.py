"""Bounded convergence scans use local synthetic tools, never a QE service."""

import copy
import hashlib
import threading
from types import SimpleNamespace

import pytest

from dft_forge.agent_loop.parameter_sweep import run_parameter_sweep
from dft_forge.compiler import DEFAULT_PSEUDO_DIR
from dft_forge.compiler.qe_parameters import convergence_proposal, resolve_qe_parameters
from dft_forge.parser import RY_TO_EV
from dft_forge.runtime.engine import GraphEngine
from dft_forge.runtime.run import ToolError
from dft_forge.tools import graph_tools
from dft_forge.web.run_views import inspect_run, list_runs


def method_for_si():
    recommendation = resolve_qe_parameters(material="Si")
    plan = convergence_proposal(recommendation)
    plan["fixed_ecutrho"] = max(plan["ecutwfc_values"]) * plan["ecutrho_ratio"]
    plan["selection_policy"] = "interior_plateau_v1"
    return {
        "mode": "demo", "digest": "confirmed-policy-fixture",
        "step": {"structure_input": {"material": "Si"},
                 "recommendation": recommendation, "convergence_plan": plan,
                 "effective": copy.deepcopy(recommendation["parameters"]),
                 "applied": copy.deepcopy(recommendation["parameters"])},
    }


class PointTool:
    name = "qe"

    def __init__(self):
        self.calls = []
        self.deltas = [[0.3, 0.2, 0.1], [0.2, 0.1, 0.05], [0.1, 0.05, 0.0]]
        self.bad = None
        self.cancel_event = None
        self.mutate = None
        self.compiler = SimpleNamespace(pseudo_dir=DEFAULT_PSEUDO_DIR)

    def execute(self, node, ctx):
        self.calls.append(copy.deepcopy(node.params))
        if self.cancel_event:
            self.cancel_event.set()
        if self.mutate:
            self.mutate()
        if self.bad == "failed" and node.node_id == "scan_c1_k1":
            raise ToolError("fixture failure", repairable=True)
        ci, ki = int(node.node_id[6]), int(node.node_id[9])
        output = {"energy_ry": -10 + self.deltas[ci][ki] * 2 / (RY_TO_EV * 1000), "natoms": 2}
        if node.node_id == "scan_c1_k1":
            if self.bad == "nan":
                output["energy_ry"] = float("nan")
            if self.bad == "missing_energy":
                output.pop("energy_ry")
            if self.bad == "natoms":
                output["natoms"] = 3
        return output


@pytest.fixture
def scan_env(tmp_path, monkeypatch):
    tool = PointTool()
    engines = []

    def get_engine(workdir):
        engine = GraphEngine(workdir, tools={"qe": tool}, max_concurrency=1,
                             execution_identity="fixture-fake",
                             execution_profile={"backend": "fake", "simulated": True})
        engines.append(engine)
        return engine

    monkeypatch.setattr(graph_tools, "_get_engine", get_engine)
    return tool, engines, tmp_path / "graphs"


def test_plateau_preserves_policy_locks_all_conditions_and_persists_run(scan_env):
    tool, engines, workdir = scan_env
    method = method_for_si()
    inputs = {"material": "Si", **method["step"]["recommendation"]["parameters"]}
    before_method, before_inputs = copy.deepcopy(method), copy.deepcopy(inputs)
    events = []
    selected, confirmed = run_parameter_sweep(inputs, method, workdir, on_event=events.append)
    evidence = confirmed["step"]["convergence_evidence"]
    assert selected["ecutwfc"] == 25
    assert selected["kpoints"] == [4, 4, 4, 1, 1, 1]
    assert selected["ecutrho"] == 300
    assert evidence["validation"] == "simulated_bounded_energy_plateau"
    assert evidence["reference"]["ecutwfc"] == 37.5
    assert len(evidence["points"]) == len(tool.calls) == 9
    assert method == before_method and inputs == before_inputs
    assert confirmed["step"]["recommendation"] == method["step"]["recommendation"]
    assert confirmed["step"]["convergence_plan"] == method["step"]["convergence_plan"]
    assert {call["ecutrho"] for call in tool.calls} == {300}
    assert {call["nbnd"] for call in tool.calls} == {12}
    assert {call["occupations"] for call in tool.calls} == {"smearing"}
    assert {call["smearing"] for call in tool.calls} == {"mp"}
    assert {call["degauss"] for call in tool.calls} == {0.005}
    assert all(node.repair == "" and node.max_repair_attempts == 0 for node in engines[0].get_template("parameter_sweep").nodes)
    assert events and all(event["node"].startswith("scan_") for event in events)
    assert all(event["phase"] == "convergence" for event in events)
    runs = list_runs(workdir.parent)
    assert len(runs) == 1 and runs[0]["run_id"] == evidence["run_id"]
    stored = inspect_run(workdir.parent, evidence["run_id"])
    assert stored["metadata"]["convergence_evidence"] == evidence
    assert stored["metadata"]["method_contract"] == confirmed


def test_whole_upper_right_rectangle_must_form_plateau(scan_env):
    tool, _, workdir = scan_env
    # The cheapest point accidentally matches the reference, but neither
    # adjacent higher grid nor cutoff does. Only the central tail is stable.
    tool.deltas = [[0, 5, 5], [5, 0.5, 0.3], [5, 0.1, 0]]
    selected, method = run_parameter_sweep({"material": "Si"}, method_for_si(), workdir)
    assert selected["ecutwfc"] == 31.25
    assert selected["kpoints"][:3] == [6, 6, 6]
    assert method["step"]["convergence_evidence"]["selected"]["cutoff_index"] == 1


def test_highest_reference_cannot_certify_itself(scan_env):
    tool, _, workdir = scan_env
    tool.deltas = [[9, 8, 7], [6, 5, 4], [3, 2, 0]]
    with pytest.raises(ValueError, match="未形成内部总能平台"):
        run_parameter_sweep({"material": "Si"}, method_for_si(), workdir)
    stored = list_runs(workdir.parent)[0]["metadata"]["convergence_evidence"]
    assert stored["validation"] == "not_validated"
    assert stored["selected"] is None


@pytest.mark.parametrize("bad", ["failed", "nan", "missing_energy", "natoms"])
def test_any_failed_or_unusable_point_blocks_the_target(scan_env, bad):
    tool, _, workdir = scan_env
    tool.bad = bad
    with pytest.raises(ValueError, match="失败或未完成"):
        run_parameter_sweep({"material": "Si"}, method_for_si(), workdir)
    stored = list_runs(workdir.parent)[0]["metadata"]["convergence_evidence"]
    assert stored["selected"] is None
    assert any(point["state"] == "failed" for point in stored["points"])


def test_cancellation_does_not_select_a_partial_grid(scan_env):
    tool, _, workdir = scan_env
    event = threading.Event()
    tool.cancel_event = event
    with pytest.raises(ValueError, match="已取消"):
        run_parameter_sweep({"material": "Si"}, method_for_si(), workdir, cancel_event=event)
    assert len(tool.calls) < 9
    assert list_runs(workdir.parent)[0]["metadata"]["convergence_evidence"]["selected"] is None


def test_pre_cancelled_scan_never_creates_a_run(scan_env):
    tool, engines, workdir = scan_env
    event = threading.Event()
    event.set()
    with pytest.raises(ValueError, match="已取消"):
        run_parameter_sweep({"material": "Si"}, method_for_si(), workdir, cancel_event=event)
    assert not engines and not tool.calls


@pytest.mark.parametrize("change", ["jobs", "rho", "kgrid", "cutoffs", "smearing"])
def test_invalid_scan_policy_is_rejected_before_execution(scan_env, change):
    tool, _, workdir = scan_env
    method = method_for_si()
    plan = method["step"]["convergence_plan"]
    if change == "jobs":
        plan["max_jobs"] = 10
    elif change == "rho":
        plan["fixed_ecutrho"] = 80
    elif change == "kgrid":
        plan["kpoints_values"][1] = [3, 3, 3, 1, 1, 1]
    elif change == "cutoffs":
        plan["ecutwfc_values"] = [25, 20, 37.5]
    else:
        plan["smearing_values"] = ["mp", "mv"]
    with pytest.raises(ValueError):
        run_parameter_sweep({"material": "Si"}, method, workdir)
    assert not tool.calls and not list_runs(workdir.parent)


def test_research_rejects_fake_execution(scan_env):
    tool, _, workdir = scan_env
    method = method_for_si()
    method["mode"] = "research"
    with pytest.raises(ValueError, match="科研模式不能"):
        run_parameter_sweep({"material": "Si"}, method, workdir)
    assert not tool.calls


def test_changed_pseudo_rejects_the_confirmed_policy(scan_env):
    tool, _, workdir = scan_env
    method = method_for_si()
    method["step"]["recommendation"]["pseudopotentials"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="文件已变化"):
        run_parameter_sweep({"material": "Si"}, method, workdir)
    assert not tool.calls


def test_pseudo_changed_during_a_point_is_not_accepted(scan_env, tmp_path):
    tool, _, workdir = scan_env
    pseudo_dir = tmp_path / "pseudos"
    pseudo_dir.mkdir()
    path = pseudo_dir / "Si_r.upf"
    path.write_bytes((DEFAULT_PSEUDO_DIR / path.name).read_bytes())
    tool.compiler.pseudo_dir = pseudo_dir
    tool.mutate = lambda: path.write_text("changed after the point began")
    with pytest.raises(ValueError, match="失败或未完成"):
        run_parameter_sweep({"material": "Si"}, method_for_si(), workdir)
    assert list_runs(workdir.parent)[0]["metadata"]["convergence_evidence"]["selected"] is None


def test_default_fake_executor_cannot_certify_missing_scf_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv("DFT_FORGE_EXECUTOR", "fake")
    with pytest.raises(ValueError, match="失败或未完成"):
        run_parameter_sweep({"material": "Si"}, method_for_si(), tmp_path / "graphs")
    evidence = list_runs(tmp_path)[0]["metadata"]["convergence_evidence"]
    assert evidence["validation"] == "not_validated" and evidence["simulated"] is True
    assert evidence["selected"] is None


def test_nonperiodic_structure_can_scan_cutoff_only(scan_env, tmp_path):
    from ase import Atoms
    from ase.io import write

    tool, _, workdir = scan_env
    structure = tmp_path / "hydrogen.xyz"
    write(structure, Atoms("H2", positions=[[9.63, 10, 10], [10.37, 10, 10]], cell=[20, 20, 20], pbc=False))
    recommendation = resolve_qe_parameters(structure=str(structure), task="t0_scf")
    assert recommendation["structure"]["pbc"] == [False, False, False]
    plan = convergence_proposal(recommendation)
    plan["fixed_ecutrho"] = max(plan["ecutwfc_values"]) * plan["ecutrho_ratio"]
    plan["selection_policy"] = "interior_plateau_v1"
    method = {"mode": "demo", "step": {
        "structure_input": {"sha256": hashlib.sha256(structure.read_bytes()).hexdigest()},
        "recommendation": recommendation, "convergence_plan": plan,
    }}
    _, result = run_parameter_sweep({"material": "H", "structure": str(structure)}, method, workdir)
    assert len(tool.calls) == 3
    assert result["step"]["convergence_evidence"]["selected"]["cutoff_index"] < 2


def test_scan_uses_confirmed_band_count(scan_env):
    tool, _, workdir = scan_env
    with pytest.raises(ValueError, match="能带总数与已确认方法不一致"):
        run_parameter_sweep({"material": "Si", "nbnd": 14}, method_for_si(), workdir)
    assert not tool.calls


def test_no_scan_policy_is_a_nonmutating_noop():
    inputs, method = {"material": "Si"}, {"step": {}}
    result_inputs, result_method = run_parameter_sweep(inputs, method, "unused")
    assert result_inputs == inputs and result_inputs is not inputs
    assert result_method == method and result_method is not method
