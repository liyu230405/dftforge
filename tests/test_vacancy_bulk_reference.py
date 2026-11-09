import math
import pytest
import dft_forge.workflows as workflows
from dft_forge.agent_loop.planning import PlanIR
from dft_forge.agent_loop.scientific_inputs import guard_scientific_inputs


def runner(energy=-160.0, natoms=16):
    def fake(template, *args, **kw):
        return {"run_id":"fixture", "state":"succeeded", "nodes": {
            "pristine_scf":{"state":"succeeded", "outputs":{"energy_ry":energy,"natoms":natoms}},
            "defect_scf":{"state":"succeeded", "outputs":{"energy_ry":-149.7,"natoms":15}},
        }}
    return fake


def test_same_method_bulk_reference_and_formation(monkeypatch, tmp_path):
    monkeypatch.setattr(workflows, "_run_graph", runner())
    r = workflows.run_neutral_vacancy({"source":"Si","element":"Si","reference_mode":"same_method_bulk_per_atom","relax":False,"workdir":str(tmp_path)})
    assert r["chemical_potential_reference"]["chemical_potential_ry"] == -10
    assert r["chemical_potential_reference"]["natoms"] == 16
    assert r["formation"]["formation_energy_ry"] == pytest.approx(.3)


@pytest.mark.parametrize("extra", [{"source":"MgO","element":"O"},{"chemical_potential_ev":0}])
def test_invalid_reference_is_rejected_before_qe(monkeypatch, tmp_path, extra):
    monkeypatch.setattr(workflows, "_run_graph", lambda *a, **k: pytest.fail("must not run QE"))
    with pytest.raises(workflows.WorkflowError):
        workflows.run_neutral_vacancy({"source":"Si","element":"Si","reference_mode":"same_method_bulk_per_atom","workdir":str(tmp_path),**extra})


@pytest.mark.parametrize("energy,natoms", [(math.nan,16),(-160,2)])
def test_invalid_bulk_evidence_does_not_produce_formation(monkeypatch,tmp_path,energy,natoms):
    monkeypatch.setattr(workflows,"_run_graph",runner(energy,natoms))
    r=workflows.run_neutral_vacancy({"source":"Si","element":"Si","reference_mode":"same_method_bulk_per_atom","relax":False,"workdir":str(tmp_path)})
    assert r["status"] == "partial" and "formation" not in r


def test_planner_can_select_computed_reference_but_not_manual_mu():
    steps=[{"tool":"workflow.neutral_vacancy","args":{"source":"Si","element":"Si","reference_mode":"same_method_bulk_per_atom"}}]
    plan=PlanIR.from_steps("Si vacancy",steps)
    guard_scientific_inputs(plan,steps,"Si vacancy",{})
    assert len(steps)==1 and plan.support_status=="complete"
    steps[0]["args"]["chemical_potential_ev"]=0
    guard_scientific_inputs(plan,steps,"Si vacancy",{})
    assert not steps and plan.support_status=="partial"
