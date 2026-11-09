from pathlib import Path
import pytest
from dft_forge.agent_loop.step_prep import prepare_step_args
from dft_forge.agent_loop.step_validator import semantic_plan_errors


REFS={"adsorbed":"a","surface":"s","molecule":"m"}

def entry(step,energy,natoms,index):
    return {"step_id":step,"energy_ry":energy,"natoms":natoms,"graph_state":"succeeded","step_index":index,"provenance":{"execution_class":"compute"}}

def prepare(args, energies):
    return prepare_step_args("thermo.eads",args,session_dir=Path("/tmp"),ctx={},built_structure=None,scf_energies=energies,node_log=[],emit=lambda e:None,idx=5)


def test_named_binding_ignores_unrelated_runs_and_scales_o2():
    args={"energy_steps":REFS}
    assert prepare(args,[entry("a",-101,28,0),entry("s",-100,27,1),entry("m",-1,2,2),entry("other",-999,100,3)]) is None
    assert [args[k] for k in ("ry_a","ry_s","ry_m")] == [-101,-100,-1]
    assert args["nat_a"]==1 and args["nat_mol"]==2


@pytest.mark.parametrize("kind", ["missing","ambiguous","failed","future","no_atoms"])
def test_named_binding_fails_closed(kind):
    energies=[entry("a",-101,28,0),entry("s",-100,27,1),entry("m",-1,2,2)]
    if kind=="missing": energies.pop()
    if kind=="ambiguous": energies.append(entry("a",-102,28,0))
    if kind=="failed": energies[0]["graph_state"]="failed"
    if kind=="future": energies[0]["step_index"]=7
    if kind=="no_atoms": energies[0]["natoms"]=None
    assert prepare({"energy_steps":REFS},energies)


def test_invalid_named_reference_is_blocked_during_planning():
    steps=[{"step_id":sid,"tool":"graph.run","args":{}} for sid in ("a","s","m")]
    steps.append({"tool":"thermo.eads","args":{"energy_steps":REFS}})
    assert not semantic_plan_errors("吸附能",steps)
    steps[0]["tool"]="structure.build2d"
    assert semantic_plan_errors("吸附能",steps)
    steps[0]["tool"]="graph.run"
    steps.reverse()
    assert semantic_plan_errors("吸附能",steps)
