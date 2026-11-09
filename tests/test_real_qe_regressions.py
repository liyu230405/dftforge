"""Bugs exposed by local QE, using captured semantics without running QE."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.io import read
from ase.io import write

from dft_forge.compiler import MATERIAL_DB, build_atoms
from dft_forge.engines.qe import QECalcTool
from dft_forge.runtime.run import ExecutionContext, NodeRun
from dft_forge.agent_loop.result_views import _graph_facts
from dft_forge.agent_loop.narration import narrate


def test_relaxed_cif_preserves_parent_pseudopotentials_and_functional(monkeypatch, tmp_path):
    tool = QECalcTool()
    atoms = build_atoms("MgO")
    atoms.set_cell(atoms.cell * .98, scale_atoms=True)
    structure = tmp_path / "relaxed.cif"
    write(structure, atoms)
    parent = {key: MATERIAL_DB["MgO"][key] for key in (
        "species", "pseudos", "masses", "input_dft", "occupations", "is_metal",
    )}
    def compile_only(material, prefix, params, workdir, atoms, **kwargs):
        tool.compiler.compile_scf(material, workdir / "test.in", atoms=atoms)
        return {"energy_ry": -32.6}
    monkeypatch.setattr(tool, "_run_scf", compile_only)
    output = tool.execute(NodeRun("scf", params={"calc":"scf", "material":"MgO", "structure":str(structure)}),
        ExecutionContext(tmp_path, "test", upstream={"relax":{"final_structure_cif":str(structure), "method_profile":parent}}))
    text = (tmp_path/"scf/test.in").read_text()
    assert "Mg.pz-n-vbc.UPF" in text and "O-q6.gth" in text
    assert "Mg.upf" not in text and "O.upf" not in text
    assert 'input_dft = "pz"' in text
    assert output["method_profile"]["pseudos"] == parent["pseudos"]
    assert output["method_profile"]["input_dft"] == parent["input_dft"]


def test_missing_nscf_energy_is_not_reported_as_zero(monkeypatch, tmp_path):
    tool = QECalcTool()
    monkeypatch.setattr(tool.compiler, "compile_scf", lambda *a, **kw: "")
    monkeypatch.setattr(tool, "_run_pw", lambda *a, **kw: SimpleNamespace(stdout="End of band structure calculation\nJOB DONE."))
    monkeypatch.setattr(tool.verifier, "verify_nscf", lambda *a: SimpleNamespace(passed=True))
    assert tool._run_nscf("Si", "si", {}, tmp_path)["energy_ry"] is None
    facts = _graph_facts({"outputs":{"scf_energy_ry":-16.9}, "nodes":{"nscf":{"outputs":{"energy_ry":0.0}}}})
    assert "=0.0" not in facts and facts.count("总能量") == 1


def test_narration_receives_nested_phonon_and_vacancy_evidence():
    class Provider:
        def chat(self, system, user):
            self.user = user
            return "ok"
    p = Provider()
    data = {"stability":{"imaginary_count":0}, "formation":{"formation_energy_ev":3.2},
            "nodes":{"matdyn":{"outputs":{"frequencies_thz":[0,0,0,12,12,12]}}}}
    narrate(p, "结果", [{"command":"workflow.phonon_dfpt"}], [{"json":data}], {})
    assert 'imaginary_count' in p.user and 'formation_energy_ev' in p.user and 'frequencies_thz' in p.user


def test_fixed_cell_relax_exports_final_positions_without_cell_card(monkeypatch, tmp_path):
    import numpy as np
    tool = QECalcTool()
    atoms = build_atoms("Si").repeat((2, 2, 2))
    del atoms[0]
    final = atoms.get_scaled_positions() + .012
    stdout = "convergence has been achieved in 4 iterations\n! total energy = -143.7 Ry\n"
    stdout += "atom 1 type 1 force = 0.00001 0.0 0.0\n"
    stdout += "bfgs converged in 4 scf cycles\nBegin final coordinates\nATOMIC_POSITIONS (crystal)\n"
    stdout += "".join(f"Si {x:.10f} {y:.10f} {z:.10f}\n" for x,y,z in final)
    stdout += "End final coordinates\nJOB DONE.\n"
    monkeypatch.setattr(tool, "_run_pw", lambda *a, **kw: SimpleNamespace(stdout=stdout))
    output = tool._run_vc_relax("Si", "vacancy", {"cell_optimization":False}, tmp_path, atoms=atoms)
    relaxed = read(output["final_structure_cif"])
    assert len(relaxed) == 15
    assert np.allclose(relaxed.cell.cellpar(), atoms.cell.cellpar())
    assert np.allclose(relaxed.get_scaled_positions(), final % 1)
    assert output["max_force_ry_bohr"] == 1e-5

    from dft_forge.runtime.run import ToolError
    stdout = stdout.replace("bfgs converged in 4 scf cycles", "maximum number of steps has been reached")
    with pytest.raises(ToolError, match="geometry"):
        tool._run_vc_relax("Si", "vacancy", {"cell_optimization":False}, tmp_path, atoms=atoms)


@pytest.mark.parametrize("pbc", ([True, True, False], [False, False, False]))
def test_fixed_cell_relax_extxyz_roundtrip_preserves_exact_pbc(monkeypatch, tmp_path, pbc):
    tool = QECalcTool()
    atoms = Atoms(
        "Si2",
        scaled_positions=[[0.2, 0.3, 0.45], [0.7, 0.8, 0.55]],
        cell=[5.0, 5.0, 12.0],
        pbc=pbc,
    )
    final = atoms.get_scaled_positions() + 0.01
    stdout = "convergence has been achieved in 2 iterations\n! total energy = -10.0 Ry\n"
    stdout += "atom 1 type 1 force = 0.00001 0.0 0.0\n"
    stdout += "bfgs converged in 2 scf cycles\nBegin final coordinates\nATOMIC_POSITIONS (crystal)\n"
    stdout += "".join(f"Si {x:.10f} {y:.10f} {z:.10f}\n" for x, y, z in final)
    stdout += "End final coordinates\nJOB DONE.\n"
    monkeypatch.setattr(tool, "_run_pw", lambda *a, **kw: SimpleNamespace(stdout=stdout, success=True))

    output = tool._run_vc_relax("Si", "fixed", {"cell_optimization": False}, tmp_path, atoms=atoms)

    assert Path(output["final_structure_cif"]).is_file()  # compatibility artifact
    relaxed = read(output["final_structure_extxyz"], format="extxyz")
    assert relaxed.pbc.tolist() == pbc
    assert np.allclose(relaxed.cell, atoms.cell)
    assert np.allclose(relaxed.get_scaled_positions(), final)


def test_qe_tool_prefers_upstream_extxyz_to_cif_for_exact_pbc(monkeypatch, tmp_path):
    tool = QECalcTool()
    atoms = Atoms(
        "Si4",
        scaled_positions=[[0, 0, 0], [.25, .25, .25], [.5, .5, .5], [.75, .75, .75]],
        cell=[10.0, 10.0, 10.0],
        pbc=False,
    )
    cif = tmp_path / "relaxed.cif"
    extxyz = tmp_path / "relaxed.extxyz"
    write(cif, atoms, format="cif")
    write(extxyz, atoms, format="extxyz")
    captured = {}

    def capture_scf(material, prefix, params, workdir, input_atoms, **kwargs):
        captured["pbc"] = input_atoms.pbc.tolist()
        return {"energy_ry": -1.0}

    monkeypatch.setattr(tool, "_run_scf", capture_scf)
    parent = {key: MATERIAL_DB["Si"][key] for key in (
        "species", "pseudos", "masses", "occupations", "smearing", "degauss",
        "is_metal", "ecutwfc_default", "ecutrho_default", "kpoints_default",
    )}
    tool.execute(
        NodeRun("scf", params={"calc": "scf", "material": "Si", "structure": str(cif)}),
        ExecutionContext(tmp_path / "run", "test", upstream={"relax": {
            "final_structure_cif": str(cif),
            "final_structure_extxyz": str(extxyz),
            "method_profile": parent,
        }}),
    )

    assert captured["pbc"] == [False, False, False]


def test_status_listing_does_not_erase_current_run():
    from dft_forge.agent_loop.result_views import _handle_graph_status
    state = SimpleNamespace(ctx={"last_run_id":"run_a"})
    reply=[]
    _handle_graph_status({"runs":[{"run_id":"run_a","status":"succeeded"}]}, {}, "status", state, {}, reply)
    assert state.ctx["last_run_id"] == "run_a"
    assert "run_a: succeeded" in reply[0] and "None" not in reply[0]


def test_pdos_interpolates_only_a_known_in_range_fermi_level(monkeypatch, tmp_path):
    import numpy as np
    from dft_forge.parser import ParsedPDOS, QEParser
    tool = QECalcTool()
    monkeypatch.setattr(tool.compiler, "compile_pdos_input", lambda *a, **kw: None)
    monkeypatch.setattr(tool, "_run_projwfc", lambda *a, **kw: SimpleNamespace())
    (tmp_path/"cu.pdos_atm#1(Cu)").write_text("fixture")
    parsed = ParsedPDOS(fermi_energy_ev=0.0, n_energy_points=2,
        energies=np.array([-1., 1.]), element_dos={"Cu":np.array([.1, .3])})
    monkeypatch.setattr(QEParser, "parse_pdos", lambda *a: parsed)
    out = tool._run_pdos("Cu", "cu", {}, tmp_path)
    assert out["fermi_ev"] == 0.0
    assert np.isclose(out["element_dos_at_fermi"]["Cu"], .2)
    for missing_or_outside in (None, 3.):
        parsed.fermi_energy_ev = missing_or_outside
        assert "element_dos_at_fermi" not in tool._run_pdos("Cu", "cu", {}, tmp_path)
    assert QEParser._xml_fermi_ev(tmp_path/"missing.xml") is None


def test_narrator_keeps_dos_projections_and_confirmed_settings():
    class Provider:
        def chat(self, system, user):
            self.user = user
            return "ok"
    p = Provider()
    data={"metadata":{"method_contract":{"step":{"effective":{"ecutwfc":30}}}},
          "outputs":{"dos_at_fermi":.27,"pdos_elements":["Cu"]},
          "nodes":{"pdos":{"outputs":{"elements":["Cu"],"element_dos_at_fermi":{"Cu":.24}}}}}
    narrate(p,"结果",[{"command":"graph.run"}],[{"json":data}],{})
    assert all(key in p.user for key in ("confirmed_method","ecutwfc","Cu","element_dos_at_fermi"))
    assert "0.27" in _graph_facts(data) and "Cu" in _graph_facts(data)


def test_template_capability_preserves_flat_and_nested_inputs():
    from dft_forge.agent_loop.planning import PlanIR, compile_plan
    from dft_forge.tools.registry import registry
    from dft_forge.tools.definitions import register_default_tools
    if not registry.get("graph.run"):
        register_default_tools()
    templates = Path(__file__).resolve().parents[1]/"templates"
    for args in ({"material":"Si","ecutwfc":30}, {"inputs":{"material":"Si","ecutwfc":30}}):
        plan=PlanIR(goal="relax",actions=[{"capability":"run.t1_vc_relax","args":args}])
        step=compile_plan(plan,registry,templates_dir=templates)[0]
        assert step["args"] == {"template_id":"t1_vc_relax","inputs":{"material":"Si","ecutwfc":30}}
        assert plan.support_status == "complete"
    for args in ({"material":"Si","inputs":{"material":"Cu"}}, {"template_id":"t2_bands"}):
        plan=PlanIR(goal="relax",actions=[{"capability":"run.t1_vc_relax","args":args}])
        compile_plan(plan,registry,templates_dir=templates)
        assert plan.validation and plan.support_status == "partial"
