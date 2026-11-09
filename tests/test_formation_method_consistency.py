"""Formation energies must subtract total energies from one method."""
from pathlib import Path

from dft_forge.agent_loop.step_prep import (
    align_formation_compounds,
    formation_method_mismatch,
    prepare_step_args,
)
from dft_forge.compiler import DEFAULT_PSEUDO_DIR, QECompiler, read_structure
from dft_forge.structure_builder import molecule_scf_inputs, reference_molecule_kind


def _profile(pseudos, dft=None, ecut=40):
    return {"pseudos": pseudos, "input_dft": dft, "effective": {"ecutwfc": ecut, "ecutrho": 8 * ecut}}


def _prep(tool, args, scf, ctx=None, tmp=Path("/tmp")):
    return prepare_step_args(tool, args, session_dir=tmp, ctx=ctx or {}, built_structure=None,
                             scf_energies=scf, node_log=[], emit=lambda e: None, idx=6)


def test_mixed_pseudopotentials_are_rejected():
    scf = [
        {"energy_ry": -150.0, "natoms": 2, "method_profile": _profile({"Mg": "Mg.pz-n-vbc.UPF", "O": "O-q6.gth"}, "pz")},
        {"energy_ry": -120.0, "natoms": 2, "method_profile": _profile({"Mg": "Mg.upf"})},
        {"energy_ry": -82.0, "natoms": 2, "method_profile": _profile({"O": "O.upf"})},
    ]
    args = {"formula": "MgO"}
    error = _prep("thermo.formation", args, scf)
    assert error and "赝势不一致" in error
    assert "compound_energy" not in args


def test_consistent_method_is_injected():
    scf = [
        {"energy_ry": -150.0, "natoms": 2, "method_profile": _profile({"Mg": "Mg.upf", "O": "O.upf"})},
        {"energy_ry": -120.0, "natoms": 2, "method_profile": _profile({"Mg": "Mg.upf"})},
        {"energy_ry": -82.0, "natoms": 2, "method_profile": _profile({"O": "O.upf"})},
    ]
    args = {"formula": "MgO"}
    assert _prep("thermo.formation", args, scf) is None
    assert args["compound_energy"] == -150.0 and args["_method_consistent"]


def test_cutoff_and_afm_labels():
    afm = _profile({"FeA": "Fe.upf", "FeB": "Fe.upf", "O": "O.upf"})
    assert formation_method_mismatch([
        {"method_profile": afm}, {"method_profile": _profile({"Fe": "Fe.upf"})}]) is None
    assert "ecutwfc" in formation_method_mismatch([
        {"method_profile": _profile({"O": "O.upf"}, ecut=40)},
        {"method_profile": _profile({"O": "O.upf"}, ecut=50)}])


def test_library_compound_switches_to_reference_pseudos(tmp_path):
    steps = [
        {"tool": "graph.run", "args": {"template_id": "t0_scf", "inputs": {"material": "MgO"}}},
        {"tool": "structure.reference", "args": {"element": "Mg"}},
        {"tool": "graph.run", "args": {"template_id": "t0_scf", "inputs": {}}},
        {"tool": "thermo.formation", "args": {"formula": "MgO"}},
    ]
    notes = align_formation_compounds(steps, tmp_path)
    inputs = steps[0]["args"]["inputs"]
    assert notes and "material" not in inputs
    content = QECompiler(DEFAULT_PSEUDO_DIR).compile_scf(
        "MgO", tmp_path / "x.in", atoms=read_structure(inputs["structure"]))
    assert "Mg.upf" in content and "O.upf" in content
    assert "pz-n-vbc" not in content and "input_dft" not in content


def test_no_rewrite_without_formation(tmp_path):
    steps = [{"tool": "graph.run", "args": {"template_id": "t0_scf", "inputs": {"material": "MgO"}}}]
    assert align_formation_compounds(steps, tmp_path) == []
    assert steps[0]["args"]["inputs"] == {"material": "MgO"}


def test_o2_reference_runs_as_triplet_molecule(tmp_path):
    assert reference_molecule_kind("O") == "o2" and reference_molecule_kind("Mg") is None
    ref = tmp_path / "ref_O.cif"
    ref.write_text("x")
    ctx = {"structure_molecule_kind": {str(ref): "o2"}}
    args = {"template_id": "t0_scf", "inputs": {}}
    assert prepare_step_args("graph.run", args, session_dir=tmp_path, ctx=ctx, built_structure=str(ref),
                             scf_energies=[], node_log=[], emit=lambda e: None, idx=4) is None
    inputs = args["inputs"]
    assert inputs["kpoints"] == [1, 1, 1, 0, 0, 0]
    assert inputs["occupations"] == "fixed"
    assert inputs["spin_polarized"] and inputs["total_magnetization_muB"] == 2.0


def test_o2_scf_input_is_spin_polarized_gamma(tmp_path):
    from dft_forge.structure_builder import build_reference

    atoms = build_reference("O").atoms
    content = QECompiler(DEFAULT_PSEUDO_DIR).compile_scf(
        "O2", tmp_path / "o2.in", atoms=atoms,
        kpoints=(1, 1, 1, 0, 0, 0), occupations="fixed",
        **{k: v for k, v in molecule_scf_inputs("o2").items() if k not in ("kpoints", "occupations")})
    assert "nspin = 2" in content and "tot_magnetization = 2" in content
    assert 'occupations = "fixed"' in content and "1 1 1  0 0 0" in content
