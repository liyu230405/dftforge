"""Pure and compiler-level tests for the new scientific workflow macros."""

from pathlib import Path

import pytest

from dft_forge.compiler import QECompiler, formula_atoms
from dft_forge.parser import QEParser
from dft_forge.workflows import (
    WorkflowError,
    chemical_potential_scan,
    compare_magnetic_results,
    convergence_grid,
    phonon_stability,
    select_converged_point,
    vacancy_formation_energy,
)
import dft_forge.workflows as workflows


def test_convergence_grid_and_condition_specific_reference():
    grid = convergence_grid(ecutwfc_values=[30, 40], kpoints_values=[4, 8])
    assert len(grid) == 4
    results = [
        {**p, "state": "succeeded", "energy_ry": -10.0 - 0.00001 * i, "natoms": 2}
        for i, p in enumerate(grid)
    ]
    selected = select_converged_point(results, threshold_mev_per_atom=1.0)
    assert selected["selected"]["ecutwfc"] == 30.0
    assert len(selected["references_by_condition"]) == 1


def test_convergence_grid_has_a_hard_job_cap():
    with pytest.raises(WorkflowError, match="max_jobs"):
        convergence_grid(ecutwfc_values=[30, 40], kpoints_values=[4, 6], max_jobs=2)


def test_neutral_vacancy_formula_uses_explicit_per_atom_mu():
    result = vacancy_formation_energy(
        defect_energy_ry=-100.0,
        pristine_energy_ry=-99.0,
        chemical_potential_ry=-0.5,
    )
    assert result["formation_energy_ry"] == pytest.approx(-1.5)


def test_binary_chemical_potential_domain_and_missing_competitor_warning():
    result = chemical_potential_scan(
        formula="MgO", formation_energy_ev_per_formula=-6.0, n_points=3
    )
    assert result["interval"]["Mg"] == [-6.0, 0.0]
    assert result["points"][0]["delta_mu_O_ev"] == pytest.approx(0.0)
    assert result["status"] == "partial"


def test_chemical_potential_infeasible_competitor_is_rejected():
    with pytest.raises(WorkflowError, match="empty"):
        chemical_potential_scan(
            formula="MgO",
            formation_energy_ev_per_formula=-6.0,
            competitors=[{"formula": "MgO", "formation_energy_ev_per_formula": -7.0}],
        )


def test_magnetic_comparison_flags_state_collapse():
    result = compare_magnetic_results([
        {"requested_order": "NM", "energy_ry": -10.0, "natoms": 2},
        {"requested_order": "FM", "energy_ry": -10.1, "natoms": 2, "atom_magnetizations_muB": [0.0, 0.0]},
        {"requested_order": "AFM", "energy_ry": -10.2, "natoms": 2, "atom_magnetizations_muB": [2.0, -2.0]},
    ])
    assert result["ground_state"] == "AFM"
    assert any("FM" in warning for warning in result["warnings"])


def test_magnetic_comparison_does_not_overclaim_afm_without_site_moments():
    result = compare_magnetic_results([
        {"requested_order": "AFM", "energy_ry": -10.0, "natoms": 2,
         "total_magnetization_muB": 0.01, "absolute_magnetization_muB": 4.2},
    ])
    assert result["ground_state_status"] == "partially_verified"
    assert any("逐原子" in warning for warning in result["warnings"])


def test_phonon_stability_counts_imaginary_and_soft_modes():
    result = phonon_stability([-0.2, -0.01, 0.0, 1.2], imaginary_threshold_thz=0.1)
    assert result["imaginary_count"] == 1
    assert result["soft_count"] == 2
    assert result["dynamically_stable_at_threshold"] is False


def test_qe_compiler_emits_magnetic_and_dfpt_inputs(tmp_path):
    compiler = QECompiler(Path(__file__).resolve().parents[1] / "assets" / "pseudos")
    afm = compiler.compile_scf(
        "Fe", tmp_path / "afm.in", atoms=formula_atoms("Fe")[0],
        magnetic_order="AFM", initial_moment_muB=2.2,
    )
    assert "nspin = 2" in afm and 'verbosity = "high"' in afm
    assert "starting_magnetization(1)" in afm
    assert "FeA" in afm and "FeB" in afm
    ph = compiler.compile_ph_input("MgO", tmp_path / "ph.in", q_grid=(2, 2, 2))
    matdyn = compiler.compile_matdyn_input("MgO", tmp_path / "matdyn.in", q_grid=(2, 2, 2))
    assert "ldisp = .true." in ph and "nq3 = 2" in ph
    assert "dos = .true." in matdyn and "nk1 = 2" in matdyn


def test_qe_parser_reads_magnetization_and_phonons():
    stdout = """
     atom:    1    charge:  7.0    magnetization:  2.10
     atom:    2    charge:  7.0    magnetization: -2.10
     total magnetization       =     0.0000 Bohr mag/cell
     absolute magnetization    =     4.2000 Bohr mag/cell
     freq (    1) =   -0.2000 [THz] = -6.6713 [cm-1]
     freq (    2) =    1.5000 [THz] = 50.0346 [cm-1]
    """
    mag = QEParser.parse_magnetization(stdout)
    ph = QEParser.parse_phonon(stdout)
    assert mag.total_magnetization_muB == pytest.approx(0.0)
    assert mag.atom_magnetizations_muB == pytest.approx([2.1, -2.1])
    assert ph.frequencies_thz == pytest.approx([-0.2, 1.5])


def test_chemical_potential_rejects_competitor_with_another_element():
    with pytest.raises(WorkflowError, match="outside parent"):
        chemical_potential_scan(
            formula="MgO",
            formation_energy_ev_per_formula=-6.0,
            competitors=[{"formula": "MgN", "formation_energy_ev_per_formula": -7.0}],
        )


def test_registered_workflow_macros_build_expected_dags(monkeypatch, tmp_path):
    """The macro owns the DAG shape; this test avoids requiring QE binaries."""
    captured = {}

    def fake_run_graph(template, workdir, inputs=None):
        captured[template.description] = template
        node_outputs = {}
        for node in template.nodes:
            if node.id.startswith("point_"):
                node_outputs[node.id] = {
                    "state": "succeeded", "error": None,
                    "outputs": {"energy_ry": -10.0, "natoms": 2},
                }
            elif node.id in {"nm", "fm", "afm"}:
                node_outputs[node.id] = {
                    "state": "succeeded", "error": None,
                    "outputs": {
                        "energy_ry": {"nm": -10.0, "fm": -10.1, "afm": -10.2}[node.id],
                        "natoms": 2,
                        "atom_magnetizations_muB": [2.0, -2.0] if node.id == "afm" else [],
                    },
                }
            else:
                node_outputs[node.id] = {
                    "state": "succeeded", "error": None,
                    "outputs": {"energy_ry": -10.0, "natoms": 2},
                }
        return {"run_id": "fake", "state": "succeeded", "nodes": node_outputs, "outputs": {}}

    monkeypatch.setattr(workflows, "_run_graph", fake_run_graph)
    conv = workflows.run_convergence({
        "material": "Si", "ecutwfc_values": [30, 40],
        "kpoints_values": [4], "workdir": str(tmp_path),
    })
    assert conv["selection"]["selected"]["ecutwfc"] == 30.0
    assert conv["error_curve"]["kind"] == "convergence"
    assert len(conv["error_curve"]["points"]) == 2

    mag = workflows.run_magnetic_compare({"material": "Fe", "workdir": str(tmp_path)})
    assert mag["comparison"]["ground_state"] == "AFM"
    ph = workflows.run_phonon_dfpt({"material": "MgO", "workdir": str(tmp_path)})
    phonon = captured["relax → SCF → ph.x → q2r.x → matdyn.x phonon workflow"]
    assert [node.id for node in phonon.nodes] == ["relax", "scf", "ph", "q2r", "matdyn"]
    assert phonon.node("ph").params["q_grid"] == [2, 2, 2]


def test_workflow_capabilities_are_visible_in_the_registry():
    from dft_forge.tools.definitions import register_default_tools
    from dft_forge.tools.registry import registry

    register_default_tools()
    ids = {tool.id for tool in registry.list_all()}
    assert {
        "workflow.convergence",
        "workflow.neutral_vacancy",
        "thermo.chemical_potential_scan",
        "workflow.magnetic_compare",
        "workflow.phonon_dfpt",
    } <= ids
