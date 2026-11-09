from types import SimpleNamespace

import pytest
from ase import Atoms

import dft_forge.workflows as workflows
from dft_forge.parser import RY_TO_EV


def _stub_bulk_builders(monkeypatch):
    pristine = SimpleNamespace(
        atoms=Atoms(
            "Si2",
            scaled_positions=[(0, 0, 0), (0.25, 0.25, 0.25)],
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        ),
        to_dict=lambda: {"label": "pristine", "natoms": 2},
        label="pristine",
    )
    defect = SimpleNamespace(
        atoms=Atoms(
            "Si",
            scaled_positions=[(0, 0, 0)],
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        ),
        to_dict=lambda: {"label": "defect", "natoms": 1},
        label="defect",
    )
    monkeypatch.setattr(
        "dft_forge.structure_builder.supercell_3d",
        lambda source, supercell: pristine,
    )
    monkeypatch.setattr(
        "dft_forge.structure_builder.vacancy_3d",
        lambda source, element, supercell, index=None: defect,
    )
    monkeypatch.setattr(
        workflows,
        "_write_structure",
        lambda result, path: f"{result.label}.cif",
    )


def _workflow_args(**overrides):
    args = {
        "source": "Si",
        "element": "Si",
        "supercell": "1x1x1",
        "charges": [0, 1],
        "fermi_levels_ev": [0.0, 1.0],
        "band_gap_ev": 2.0,
        "reference_mode": "explicit",
        "chemical_potential_ry": 0.0,
        "correction_mode": "explicit",
        "potential_alignment_ev_by_charge": {"1": 0.1},
        "electrostatic_correction_ev_by_charge": {"1": 0.2},
        "spin_by_charge": {
            "1": {
                "spin_polarized": True,
                "magnetic_order": "FM",
                "magnetic_element": "Si",
                "initial_moment_muB": 1.0,
                "total_magnetization_muB": 1.0,
                "occupations": "smearing",
                "smearing": "mv",
                "degauss": 0.01,
            }
        },
        "relax": False,
    }
    args.update(overrides)
    return args


def _method_profile():
    return {
        "pseudos": {"Si": "Si.upf"},
        "effective": {
            "ecutwfc": 40.0,
            "ecutrho": 320.0,
            "kpoints": [2, 2, 2, 0, 0, 0],
        },
    }


def _successful_graph_payload(template, *, energies=None, failed=()):
    energies = energies or {0: -9.9, 1: -9.95}
    failed = set(failed)
    nodes = {}
    for node in template.nodes:
        if node.id in failed:
            nodes[node.id] = {"state": "failed", "error": "fixture failure", "outputs": None}
        elif node.id == "pristine_scf":
            nodes[node.id] = {
                "state": "succeeded",
                "error": None,
                "outputs": {
                    "energy_ry": -10.0, "natoms": 2,
                    "method_profile": _method_profile(),
                },
            }
        elif node.id.endswith("_relax"):
            nodes[node.id] = {
                "state": "succeeded",
                "error": None,
                "outputs": {"final_structure_cif": f"{node.id}.cif"},
            }
        else:
            charge = 0 if node.id == "q_0_scf" else (1 if "q_p1" in node.id else -1)
            nodes[node.id] = {
                "state": "succeeded",
                "error": None,
                "outputs": {
                    "energy_ry": energies[charge],
                    "natoms": 1,
                    "method_profile": _method_profile(),
                    **({"total_magnetization_muB": 1.0} if charge == 1 else {}),
                },
            }
    return {"run_id": "fixture", "state": "succeeded", "nodes": nodes, "outputs": {}}


def test_charged_vacancy_formation_energy_reports_each_explicit_term():
    result = workflows.charged_vacancy_formation_energy(
        defect_energy_ry=-99.5,
        pristine_energy_ry=-100.0,
        chemical_potential_ry=-1.0,
        charge=2,
        fermi_level_from_vbm_ev=0.3,
        potential_alignment_ev=-0.1,
        electrostatic_correction_ev=0.4,
    )

    raw_ev = -0.5 * RY_TO_EV
    assert result == pytest.approx(
        {
            "charge": 2,
            "formation_energy_ev": raw_ev + 0.6 - 0.2 + 0.4,
            "raw_energy_and_mu_ev": raw_ev,
            "fermi_level_from_vbm_ev": 0.3,
            "fermi_term_ev": 0.6,
            "potential_alignment_ev": -0.1,
            "potential_alignment_term_ev": -0.2,
            "electrostatic_correction_ev": 0.4,
        }
    )


def test_makov_payne_uses_q_squared_and_cubic_cell_length():
    cell = Atoms(cell=[20.0, 20.0, 20.0], pbc=True).cell
    positive = workflows.makov_payne_correction_ev(
        cell, charge=2, dielectric_constant=10.0
    )
    negative = workflows.makov_payne_correction_ev(
        cell, charge=-2, dielectric_constant=10.0
    )
    expected = 4 * 2.837297 * 14.3996454784255 / (2 * 10.0 * 20.0)

    assert positive["electrostatic_correction_ev"] == pytest.approx(expected)
    assert negative["electrostatic_correction_ev"] == pytest.approx(expected)
    assert positive["cell_length_angstrom"] == pytest.approx(20.0)


def test_makov_payne_rejects_non_cubic_cell():
    cell = Atoms(cell=[20.0, 20.0, 24.0], pbc=True).cell

    with pytest.raises(workflows.WorkflowError, match="near-cubic"):
        workflows.makov_payne_correction_ev(
            cell, charge=1, dielectric_constant=11.7
        )


@pytest.mark.parametrize(
    ("missing", "message"),
    [
        ("spin_by_charge", "spin_by_charge"),
        ("potential_alignment_ev_by_charge", "potential_alignment"),
        ("electrostatic_correction_ev_by_charge", "electrostatic_correction"),
    ],
)
def test_missing_charged_defect_evidence_fails_before_graph(
    monkeypatch, tmp_path, missing, message
):
    args = _workflow_args(workdir=str(tmp_path))
    args.pop(missing)
    monkeypatch.setattr(
        workflows,
        "_run_graph",
        lambda *args, **kwargs: pytest.fail("invalid evidence reached QE graph execution"),
    )

    with pytest.raises(workflows.WorkflowError, match=message):
        workflows.run_charged_vacancy(args)


def test_missing_makov_payne_dielectric_fails_closed_before_graph(monkeypatch, tmp_path):
    _stub_bulk_builders(monkeypatch)
    args = _workflow_args(
        workdir=str(tmp_path),
        correction_mode="makov_payne_cubic",
    )
    args.pop("electrostatic_correction_ev_by_charge")
    monkeypatch.setattr(
        workflows,
        "_run_graph",
        lambda *args, **kwargs: pytest.fail("missing dielectric reached QE graph execution"),
    )

    with pytest.raises(workflows.WorkflowError, match="dielectric"):
        workflows.run_charged_vacancy(args)


def test_charge_and_spin_configuration_reaches_relax_and_scf(monkeypatch, tmp_path):
    _stub_bulk_builders(monkeypatch)
    captured = {}

    def fake_run(template, *args, **kwargs):
        captured["template"] = template
        return _successful_graph_payload(template)

    monkeypatch.setattr(workflows, "_run_graph", fake_run)
    workflows.run_charged_vacancy(
        _workflow_args(workdir=str(tmp_path), relax=True)
    )

    template = captured["template"]
    assert [node.id for node in template.nodes] == [
        "pristine_scf",
        "q_0_relax",
        "q_0_scf",
        "q_p1_relax",
        "q_p1_scf",
    ]
    for node_id in ("q_0_relax", "q_0_scf"):
        node = template.node(node_id)
        assert node.params["tot_charge"] == 0
        assert node.params["spin_polarized"] is False
    for node_id in ("q_p1_relax", "q_p1_scf"):
        node = template.node(node_id)
        assert node.params["tot_charge"] == 1
        assert node.params["spin_polarized"] is True
        assert node.params["magnetic_order"] == "FM"
        assert node.params["initial_moment_muB"] == pytest.approx(1.0)
        assert node.params["total_magnetization_muB"] == pytest.approx(1.0)
        assert node.params["occupations"] == "smearing"
        assert node.params["smearing"] == "mv"
        assert node.params["degauss"] == pytest.approx(0.01)
    assert all(node.repair == "" and node.max_repair_attempts == 0 for node in template.nodes)
    assert template.node("q_p1_scf").depends_on == ["q_p1_relax"]
    assert (
        template.node("q_p1_scf").params["structure"]
        == "${nodes.q_p1_relax.outputs.final_structure_cif}"
    )


def test_failed_charge_state_is_not_reduced_into_formation_lines(monkeypatch, tmp_path):
    _stub_bulk_builders(monkeypatch)
    monkeypatch.setattr(
        workflows,
        "_run_graph",
        lambda template, *args, **kwargs: _successful_graph_payload(
            template, failed={"q_p1_scf"}
        ),
    )

    result = workflows.run_charged_vacancy(
        _workflow_args(workdir=str(tmp_path))
    )

    assert result["status"] == "partial"
    assert [line["charge"] for line in result["formation_lines"]] == [0]
    assert "transition_levels" not in result
    assert "lack verified energy/atom-count evidence" in result["blocking_reasons"][0]


def test_failed_pristine_reference_is_not_reduced_into_formation_lines(
    monkeypatch, tmp_path
):
    _stub_bulk_builders(monkeypatch)
    monkeypatch.setattr(
        workflows,
        "_run_graph",
        lambda template, *args, **kwargs: _successful_graph_payload(
            template, failed={"pristine_scf"}
        ),
    )

    result = workflows.run_charged_vacancy(
        _workflow_args(workdir=str(tmp_path))
    )

    assert result["status"] == "partial"
    assert result["formation_lines"] == []
    assert "transition_levels" not in result


def test_transition_level_is_derived_from_verified_line_intercepts(monkeypatch, tmp_path):
    _stub_bulk_builders(monkeypatch)
    monkeypatch.setattr(
        workflows,
        "_run_graph",
        lambda template, *args, **kwargs: _successful_graph_payload(template),
    )

    result = workflows.run_charged_vacancy(
        _workflow_args(workdir=str(tmp_path))
    )

    assert result["status"] == "complete"
    assert result["charge_validations"]["1"]["verified"] is True
    lines = {line["charge"]: line for line in result["formation_lines"]}
    expected = lines[0]["intercept_ev_at_vbm"] - lines[1]["intercept_ev_at_vbm"]
    assert result["transition_levels"] == [
        {
            "charges": [0, 1],
            "fermi_level_from_vbm_ev": pytest.approx(expected),
            "within_supplied_gap": True,
        }
    ]
    assert lines[0]["points"][1]["formation_energy_ev"] == pytest.approx(
        lines[0]["intercept_ev_at_vbm"]
    )
    assert lines[1]["points"][1]["formation_energy_ev"] == pytest.approx(
        lines[1]["intercept_ev_at_vbm"] + 1.0
    )


def test_charged_state_magnetization_mismatch_blocks_formation_line(monkeypatch, tmp_path):
    _stub_bulk_builders(monkeypatch)

    def payload(template, *args, **kwargs):
        result = _successful_graph_payload(template)
        result["nodes"]["q_p1_scf"]["outputs"]["total_magnetization_muB"] = 0.0
        return result

    monkeypatch.setattr(workflows, "_run_graph", payload)
    result = workflows.run_charged_vacancy(_workflow_args(workdir=str(tmp_path)))

    assert result["status"] == "partial"
    assert [line["charge"] for line in result["formation_lines"]] == [0]
    assert result["charge_validations"]["1"]["verified"] is False
    assert "requested total magnetization" in result["blocking_reasons"][0]


@pytest.mark.parametrize("missing_node", ["pristine_scf", "q_p1_scf"])
def test_missing_method_provenance_blocks_charged_formation(
    monkeypatch, tmp_path, missing_node
):
    _stub_bulk_builders(monkeypatch)

    def payload(template, *args, **kwargs):
        result = _successful_graph_payload(template)
        result["nodes"][missing_node]["outputs"].pop("method_profile")
        return result

    monkeypatch.setattr(workflows, "_run_graph", payload)
    result = workflows.run_charged_vacancy(_workflow_args(workdir=str(tmp_path)))

    assert result["status"] == "partial"
    assert result["formation_lines"] == []
    assert "method profile lacks" in result["blocking_reasons"][0]


def test_nonfinite_charge_state_energy_blocks_charged_formation(monkeypatch, tmp_path):
    _stub_bulk_builders(monkeypatch)

    def payload(template, *args, **kwargs):
        result = _successful_graph_payload(template)
        result["nodes"]["q_p1_scf"]["outputs"]["energy_ry"] = float("nan")
        return result

    monkeypatch.setattr(workflows, "_run_graph", payload)
    result = workflows.run_charged_vacancy(_workflow_args(workdir=str(tmp_path)))

    assert result["status"] == "partial"
    assert [line["charge"] for line in result["formation_lines"]] == [0]
    assert "transition_levels" not in result
