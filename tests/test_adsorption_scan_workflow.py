from pathlib import Path
from types import SimpleNamespace

import jsonschema
import pytest
from ase import Atoms

import dft_forge.workflows as workflows
from dft_forge.compiler import QECompiler
from dft_forge.parser import RY_TO_EV


def _method_profile(pseudos, *, kpoints=(2, 2, 1, 0, 0, 0), ecutwfc=40.0):
    return {
        "pseudos": dict(pseudos),
        "input_dft": "PBE",
        "effective": {
            "ecutwfc": ecutwfc,
            "ecutrho": 320.0,
            "kpoints": list(kpoints),
        },
    }


def _stub_adsorption_builders(monkeypatch):
    clean = SimpleNamespace(
        atoms=Atoms(
            "C2",
            scaled_positions=[(0, 0, 0.5), (1 / 3, 2 / 3, 0.5)],
            cell=[2.46, 2.46, 15.0],
            pbc=[True, True, False],
        ),
        selected_site=None,
        label="clean",
    )
    molecule = SimpleNamespace(
        atoms=Atoms(
            "O2",
            positions=[(5.0, 5.0, 4.4), (5.0, 5.0, 5.6)],
            cell=[10.0, 10.0, 10.0],
            pbc=False,
        ),
        selected_site=None,
        label="reference",
    )

    def build_2d(kind, *, adsorbate=None, **kwargs):
        assert kind == "graphene"
        if adsorbate is None:
            return clean
        site = adsorbate["site"]
        site_index = adsorbate["site_index"]
        return SimpleNamespace(
            atoms=Atoms(
                "C2O",
                positions=[(0, 0, 7.5), (1.2, 0.7, 7.5), (0.6, 0.35, 9.0)],
                cell=[2.46, 2.46, 15.0],
                pbc=[True, True, False],
            ),
            selected_site={
                "site_id": f"{site}:{site_index}:C0",
                "anchor_element": "C",
                "requested_site_index": site_index,
            },
            label=f"{site}_{site_index}",
        )

    monkeypatch.setattr("dft_forge.structure_builder.build_2d", build_2d)
    monkeypatch.setattr(
        "dft_forge.structure_builder.build_molecule",
        lambda kind, **kwargs: molecule,
    )
    def write_structure(result, path):
        from ase.io import write

        path.parent.mkdir(parents=True, exist_ok=True)
        write(str(path), result.atoms, format="cif")
        return str(path)

    monkeypatch.setattr(workflows, "_write_structure", write_structure)


def _args(tmp_path, **overrides):
    args = {
        "kind": "graphene",
        "adsorbate_element": "O",
        "supercell": "2x2",
        "sites": [
            {"site": "top", "site_index": 0, "height": 1.5},
            {"site": "bridge", "site_index": 0, "height": 1.6},
            {"site": "hollow", "site_index": 0, "height": 1.7},
        ],
        "reference_molecule": "o2",
        "relax": True,
        "ecutwfc": 40,
        "ecutrho": 320,
        "kpoints": [2, 2, 1, 0, 0, 0],
        "workdir": str(tmp_path),
    }
    args.update(overrides)
    return args


def _graph_payload(
    template,
    *,
    site_energies=(-106.0, -105.5, -105.0),
    failed=(),
    reference_magnetization=2.0,
    mismatch_site=None,
    site_magnetizations=None,
    final_site_distances=None,
):
    failed = set(failed)
    nodes = {}
    for node in template.nodes:
        if node.id in failed:
            nodes[node.id] = {
                "state": "failed",
                "error": "fixture failure",
                "outputs": None,
            }
        elif node.id.endswith("_relax"):
            structure = Path(node.params["structure"])
            final_path = structure
            if node.id.startswith("site_") and final_site_distances is not None:
                from ase.io import read, write

                atoms = read(str(structure))
                atoms.positions[-1, 2] = atoms.positions[0, 2] + final_site_distances[
                    int(node.id.split("_")[1]) - 1
                ]
                final_path = structure.with_name(f"{node.id}_relaxed.cif")
                write(str(final_path), atoms, format="cif")
            nodes[node.id] = {
                "state": "succeeded",
                "error": None,
                "outputs": {
                    "final_structure_cif": str(final_path),
                    "max_force_ry_bohr": 2.0e-4,
                },
            }
        elif node.id == "clean_scf":
            nodes[node.id] = {
                "state": "succeeded",
                "error": None,
                "outputs": {
                    "energy_ry": -100.0,
                    "natoms": 2,
                    "method_profile": _method_profile({"C": "C.upf"}),
                },
            }
        elif node.id == "reference_scf":
            nodes[node.id] = {
                "state": "succeeded",
                "error": None,
                "outputs": {
                    "energy_ry": -10.0,
                    "natoms": 2,
                    "total_magnetization_muB": reference_magnetization,
                    "method_profile": _method_profile(
                        {"O": "O.upf"}, kpoints=(1, 1, 1, 0, 0, 0)
                    ),
                },
            }
        else:
            ordinal = int(node.id.split("_")[1])
            ecutwfc = 50.0 if ordinal == mismatch_site else 40.0
            nodes[node.id] = {
                "state": "succeeded",
                "error": None,
                "outputs": {
                    "energy_ry": site_energies[ordinal - 1],
                    "natoms": 3,
                    **({
                        "total_magnetization_muB": site_magnetizations[ordinal - 1]
                    } if site_magnetizations is not None else {}),
                    "method_profile": _method_profile(
                        {"C": "C.upf", "O": "O.upf"}, ecutwfc=ecutwfc
                    ),
                },
            }
    return {
        "run_id": "fixture",
        "state": "failed" if failed else "succeeded",
        "nodes": nodes,
        "outputs": {},
    }


def test_three_sites_share_references_and_form_fixed_cell_relax_scf_pairs(
    monkeypatch, tmp_path
):
    _stub_adsorption_builders(monkeypatch)
    captured = {}

    def run(template, *args, **kwargs):
        captured["template"] = template
        return _graph_payload(template)

    monkeypatch.setattr(workflows, "_run_graph", run)
    result = workflows.run_adsorption_scan(_args(tmp_path))

    template = captured["template"]
    assert [node.id for node in template.nodes] == [
        "clean_relax",
        "clean_scf",
        "reference_relax",
        "reference_scf",
        "site_1_relax",
        "site_1_scf",
        "site_2_relax",
        "site_2_scf",
        "site_3_relax",
        "site_3_scf",
    ]
    assert sum(node.id == "clean_scf" for node in template.nodes) == 1
    assert sum(node.id == "reference_scf" for node in template.nodes) == 1
    for prefix in ("clean", "reference", "site_1", "site_2", "site_3"):
        relax = template.node(f"{prefix}_relax")
        scf = template.node(f"{prefix}_scf")
        assert relax.params["calc"] == "relax"
        assert scf.params["calc"] == "scf"
        assert scf.depends_on == [f"{prefix}_relax"]
        assert scf.params["structure"] == (
            f"${{nodes.{prefix}_relax.outputs.final_structure_cif}}"
        )
    assert all(node.repair == "" and node.max_repair_attempts == 0 for node in template.nodes)
    for node_id in ("reference_relax", "reference_scf"):
        params = template.node(node_id).params
        assert params["kpoints"] == [1, 1, 1, 0, 0, 0]
        assert params["spin_polarized"] is True
        assert params["total_magnetization_muB"] == pytest.approx(2.0)
        assert params["occupations"] == "fixed"

    assert result["status"] == "complete"
    assert [row["molecule_ratio"] for row in result["site_results"]] == [
        0.5,
        0.5,
        0.5,
    ]


def test_best_site_and_relative_energies_use_verified_adsorption_energies(
    monkeypatch, tmp_path
):
    _stub_adsorption_builders(monkeypatch)
    monkeypatch.setattr(
        workflows,
        "_run_graph",
        lambda template, *args, **kwargs: _graph_payload(template),
    )

    result = workflows.run_adsorption_scan(_args(tmp_path))
    rows = result["site_results"]

    assert [row["e_ads_ry"] for row in rows] == pytest.approx([-1.0, -0.5, 0.0])
    assert result["best_site"] == {
        "site_id": "top:0:C0",
        "e_ads_ev": pytest.approx(-RY_TO_EV),
    }
    assert [row["relative_to_lowest_successful_ev"] for row in rows] == pytest.approx(
        [0.0, 0.5 * RY_TO_EV, RY_TO_EV]
    )


def test_o2_magnetization_collapse_blocks_every_adsorption_reduction(
    monkeypatch, tmp_path
):
    _stub_adsorption_builders(monkeypatch)
    monkeypatch.setattr(
        workflows,
        "_run_graph",
        lambda template, *args, **kwargs: _graph_payload(
            template, reference_magnetization=0.0
        ),
    )

    result = workflows.run_adsorption_scan(_args(tmp_path))

    assert result["status"] == "partial"
    assert result["reference_validation"]["verified"] is False
    assert all("e_ads_ev" not in row for row in result["site_results"])
    assert "best_site" not in result
    assert any("O2 reference" in reason for reason in result["blocking_reasons"])


def test_one_failed_site_is_partial_and_ranks_only_successful_sites(
    monkeypatch, tmp_path
):
    _stub_adsorption_builders(monkeypatch)
    monkeypatch.setattr(
        workflows,
        "_run_graph",
        lambda template, *args, **kwargs: _graph_payload(
            template, failed={"site_2_scf"}
        ),
    )

    result = workflows.run_adsorption_scan(_args(tmp_path))

    assert result["status"] == "partial"
    assert "best_site" not in result
    assert result["best_site_among_successful"]["site_id"] == "top:0:C0"
    assert "e_ads_ev" not in result["site_results"][1]
    assert result["site_results"][1]["blocking_reasons"]


def test_method_mismatch_blocks_affected_site_from_ranking(monkeypatch, tmp_path):
    _stub_adsorption_builders(monkeypatch)
    monkeypatch.setattr(
        workflows,
        "_run_graph",
        lambda template, *args, **kwargs: _graph_payload(template, mismatch_site=2),
    )

    result = workflows.run_adsorption_scan(_args(tmp_path))

    assert result["status"] == "partial"
    assert "e_ads_ev" not in result["site_results"][1]
    assert any(
        "method mismatch for ecutwfc" in reason
        for reason in result["site_results"][1]["blocking_reasons"]
    )


def test_requested_site_magnetization_is_verified_before_ranking(monkeypatch, tmp_path):
    _stub_adsorption_builders(monkeypatch)
    monkeypatch.setattr(
        workflows,
        "_run_graph",
        lambda template, *args, **kwargs: _graph_payload(
            template, site_magnetizations=(2.0, 0.0, 2.0)
        ),
    )

    result = workflows.run_adsorption_scan(
        _args(
            tmp_path,
            adsorbed_spin={
                "spin_polarized": True,
                "magnetic_order": "FM",
                "magnetic_element": "O",
                "initial_moment_muB": 2.0,
                "total_magnetization_muB": 2.0,
            },
        )
    )

    assert result["status"] == "partial"
    assert result["site_results"][0]["spin_validation"]["verified"] is True
    assert result["site_results"][1]["spin_validation"]["verified"] is False
    assert "e_ads_ev" not in result["site_results"][1]
    assert "requested total magnetization" in result["site_results"][1]["blocking_reasons"][0]


def test_desorbed_final_geometry_is_excluded_from_scientific_ranking(monkeypatch, tmp_path):
    _stub_adsorption_builders(monkeypatch)
    monkeypatch.setattr(
        workflows,
        "_run_graph",
        lambda template, *args, **kwargs: _graph_payload(
            template, final_site_distances=(1.8, 4.0, 1.9)
        ),
    )

    result = workflows.run_adsorption_scan(_args(tmp_path))

    assert result["status"] == "partial"
    assert result["best_site_among_successful"]["site_id"] == "top:0:C0"
    row = result["site_results"][1]
    assert row["state"] == "succeeded"
    assert row["geometry_validation"]["verified"] is False
    assert row["geometry_validation"]["nearest_distance_angstrom"] > 3.0
    assert "e_ads_ev" not in row
    assert "final adsorption geometry" in row["blocking_reasons"][0]


def test_missing_final_structure_fails_closed_even_if_scf_succeeded(monkeypatch, tmp_path):
    _stub_adsorption_builders(monkeypatch)

    def payload(template, *args, **kwargs):
        result = _graph_payload(template)
        result["nodes"]["site_2_relax"]["outputs"]["final_structure_cif"] = "absent.cif"
        return result

    monkeypatch.setattr(workflows, "_run_graph", payload)
    result = workflows.run_adsorption_scan(_args(tmp_path))

    assert result["status"] == "partial"
    assert result["site_results"][1]["geometry_validation"]["verified"] is False
    assert "e_ads_ev" not in result["site_results"][1]


def test_final_extxyz_has_priority_and_must_preserve_slab_pbc(monkeypatch, tmp_path):
    _stub_adsorption_builders(monkeypatch)

    def payload(template, *args, **kwargs):
        from ase.io import read, write

        result = _graph_payload(template)
        source = Path(result["nodes"]["site_2_relax"]["outputs"]["final_structure_cif"])
        atoms = read(str(source))
        atoms.pbc = [True, True, True]  # Wrong PBC despite a valid legacy CIF.
        extxyz = tmp_path / "site_2_bad.extxyz"
        write(str(extxyz), atoms, format="extxyz")
        result["nodes"]["site_2_relax"]["outputs"]["final_structure_extxyz"] = str(extxyz)
        return result

    monkeypatch.setattr(workflows, "_run_graph", payload)
    result = workflows.run_adsorption_scan(_args(tmp_path))

    assert result["status"] == "partial"
    row = result["site_results"][1]
    assert row["geometry_validation"]["structure_source"] == "final_relaxed_extxyz"
    assert "PBC does not match" in row["geometry_validation"]["reason"]
    assert "e_ads_ev" not in row


@pytest.mark.parametrize("energy", [float("nan"), float("inf"), True])
def test_invalid_site_energy_never_enters_ranking(monkeypatch, tmp_path, energy):
    _stub_adsorption_builders(monkeypatch)

    def payload(template, *args, **kwargs):
        result = _graph_payload(template)
        result["nodes"]["site_2_scf"]["outputs"]["energy_ry"] = energy
        return result

    monkeypatch.setattr(workflows, "_run_graph", payload)
    result = workflows.run_adsorption_scan(_args(tmp_path))

    assert result["status"] == "partial"
    assert "e_ads_ev" not in result["site_results"][1]


def test_missing_reference_method_profile_blocks_adsorption(monkeypatch, tmp_path):
    _stub_adsorption_builders(monkeypatch)

    def payload(template, *args, **kwargs):
        result = _graph_payload(template)
        result["nodes"]["reference_scf"]["outputs"].pop("method_profile")
        return result

    monkeypatch.setattr(workflows, "_run_graph", payload)
    result = workflows.run_adsorption_scan(_args(tmp_path))

    assert result["status"] == "partial"
    assert all("e_ads_ev" not in row for row in result["site_results"])
    assert "method profile lacks" in result["blocking_reasons"][0]


def test_unrelaxed_scan_checks_initial_geometry(monkeypatch, tmp_path):
    _stub_adsorption_builders(monkeypatch)
    monkeypatch.setattr(
        workflows,
        "_run_graph",
        lambda template, *args, **kwargs: _graph_payload(template),
    )

    result = workflows.run_adsorption_scan(_args(tmp_path, relax=False))

    assert result["status"] == "complete"
    assert all(row["geometry_validation"]["verified"] for row in result["site_results"])
    assert all(
        row["geometry_validation"]["structure_source"] == "initial_structure"
        for row in result["site_results"]
    )


def test_adsorption_scan_tool_is_registered_with_multi_site_schema():
    from dft_forge.tools.definitions import register_default_tools
    from dft_forge.tools.registry import registry

    register_default_tools()
    tool = registry.get("workflow.adsorption_scan")

    assert tool is not None
    assert tool.side_effect == "external_execution"
    jsonschema.Draft7Validator.check_schema(tool.input_schema)
    item_schema = tool.input_schema["properties"]["sites"]["items"]["anyOf"][1]
    assert item_schema["properties"]["site_index"] == {
        "type": "integer",
        "minimum": 0,
    }
    assert item_schema["properties"]["height"]["exclusiveMinimum"] == 0
    assert tool.input_schema["properties"]["reference_magnetization_tolerance_muB"] == {
        "type": "number",
        "minimum": 0,
    }
    assert tool.input_schema["properties"]["adsorbed_magnetization_tolerance_muB"] == {
        "type": "number",
        "minimum": 0,
    }
    assert tool.input_schema["properties"]["max_adsorbate_surface_distance_angstrom"] == {
        "type": "number",
        "exclusiveMinimum": 0,
    }
    jsonschema.validate(
        {
            "kind": "graphene",
            "adsorbate_element": "O",
            "sites": [
                {"site": "top", "site_index": 0, "height": 1.5},
                {"site": "bridge", "site_index": 1, "height": 1.7},
            ],
        },
        tool.input_schema,
    )


def test_o2_fixed_cell_relax_compiles_spin_constraint_without_cell_block(tmp_path):
    pseudo_dir = Path(__file__).resolve().parents[1] / "assets" / "pseudos"
    compiler = QECompiler(pseudo_dir=pseudo_dir)
    atoms = Atoms(
        "O2",
        positions=[(5.0, 5.0, 4.4), (5.0, 5.0, 5.6)],
        cell=[10.0, 10.0, 10.0],
        pbc=False,
    )

    content = compiler.compile_t1(
        "O2",
        tmp_path / "o2_relax.in",
        atoms=atoms,
        cell_optimization=False,
        kpoints=(1, 1, 1, 0, 0, 0),
        spin_polarized=True,
        magnetic_order="FM",
        magnetic_element="O",
        initial_moment_muB=1.0,
        total_magnetization_muB=2.0,
        occupations="fixed",
    )

    assert 'calculation = "relax"' in content
    assert "nspin = 2" in content
    assert "tot_magnetization = 2" in content
    assert 'occupations = "fixed"' in content
    assert "starting_magnetization(1)" in content
    assert "\n&CELL\n" not in content
    assert "K_POINTS automatic\n  1 1 1  0 0 0" in content
