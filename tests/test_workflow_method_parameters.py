from types import SimpleNamespace

import pytest

import dft_forge.workflows as workflows
from dft_forge.runtime.graph import GraphTemplate, NodeSpec


def _fake_runner(captured):
    def run(template, workdir, inputs=None, **controls):
        captured[template.description] = template
        nodes = {}
        for index, node in enumerate(template.nodes):
            nodes[node.id] = {
                "state": "succeeded",
                "error": None,
                "outputs": {"energy_ry": -10.0 - index / 100.0, "natoms": 2},
            }
        return {"run_id": "fake", "state": "succeeded", "nodes": nodes, "outputs": {}}

    return run


def test_neutral_vacancy_passes_method_parameters_to_each_geometry(monkeypatch, tmp_path):
    captured = {}

    class FakeStructure:
        def __init__(self, label):
            self.label = label

        def to_dict(self):
            return {"label": self.label}

    monkeypatch.setattr(
        "dft_forge.structure_builder.supercell_3d",
        lambda source, supercell: FakeStructure("pristine"),
    )
    monkeypatch.setattr(
        "dft_forge.structure_builder.vacancy_3d",
        lambda source, element, supercell, index=None: FakeStructure("defect"),
    )
    monkeypatch.setattr(
        workflows,
        "_write_structure",
        lambda structure, path: f"{structure.label}.cif",
    )
    monkeypatch.setattr(workflows, "_run_graph", _fake_runner(captured))

    workflows.run_neutral_vacancy(
        {
            "source": "MgO",
            "element": "O",
            "relax": True,
            "chemical_potential_ry": -0.5,
            "ecutwfc": 70,
            "ecutrho": 560,
            "kpoints": [3, 3, 3],
            "conv_thr": 1e-10,
            "workdir": str(tmp_path),
        }
    )

    template = captured["neutral bulk vacancy formation workflow"]
    assert [node.id for node in template.nodes] == [
        "pristine_scf",
        "defect_relax",
        "defect_scf",
    ]
    expected = {
        "material": "MgO",
        "ecutwfc": 70,
        "ecutrho": 560,
        "kpoints": [3, 3, 3],
        "conv_thr": 1e-10,
    }
    for node in template.nodes:
        assert {key: node.params[key] for key in expected} == expected
    assert template.node("pristine_scf").params["structure"] == "pristine.cif"
    assert template.node("defect_relax").params["structure"] == "defect.cif"
    assert (
        template.node("defect_scf").params["structure"]
        == "${nodes.defect_relax.outputs.final_structure_cif}"
    )


def test_magnetic_states_share_explicit_method_parameters(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(workflows, "_run_graph", _fake_runner(captured))

    workflows.run_magnetic_compare(
        {
            "material": "Fe",
            "structure": "fe.cif",
            "orders": ["NM", "FM", "AFM"],
            "magnetic_element": "Fe",
            "ecutwfc": 80,
            "ecutrho": 640,
            "kpoints": [2, 2, 2],
            "conv_thr": 1e-9,
            "workdir": str(tmp_path),
        }
    )

    template = captured["collinear NM/FM/AFM magnetic comparison"]
    assert [node.id for node in template.nodes] == ["nm", "fm", "afm"]
    expected = {
        "material": "Fe",
        "structure": "fe.cif",
        "ecutwfc": 80,
        "ecutrho": 640,
        "kpoints": [2, 2, 2],
        "conv_thr": 1e-9,
    }
    for node in template.nodes:
        assert {key: node.params[key] for key in expected} == expected


def test_phonon_nodes_keep_method_parameters_and_tool_specific_controls(
    monkeypatch, tmp_path
):
    captured = {}
    monkeypatch.setattr(workflows, "_run_graph", _fake_runner(captured))

    workflows.run_phonon_dfpt(
        {
            "material": "MgO",
            "structure": "input.cif",
            "q_grid": [2, 2, 2],
            "ecutwfc": 65,
            "ecutrho": 520,
            "kpoints": [4, 4, 4],
            "conv_thr": 1e-11,
            "tr2_ph": 1e-13,
            "workdir": str(tmp_path),
        }
    )

    template = next(
        template
        for template in captured.values()
        if [node.id for node in template.nodes] == ["relax", "scf", "ph", "q2r", "matdyn"]
    )
    assert [node.id for node in template.nodes] == ["relax", "scf", "ph", "q2r", "matdyn"]
    expected = {
        "material": "MgO",
        "ecutwfc": 65,
        "ecutrho": 520,
        "kpoints": [4, 4, 4],
        "conv_thr": 1e-11,
    }
    for node in template.nodes[:3]:
        assert {key: node.params[key] for key in expected} == expected
    for node in template.nodes[3:]:
        assert node.params["material"] == "MgO"
        assert all(key not in node.params for key in expected if key != "material")
    assert template.node("relax").params["structure"] == "input.cif"
    assert template.node("scf").params["structure"] == "${nodes.relax.outputs.final_structure_cif}"
    assert "structure" not in template.node("ph").params
    assert "structure" not in template.node("q2r").params
    assert "structure" not in template.node("matdyn").params
    assert template.node("ph").params["q_grid"] == [2, 2, 2]
    assert template.node("ph").params["tr2_ph"] == 1e-13
    assert "q_grid" not in template.node("q2r").params
    assert "tr2_ph" not in template.node("q2r").params
    assert template.node("matdyn").params["q_grid"] == [2, 2, 2]
    assert "tr2_ph" not in template.node("matdyn").params


def test_method_contract_uses_atomic_run_metadata_and_legacy_path_stays_compatible(
    monkeypatch, tmp_path
):
    calls = []

    class FakeEngine:
        def register_template(self, template):
            calls.append(("register", template.template_id))

        def run_template(self, *args, **kwargs):
            calls.append(("run", args, kwargs))
            return SimpleNamespace()

    engine = FakeEngine()
    monkeypatch.setattr("dft_forge.tools.graph_tools._get_engine", lambda workdir: engine)
    monkeypatch.setattr(workflows, "_graph_payload", lambda *args: {"ok": True})
    template = GraphTemplate(template_id="t", nodes=[NodeSpec(id="n", tool="qe")])

    workflows._run_graph(template, tmp_path, method_contract={"step": 1})
    assert calls[-1][0] == "run"
    assert calls[-1][2]["run_metadata"] == {"method_contract": {"step": 1}}

    calls.clear()
    workflows._run_graph(template, tmp_path)
    assert calls[-1][0] == "run"
    assert "run_metadata" not in calls[-1][2]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ecutwfc_values": [float("nan")]},
        {"ecutwfc_values": [float("inf")]},
        {"ecutwfc_values": [0]},
        {"degauss_values": [float("nan")]},
        {"degauss_values": [-0.1]},
        {"kpoints_values": [[2.5, 2, 2]]},
        {"kpoints_values": [[2, 2, 2, 2, 0, 0]]},
        {"kpoints_values": [True]},
        {"max_jobs": 0},
        {"max_jobs": 1.5},
        {"max_jobs": True},
        {"max_jobs": 257},
    ],
)
def test_convergence_grid_rejects_invalid_preflight_values(kwargs):
    with pytest.raises(workflows.WorkflowError):
        workflows.convergence_grid(**kwargs)


def test_convergence_grid_checks_job_cap_before_building_points():
    with pytest.raises(workflows.WorkflowError, match="above max_jobs"):
        workflows.convergence_grid(
            ecutwfc_values=list(range(1, 21)),
            kpoints_values=list(range(1, 21)),
            max_jobs=256,
        )


def test_run_convergence_preflights_before_calling_graph(monkeypatch, tmp_path):
    def should_not_run(*args, **kwargs):
        raise AssertionError("invalid convergence input reached graph execution")

    monkeypatch.setattr(workflows, "_run_graph", should_not_run)
    with pytest.raises(workflows.WorkflowError):
        workflows.run_convergence(
            {"material": "Si", "ecutwfc_values": [float("nan")], "workdir": str(tmp_path)}
        )
