"""Regression tests for lattice facts shown in graph result narration."""

from dft_forge.agent_loop.result_views import _graph_facts


def test_cu_analysis_reports_actual_and_conventional_cells():
    data = {
        "outputs": {"a_angstrom": 3.61889093374459, "volume_a3": 11.912756},
        "nodes": {
            "vc_relax": {
                "outputs": {
                    "a_angstrom": 3.61889093374459,
                    "analysis": {
                        "lattice_abc_angstrom": [2.5589, 2.5589, 2.5682],
                        "lattice_angles_deg": [60.12, 60.12, 60.0],
                        "symmetry": {
                            "space_group": "R-3m",
                            "conventional_cell": {
                                "a_angstrom": 2.5589,
                                "b_angstrom": 2.5589,
                                "c_angstrom": 6.3021,
                            },
                        },
                    },
                }
            }
        },
    }

    facts = _graph_facts(data)

    assert "近 FCC 等效立方 a (Å)=3.618891" in facts
    assert "实际晶胞 a (Å)=2.5589" in facts
    assert "实际晶胞 b (Å)=2.5589" in facts
    assert "实际晶胞 c (Å)=2.5682" in facts
    assert "实际晶胞 α (°)=60.12" in facts
    assert "实际晶胞 β (°)=60.12" in facts
    assert "实际晶胞 γ (°)=60.0" in facts
    assert "常规胞 a (Å)=2.5589" in facts
    assert "常规胞 b (Å)=2.5589" in facts
    assert "常规胞 c (Å)=6.3021" in facts
    assert "空间群=R-3m" in facts


def test_ideal_fcc_equivalent_cell_uses_fcc_label():
    facts = _graph_facts(
        {
            "outputs": {"a_angstrom": 3.535534},
            "nodes": {
                "relax": {
                    "outputs": {
                        "a_angstrom": 3.535534,
                        "analysis": {
                            "lattice_abc_angstrom": [2.5, 2.5, 2.5],
                            "lattice_angles_deg": [60.0, 60.0, 60.0],
                        },
                    }
                }
            },
        }
    )

    assert "近 FCC 等效立方 a (Å)=3.535534" in facts


def test_without_analysis_legacy_lattice_value_is_noncommittal():
    facts = _graph_facts({"outputs": {"a_angstrom": 3.2}})

    assert "解析晶格参数 a (Å)=3.2" in facts
    assert "平衡晶格参数" not in facts
    assert "近 FCC 等效立方" not in facts
