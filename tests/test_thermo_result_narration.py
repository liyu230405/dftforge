"""Scientific wording for formation and adsorption result summaries."""

from pathlib import Path

import pytest

from dft_forge.agent_loop.result_views import RunState, handle_tool_result


def _render(tmp_path: Path, tool_id: str, data: dict) -> tuple[str, dict]:
    result = {}
    replies = []
    handle_tool_result(tool_id, data, {}, "测试", RunState({}, tmp_path), result, replies)
    return "\n".join(replies), result


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (-0.4, "相对所选元素参考态的形成反应有利（负值）"),
        (0.4, "相对所选元素参考态的形成反应不利（正值）"),
        (0.0, "相对所选元素参考态能量差为零"),
    ],
)
def test_formation_narration_describes_reference_reaction(tmp_path, value, expected):
    text, result = _render(
        tmp_path,
        "thermo.formation",
        {
            "formula": "AB",
            "formation_energy_ev_per_atom": value,
            "formation_energy_ev_per_formula": value * 2,
        },
    )

    assert expected in text
    assert "相对单质稳定" not in text
    assert "相对单质不稳定" not in text
    assert "凸包或全局热力学稳定性" in text
    assert result["metrics"] == {"formation_energy_ev_per_atom": value}


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (-0.4, "相对所选参考态放热（负值）"),
        (0.4, "相对所选参考态吸热（正值）"),
        (0.0, "相对所选参考态能量差为零"),
    ],
)
def test_adsorption_narration_describes_reference_heat(tmp_path, value, expected):
    text, result = _render(
        tmp_path,
        "thermo.eads",
        {
            "e_ads_ev": value,
            "e_ads_ev_per_adsorbate": value / 2,
        },
    )

    assert expected in text
    assert "eV/原子" not in text
    assert "强吸附" not in text
    assert "中等吸附" not in text
    assert "弱吸附" not in text
    assert "体系稳定" not in text
    assert result["metrics"] == {"e_ads_ev": value}
