"""Regression coverage for macro intent, schema, and result narration contracts."""

from __future__ import annotations

import json

from dft_forge.agent_loop import AgentLoop
from dft_forge.agent_loop.intent import detect_intents
from dft_forge.agent_loop.llm_planner import LLMPlanner
from dft_forge.agent_loop.result_views import _handle_scientific_workflow
from dft_forge.agent_loop.step_validator import validate_steps
from dft_forge.tools.registry import registry


class _NeverCalledProvider:
    calls = 0

    def chat(self, _system, _user):
        self.calls += 1
        return json.dumps({"steps": []})


def _render(tool_id: str, data: dict) -> tuple[str, dict]:
    reply: list[str] = []
    result: dict = {}
    _handle_scientific_workflow(tool_id, data, reply, result)
    return reply[0], result


def test_adsorbate_after_chinese_verb_and_generic_three_sites_are_preserved():
    message = "石墨烯吸附N的吸附能，比较三个吸附位点"
    signals = detect_intents(message)

    assert signals.ads_el == "N"
    assert signals.ads_sites == ["top", "bridge", "hollow"]
    steps = AgentLoop().plan(message)
    assert steps[0]["tool"] == "workflow.adsorption_scan"
    assert steps[0]["args"]["adsorbate_element"] == "N"
    assert steps[0]["args"]["sites"] == ["top", "bridge", "hollow"]
    assert validate_steps(steps, registry) == []


def test_missing_adsorbate_never_defaults_to_oxygen_on_rules_or_llm_path():
    message = "石墨烯多位点吸附能"
    rule_steps = AgentLoop().plan(message)
    assert rule_steps[0]["tool"] == "help"
    assert "明确原子吸附物种" in rule_steps[0]["description"]

    provider = _NeverCalledProvider()
    planner = LLMPlanner(registry)
    planner._provider, planner._checked = provider, True
    llm_steps, _ = planner.plan(message, {})
    assert provider.calls == 0
    assert llm_steps == rule_steps


def test_molecular_adsorbate_is_not_coerced_to_element_symbol():
    signals = detect_intents("CO吸附在石墨烯的吸附能")
    assert signals.ads_el == "CO"
    steps = AgentLoop().plan("CO吸附在石墨烯的吸附能")
    assert steps[0]["tool"] == "help"
    assert "分子吸附" in steps[0]["description"]


def test_labeled_adsorbate_species_is_not_treated_as_missing():
    for message, element in (
        ("计算石墨烯吸附能（O 原子）", "O"),
        ("石墨烯吸附能，吸附物种为 N", "N"),
    ):
        steps = AgentLoop().plan(message)
        assert steps[0]["tool"] == "workflow.adsorption_scan"
        assert steps[0]["args"]["adsorbate_element"] == element


def test_partial_charged_vacancy_narration_keeps_partial_status_and_reason():
    text, metrics = _render("workflow.charged_vacancy", {
        "status": "partial",
        "correction_mode": "explicit",
        "formation_lines": [{
            "charge": 0,
            "intercept_ev_at_vbm": 1.0,
            "points": [],
        }],
        "blocking_reasons": ["q=+1 SCF failed"],
    })

    assert "部分完成" in text
    assert "q=+1 SCF failed" in text
    assert "formation_lines" in metrics["metrics"]


def test_adsorption_narration_limits_scope_and_only_names_o2_triplet_for_o2():
    base = {
        "status": "partial",
        "site_results": [
            {"site_id": "top", "e_ads_ev": 1.2},
            {"site_id": "bridge", "state": "failed"},
        ],
        "best_site_among_successful": {"site_id": "top", "e_ads_ev": 1.2},
        "reference_validation": {"verified": True},
        "blocking_reasons": ["bridge SCF failed"],
    }
    nitrogen, _ = _render("workflow.adsorption_scan", {**base, "reference_molecule": "n2"})
    oxygen, _ = _render("workflow.adsorption_scan", {**base, "reference_molecule": "o2"})

    assert "部分完成" in nitrogen and "暂定排序" in nitrogen
    assert "bridge SCF failed" in nitrogen
    assert "不是全局构型搜索" in nitrogen
    assert "O₂三重态" not in nitrogen
    assert "O₂三重态验证=True" in oxygen


def test_result_narration_does_not_crash_on_missing_vbm_value_or_nan_adsorption():
    charged, _ = _render("workflow.charged_vacancy", {
        "status": "complete",
        "correction_mode": "explicit",
        "formation_lines": [{"charge": 1, "points": []}],
    })
    adsorbed, _ = _render("workflow.adsorption_scan", {
        "status": "partial",
        "reference_molecule": "o2",
        "site_results": [{"site_id": "top", "e_ads_ev": float("nan")}],
        "blocking_reasons": ["non-finite energy"],
    })

    assert "未提供" in charged
    assert "未产生可验证的吸附能" in adsorbed
