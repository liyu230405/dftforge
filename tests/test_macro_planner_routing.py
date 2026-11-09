"""Deterministic planner routing for composite defect/adsorption macros."""

from __future__ import annotations

import json

from dft_forge.agent_loop import AgentLoop
from dft_forge.agent_loop.intent import detect_intents
from dft_forge.agent_loop.llm_planner import LLMPlanner
from dft_forge.agent_loop.step_validator import semantic_plan_errors, validate_steps
from dft_forge.tools.registry import registry


def test_multisite_adsorption_intent_preserves_every_site_in_user_order():
    signals = detect_intents(
        "N adsorption energy on MoS2 at hollow/bridge/top sites"
    )
    assert signals.ads_el == "N"
    assert signals.ads_sites == ["hollow", "bridge", "top"]


def test_multisite_atomic_adsorption_routes_to_one_shared_reference_macro():
    message = "O吸附在石墨烯的 top/bridge/hollow 三个位点的吸附能"
    steps = AgentLoop().plan(message)
    assert steps == [{
        "tool": "workflow.adsorption_scan",
        "args": {
            "kind": "graphene",
            "adsorbate_element": "O",
            "sites": ["top", "bridge", "hollow"],
            "supercell": "3x3",
        },
        "description": steps[0]["description"],
    }]
    assert validate_steps(steps, registry) == []
    assert semantic_plan_errors(message, steps) == []


def test_chinese_multisite_names_are_preserved_without_defaulting_to_one_site():
    steps = AgentLoop().plan("N 在石墨烯顶位、桥位和空心位的吸附能")
    assert len(steps) == 1
    assert steps[0]["tool"] == "workflow.adsorption_scan"
    assert steps[0]["args"]["adsorbate_element"] == "N"
    assert steps[0]["args"]["sites"] == ["top", "bridge", "hollow"]


def test_named_multisite_subset_does_not_invent_an_unrequested_site():
    steps = AgentLoop().plan(
        "O adsorption energy on graphene at top/bridge multiple sites"
    )
    assert steps[0]["tool"] == "workflow.adsorption_scan"
    assert steps[0]["args"]["sites"] == ["top", "bridge"]


def test_explicit_nonzero_vacancy_charges_can_never_route_to_neutral_macro():
    message = "Si 的 Si 空位形成能，电荷态 +1/+2/-1"
    signals = detect_intents(message)
    assert signals.vacancy_charges == [1, 2, -1]
    steps = AgentLoop().plan(message)
    assert len(steps) == 1
    assert steps[0]["tool"] == "workflow.charged_vacancy"
    assert steps[0]["args"] == {
        "source": "Si",
        "element": "Si",
        "supercell": "2x2x2",
        "charges": [1, 2, -1],
    }
    assert validate_steps(steps, registry) == []


def test_charge_marker_accepts_unsigned_state_and_preserves_neutral_companion():
    steps = AgentLoop().plan("MgO O vacancy formation energy, charge states 0, 1 and -1")
    assert steps[0]["tool"] == "workflow.charged_vacancy"
    assert steps[0]["args"]["charges"] == [0, 1, -1]


def test_vacancy_without_explicit_nonzero_charge_remains_neutral():
    for message in ("MgO 的 O 空位形成能", "Si 的 Si 空位形成能 q=0"):
        steps = AgentLoop().plan(message)
        assert len(steps) == 1
        assert steps[0]["tool"] == "workflow.neutral_vacancy"


def test_signed_energy_observation_is_not_misread_as_a_charge_state():
    for message in (
        "Si vacancy formation energy with chemical potential -5 eV",
        "Si vacancy formation energy over Fermi level +1 eV",
    ):
        signals = detect_intents(message)
        assert signals.vacancy_charges == []
        assert AgentLoop().plan(message)[0]["tool"] == "workflow.neutral_vacancy"


class _CapturingProvider:
    def __init__(self, answer=None):
        self.answer = answer or json.dumps({"reply": "ok", "steps": []})
        self.calls = 0
        self.system = ""

    def chat(self, system, user):
        self.calls += 1
        self.system = system
        return self.answer


def test_llm_path_routes_loss_sensitive_macros_before_free_form_generation():
    provider = _CapturingProvider(json.dumps({
        "steps": [{
            "tool": "workflow.neutral_vacancy",
            "args": {"source": "Si", "element": "Si"},
        }]
    }))
    planner = LLMPlanner(registry)
    planner._provider, planner._checked = provider, True
    steps, _ = planner.plan("Si 的 Si 空位形成能，q=+1/+2", {})
    # Routing happened without a planning round-trip: the provider was only
    # asked for narrative prose, which cannot name a tool or change arguments.
    # The model's own suggestion (neutral_vacancy) is discarded either way.
    assert "不可更改" in provider.system
    assert steps[0]["tool"] == "workflow.charged_vacancy"
    assert steps[0]["args"]["charges"] == [1, 2]


def test_llm_prompt_names_macros_and_forbids_guessed_charged_defect_terms():
    provider = _CapturingProvider()
    planner = LLMPlanner(registry)
    planner._provider, planner._checked = provider, True
    planner.plan("你支持哪些计算", {})
    assert provider.calls == 1
    for text in (
        "workflow.adsorption_scan",
        "workflow.charged_vacancy",
        "Never guess dielectric screening",
        "potential alignment",
    ):
        assert text in provider.system
