"""Deterministically routed macros get narration without losing the routing.

The invariant under test: narration is narrative-only. Tool ids, arguments and
step counts are decided by deterministic rules before narration runs, and no
reply — well-formed, malformed, adversarial or absent — may change them.
"""

import json

import pytest

from dft_forge.agent_loop.llm_planner import LLMPlanner
from dft_forge.agent_loop.macro_narration import (
    apply_macro_narration,
    narrate_macro_plan,
)
from dft_forge.agent_loop.planning import PlanIR, render_plan_reply
from dft_forge.tools.registry import registry

ADSORPTION_MESSAGE = "算 O 在 MoS2 单层上 top、bridge、hollow 三个位点的吸附能，用 O2 做参考态。先给规划，不要实际运行。"

DECIDED_STEPS = [{
    "tool": "workflow.adsorption_scan",
    "args": {
        "kind": "mos2",
        "adsorbate_element": "O",
        "sites": ["top", "bridge", "hollow"],
        "supercell": "3x3",
    },
    "description": "共享纯表面和 O2 参考，固定晶胞松弛并比较吸附能",
}]


class FakeProvider:
    """Records prompts and returns a scripted reply."""

    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def chat(self, system, user):
        self.calls.append({"system": system, "user": user})
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def _good_reply(**overrides):
    payload = {
        "reply": "该方案对 MoS2 单层的三个请求位点各构建一个 O 吸附构型，"
                 "共享同一纯表面与 O2 三重态参考，固定晶胞仅松弛离子位置。",
        "goal": "比较 O@MoS2 三个位点的吸附能",
        "assumptions": ["固定晶胞近似适用于该表面", "O2 取三重态基态作为气相参考"],
        "expected_outputs": ["每个位点的 E_ads (eV)", "位点排序"],
    }
    payload.update(overrides)
    return json.dumps(payload, ensure_ascii=False)


def test_narration_fills_reply_for_decided_macro():
    narrative = narrate_macro_plan(FakeProvider(_good_reply()), ADSORPTION_MESSAGE, DECIDED_STEPS)
    assert narrative is not None
    assert "三个请求位点" in narrative["reply"]
    assert narrative["goal"] == "比较 O@MoS2 三个位点的吸附能"
    assert len(narrative["assumptions"]) == 2


def test_narration_prompt_carries_decided_tool_and_args_only():
    provider = FakeProvider(_good_reply())
    narrate_macro_plan(provider, ADSORPTION_MESSAGE, DECIDED_STEPS)
    sent = json.loads(provider.calls[0]["user"])
    assert sent["decided_plan"][0]["tool"] == "workflow.adsorption_scan"
    assert sent["decided_plan"][0]["args"]["sites"] == ["top", "bridge", "hollow"]
    # The model is told the plan is fixed, so it cannot propose substitutions.
    assert "不可更改" in provider.calls[0]["system"]


@pytest.mark.parametrize("reply", [
    "not json at all",
    "{}",
    '{"reply": ""}',
    '{"reply": "   "}',
    "[]",
    '{"goal": "只有目标没有说明"}',
    json.dumps({"reply": None}),
])
def test_unusable_narration_returns_none(reply):
    assert narrate_macro_plan(FakeProvider(reply), ADSORPTION_MESSAGE, DECIDED_STEPS) is None


def test_provider_failure_returns_none():
    assert narrate_macro_plan(
        FakeProvider(RuntimeError("timeout")), ADSORPTION_MESSAGE, DECIDED_STEPS
    ) is None


def test_missing_provider_or_steps_returns_none():
    assert narrate_macro_plan(None, ADSORPTION_MESSAGE, DECIDED_STEPS) is None
    assert narrate_macro_plan(FakeProvider(_good_reply()), ADSORPTION_MESSAGE, []) is None


def test_missing_adsorbate_help_path_is_not_narrated(monkeypatch):
    """A question back to the user is not a decided plan, so nothing narrates it."""
    planner = LLMPlanner(registry)
    provider = FakeProvider(_good_reply())
    monkeypatch.setattr(planner, "_get_provider", lambda: provider)

    steps, _ = planner.plan("石墨烯多位点吸附能", {})

    assert steps[0]["tool"] == "help"
    assert provider.calls == []
    assert planner.last_plan_ir.source == "deterministic-macro"


def test_fenced_json_is_accepted():
    fenced = "```json\n" + _good_reply() + "\n```"
    narrative = narrate_macro_plan(FakeProvider(fenced), ADSORPTION_MESSAGE, DECIDED_STEPS)
    assert narrative is not None and narrative["reply"]


def test_narration_cannot_change_actions_or_args():
    plan = PlanIR.from_steps(ADSORPTION_MESSAGE, DECIDED_STEPS, source="deterministic-macro")
    before = json.dumps(plan.actions, sort_keys=True, ensure_ascii=False)
    hostile = json.dumps({
        "reply": "换用中性空位流程更合适",
        # Every one of these must be ignored: they are not narrative fields.
        "actions": [{"tool": "workflow.neutral_vacancy", "args": {}}],
        "steps": [{"tool": "workflow.neutral_vacancy", "args": {}}],
        "tool": "workflow.neutral_vacancy",
        "args": {"sites": ["top"]},
        "execution_mode": "execute",
        "support_status": "unsupported",
    }, ensure_ascii=False)
    applied = apply_macro_narration(
        plan, narrate_macro_plan(FakeProvider(hostile), ADSORPTION_MESSAGE, DECIDED_STEPS)
    )
    assert applied is True
    assert json.dumps(plan.actions, sort_keys=True, ensure_ascii=False) == before
    assert plan.actions[0]["tool"] == "workflow.adsorption_scan"
    assert plan.actions[0]["args"]["sites"] == ["top", "bridge", "hollow"]
    # Routing-critical state is untouched by narration.
    assert plan.execution_mode == "execute"
    assert plan.support_status == "complete"
    assert plan.source == "deterministic-macro"


def test_apply_narration_keeps_rule_supplied_fields():
    plan = PlanIR.from_steps(ADSORPTION_MESSAGE, DECIDED_STEPS, source="deterministic-macro")
    plan.expected_outputs = ["规则给出的输出项"]
    original_goal = plan.goal
    apply_macro_narration(plan, {
        "reply": "说明文本",
        "goal": "模型改写的目标",
        "expected_outputs": ["模型改写的输出"],
    })
    assert plan.reply == "说明文本"
    assert plan.goal == original_goal
    assert plan.expected_outputs == ["规则给出的输出项"]


def test_apply_narration_noop_without_narrative():
    plan = PlanIR.from_steps(ADSORPTION_MESSAGE, DECIDED_STEPS, source="deterministic-macro")
    assert apply_macro_narration(plan, None) is False
    assert plan.reply == ""


def test_rendered_plan_gains_explanation_but_keeps_steps():
    plan = PlanIR.from_steps(ADSORPTION_MESSAGE, DECIDED_STEPS, source="deterministic-macro")
    bare = render_plan_reply(plan, DECIDED_STEPS)
    apply_macro_narration(
        plan, narrate_macro_plan(FakeProvider(_good_reply()), ADSORPTION_MESSAGE, DECIDED_STEPS)
    )
    narrated = render_plan_reply(plan, DECIDED_STEPS)
    assert len(narrated) > len(bare)
    assert "workflow.adsorption_scan" in narrated
    assert "规划步骤" in narrated


def test_planner_routes_adsorption_deterministically_with_narration(monkeypatch):
    """End to end through LLMPlanner: routing is deterministic, prose is added."""
    planner = LLMPlanner(registry)
    provider = FakeProvider(_good_reply())
    monkeypatch.setattr(planner, "_get_provider", lambda: provider)

    steps, direct = planner.plan(ADSORPTION_MESSAGE, {})

    assert direct is None
    assert [s["tool"] for s in steps] == ["workflow.adsorption_scan"]
    assert steps[0]["args"]["sites"] == ["top", "bridge", "hollow"]
    plan_ir = planner.last_plan_ir
    assert plan_ir.source == "deterministic-macro"
    assert plan_ir.reply, "deterministic macro should carry a scientific explanation"
    # Exactly one call: narration. No planning round-trip was made.
    assert len(provider.calls) == 1
    assert "不可更改" in provider.calls[0]["system"]


def test_planner_keeps_routing_when_narration_fails(monkeypatch):
    planner = LLMPlanner(registry)
    provider = FakeProvider(RuntimeError("api down"))
    monkeypatch.setattr(planner, "_get_provider", lambda: provider)

    steps, direct = planner.plan(ADSORPTION_MESSAGE, {})

    assert direct is None
    assert [s["tool"] for s in steps] == ["workflow.adsorption_scan"]
    assert steps[0]["args"]["sites"] == ["top", "bridge", "hollow"]
    assert planner.last_plan_ir.source == "deterministic-macro"
    assert planner.last_plan_ir.reply == ""
