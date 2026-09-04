"""Task: LLM output schema validation (step args checked against tool schemas)."""

from __future__ import annotations

import pytest

from dft_forge.agent_loop import AgentLoop
from dft_forge.agent_loop.llm_planner import LLMPlanner
from dft_forge.agent_loop.step_validator import validate_steps
from dft_forge.tools.definitions import register_default_tools
from dft_forge.tools.registry import registry as reg


def _ensure_registered():
    if not reg._tools:
        register_default_tools()


@pytest.fixture(scope="module", autouse=True)
def _register():
    _ensure_registered()


class TestValidateSteps:
    def test_valid_steps_pass(self):
        steps = [
            {"tool": "graph.run", "args": {"template_id": "t2_bands", "inputs": {"material": "Si"}}},
            {"tool": "structure.build2d", "args": {"kind": "graphene", "supercell": "3x3"}},
        ]
        assert validate_steps(steps, reg) == []

    def test_missing_required_field_reported(self):
        steps = [{"tool": "graph.run", "args": {"inputs": {"material": "Si"}}}]
        errors = validate_steps(steps, reg)
        assert any("graph.run" in e and "template_id" in e for e in errors)

    def test_wrong_type_reported(self):
        steps = [{"tool": "structure.build2d", "args": {"kind": "graphene", "vacuum": "big"}}]
        errors = validate_steps(steps, reg)
        assert any("vacuum" in e for e in errors)

    def test_enum_violation_reported(self):
        steps = [{"tool": "structure.build2d", "args": {"kind": "unobtanium"}}]
        errors = validate_steps(steps, reg)
        assert any("kind" in e for e in errors)

    def test_args_not_object_reported(self):
        steps = [{"tool": "structure.build2d", "args": "graphene"}]
        errors = validate_steps(steps, reg)
        assert any("must be an object" in e for e in errors)

    def test_unknown_tool_skipped(self):
        steps = [{"tool": "not.a.tool", "args": {"whatever": 1}}]
        assert validate_steps(steps, reg) == []

    def test_null_args_treated_as_empty(self):
        steps = [{"tool": "graph.templates", "args": None}]
        assert validate_steps(steps, reg) == []


class _ScriptedProvider:
    """Returns queued raw LLM answers in order."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = 0
        self.prompts_seen = []

    def chat(self, system: str, user: str) -> str:
        self.calls += 1
        self.prompts_seen.append(user)
        return self.answers.pop(0)


class TestPlannerSchemaRetry:
    def test_bad_args_trigger_repair_retry_then_pass(self):
        provider = _ScriptedProvider([
            # attempt 1: vacuum is a string (schema says number)
            '{"reply": "ok", "steps": [{"tool": "structure.build2d", '
            '"args": {"kind": "graphene", "vacuum": "15"}, "description": "d"}]}',
            # attempt 2: fixed
            '{"reply": "ok", "steps": [{"tool": "structure.build2d", '
            '"args": {"kind": "graphene", "vacuum": 15.0}, "description": "d"}]}',
        ])
        planner = LLMPlanner(reg)
        planner._provider = provider
        planner._checked = True
        steps, _reply = planner.plan("构建石墨烯单层", {})
        assert provider.calls == 2
        assert steps[0]["args"]["vacuum"] == 15.0
        # the retry prompt must tell the LLM what was wrong
        assert "invalid args" in provider.prompts_seen[1]

    def test_persistent_bad_args_pass_through_loudly(self, caplog):
        provider = _ScriptedProvider([
            '{"reply": "ok", "steps": [{"tool": "structure.build2d", '
            '"args": {"kind": "graphene", "vacuum": "15"}, "description": "d"}]}',
            '{"reply": "ok", "steps": [{"tool": "structure.build2d", '
            '"args": {"kind": "graphene", "vacuum": "still bad"}, "description": "d"}]}',
        ])
        planner = LLMPlanner(reg)
        planner._provider = provider
        planner._checked = True
        with caplog.at_level("WARNING"):
            steps, _reply = planner.plan("构建石墨烯单层", {})
        # not rejected: boundary coercion handles shape variance
        assert steps is not None and steps[0]["tool"] == "structure.build2d"
        assert any("still invalid" in r.message for r in caplog.records)

    def test_valid_first_answer_needs_no_retry(self):
        provider = _ScriptedProvider([
            '{"reply": "ok", "steps": [{"tool": "graph.run", '
            '"args": {"template_id": "t1_vc_relax", "inputs": {"material": "Si"}}, '
            '"description": "d"}]}',
        ])
        planner = LLMPlanner(reg)
        planner._provider = provider
        planner._checked = True
        steps, _ = planner.plan("算 Si 的结构优化", {})
        assert provider.calls == 1
        assert steps[0]["args"]["template_id"] == "t1_vc_relax"


class TestAgentLoopImportsValidator:
    def test_facade_still_works(self):
        loop = AgentLoop()
        assert hasattr(loop, "llm_planner")
