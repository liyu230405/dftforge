"""Tests for the LLM planner and agent modules."""

from pathlib import Path

import pytest

from dft_forge.llm import DummyLLMProvider, LLMRequest, LLMResponse
from dft_forge.planner import Planner
from dft_forge.agent import Agent, SolveResult
from dft_forge.ledger import EvidenceLedger


class TestDummyLLMProvider:
    def test_plan_vcrelax(self):
        provider = DummyLLMProvider()
        response = provider.plan(LLMRequest(prompt="Run a vc-relax on Si"))
        assert response.workflow is not None
        assert response.workflow.task_type == "T1"
        assert response.workflow.material == "Si"

    def test_plan_bands(self):
        provider = DummyLLMProvider()
        response = provider.plan(LLMRequest(prompt="Compute band structure for Al"))
        assert response.workflow.task_type == "T2"
        assert response.workflow.material == "Al"
        assert len(response.workflow.steps) == 3

    def test_plan_dos(self):
        provider = DummyLLMProvider()
        response = provider.plan(LLMRequest(prompt="Compute DOS for MgO"))
        assert response.workflow.task_type == "T2"
        assert response.workflow.material == "MgO"
        assert len(response.workflow.steps) == 3


class TestPlanner:
    def test_plan_from_prompt(self):
        planner = Planner()
        workflow = planner.plan_from_prompt("vc-relax Si", available_materials=["Si", "Al"])
        assert workflow.task_id.startswith("Dummy_T1_Si")
        assert workflow.validate() == []

    def test_plan_recovery_patch(self):
        planner = Planner()
        patch = planner.plan_recovery_patch(
            None, "step1", "increase_nbnd", new_params={"nbnd": 20}
        )
        assert patch["action"] == "increase_nbnd"
        assert patch["new_params"]["nbnd"] == 20

    def test_invalid_recovery_action_raises(self):
        planner = Planner()
        with pytest.raises(ValueError):
            planner.plan_recovery_patch(None, "step1", "delete_files")


class TestAgent:
    def test_solve_known_task(self, tmp_path):
        ledger = EvidenceLedger(tmp_path / "ledger.db")
        agent = Agent(ledger=ledger)
        result = agent.solve_from_prompt("vc-relax Si", tmp_path, task_id="T1_Si_vcrelax")
        assert result.task_id == "T1_Si_vcrelax"
        assert result.ledger_rows == 1
        assert result.result is not None

    def test_solve_writes_output(self, tmp_path):
        agent = Agent()
        out_file = tmp_path / "result.json"
        result = agent.solve_from_prompt(
            "vc-relax Si", tmp_path, out=out_file, task_id="T1_Si_vcrelax"
        )
        assert out_file.exists()
        assert result.status in {"pass", "fail", "incomplete", "unsupported"}

    def test_solve_unsupported_workflow(self, tmp_path):
        agent = Agent()
        result = agent.solve_from_prompt("unknown task type", tmp_path)
        assert result.status == "unsupported"
