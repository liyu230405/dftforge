"""Tests for the OpenAI-compatible LLM provider (no network)."""

import json
import os
from unittest import mock

import pytest

from dft_forge.llm import (
    DummyLLMProvider,
    LLMRequest,
    OpenAICompatProvider,
    _extract_json,
    get_llm_provider,
)


VALID_T1 = json.dumps(
    {
        "task_id": "si_vc_relax",
        "task_type": "T1",
        "material": "Si",
        "reasoning": "structure optimization",
        "steps": [
            {"step_id": "vc-relax", "step_type": "vc-relax", "prefix": "si", "depends_on": []}
        ],
    }
)

VALID_T2 = json.dumps(
    {
        "task_id": "si_bands",
        "task_type": "T2",
        "material": "Si",
        "reasoning": "band structure",
        "steps": [
            {"step_id": "scf", "step_type": "scf", "prefix": "si", "depends_on": []},
            {"step_id": "nscf", "step_type": "bands_nscf", "prefix": "si", "depends_on": ["scf"]},
        ],
    }
)


def _request(prompt="optimize Si structure"):
    return LLMRequest(prompt=prompt, available_materials=["Si", "Al", "MgO"])


class TestExtractJson:
    def test_plain_json(self):
        assert _extract_json(VALID_T1)["task_type"] == "T1"

    def test_fenced_json(self):
        fenced = f"```json\n{VALID_T2}\n```"
        assert _extract_json(fenced)["task_type"] == "T2"

    def test_json_with_prose(self):
        text = f"Here is the plan:\n{VALID_T1}\nHope this helps."
        assert _extract_json(text)["material"] == "Si"

    def test_no_json_raises(self):
        with pytest.raises(ValueError):
            _extract_json("sorry, I cannot do that")


class TestProviderSelection:
    def test_default_is_dummy(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            assert isinstance(get_llm_provider(), DummyLLMProvider)

    def test_openai_alias(self):
        env = {"DFT_FORGE_LLM_PROVIDER": "openai", "DFT_FORGE_LLM_API_KEY": "test"}
        with mock.patch.dict(os.environ, env, clear=True):
            provider = get_llm_provider()
        assert isinstance(provider, OpenAICompatProvider)
        assert provider.api_key == "test"

    def test_unknown_provider_raises(self):
        with mock.patch.dict(os.environ, {"DFT_FORGE_LLM_PROVIDER": "nope"}, clear=True):
            with pytest.raises(ValueError):
                get_llm_provider()


class TestOpenAICompatPlan:
    def test_valid_workflow_returned(self):
        provider = OpenAICompatProvider(api_key="k")
        with mock.patch.object(provider, "_chat", return_value=VALID_T1):
            response = provider.plan(_request())
        assert response.error is None
        assert response.workflow.task_type == "T1"
        assert response.workflow.steps[0].step_type == "vc-relax"
        assert response.workflow.validate() == []

    def test_invalid_then_repair(self):
        provider = OpenAICompatProvider(api_key="k", max_attempts=2)
        with mock.patch.object(provider, "_chat", side_effect=["not json", VALID_T2]):
            response = provider.plan(_request("compute Si band structure"))
        assert response.error is None
        assert response.workflow.task_type == "T2"

    def test_persistent_failure_returns_error(self):
        provider = OpenAICompatProvider(api_key="k", max_attempts=2)
        with mock.patch.object(provider, "_chat", return_value="garbage"):
            response = provider.plan(_request())
        assert response.workflow is None
        assert "failed" in response.error

    def test_bad_task_type_derived_from_steps(self):
        data = json.loads(VALID_T2)
        data["task_type"] = "T9"
        provider = OpenAICompatProvider(api_key="k")
        with mock.patch.object(provider, "_chat", return_value=json.dumps(data)):
            response = provider.plan(_request())
        assert response.error is None
        assert response.workflow.task_type == "T2"

    def test_user_message_lists_materials(self):
        provider = OpenAICompatProvider(api_key="k")
        message = provider._build_user_message(_request())
        assert "Si" in message and "MgO" in message
