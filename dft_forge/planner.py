"""Scientific planner: converts prompts into structured WorkflowIR / PlanPatch.

The planner is the only place where the LLM influences the computation plan.
All outputs are validated against the protocol schemas before execution.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from dft_forge.llm import LLMProvider, LLMRequest
from dft_forge.protocol.schemas import StepIR, WorkflowIR


class Planner:
    """Deterministic planner wrapper around an LLM provider.

    Responsibilities:
    - Translate a user prompt into a validated ``WorkflowIR``
    - Translate verifier failures into a validated ``PlanPatch``
    - Enforce schema validation and safe-action whitelists
    """

    def __init__(self, provider: Optional[LLMProvider] = None):
        self.provider = provider or _default_provider()

    def plan_from_prompt(
        self,
        prompt: str,
        *,
        task_spec: Optional[Dict[str, Any]] = None,
        available_materials: Optional[Sequence[str]] = None,
        available_calcs: Optional[Sequence[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> WorkflowIR:
        request = LLMRequest(
            prompt=prompt,
            task_spec=task_spec,
            available_materials=available_materials,
            available_calcs=available_calcs,
            context=context,
        )
        response = self.provider.plan(request)

        if response.error:
            raise RuntimeError(f"LLM planning failed: {response.error}")

        if response.workflow is None:
            raise RuntimeError("LLM provider returned no workflow")

        errors = response.workflow.validate()
        if errors:
            raise ValueError(f"Invalid workflow: {errors}")

        return response.workflow

    def plan_recovery_patch(
        self,
        workflow: WorkflowIR,
        target_step_id: str,
        action: str,
        new_params: Optional[Dict[str, Any]] = None,
        reasoning: str = "",
    ) -> Dict[str, Any]:
        """Build a bounded recovery patch.

        This does not call the LLM by default; it packages an allowed
        recovery action into a validated patch structure. The caller is
        responsible for ensuring the action is safe and allowed.
        """
        allowed_actions = {
            "modify_params",
            "change_kpoints",
            "increase_nbnd",
            "retry",
            "restart_from_prev",
        }
        if action not in allowed_actions:
            raise ValueError(f"Unsupported recovery action: {action}")

        patch = {
            "target_step": target_step_id,
            "action": action,
            "new_params": new_params or {},
            "reasoning": reasoning,
        }
        return patch

    def plan_from_file(self, prompt_path: Path) -> WorkflowIR:
        prompt = Path(prompt_path).read_text()
        return self.plan_from_prompt(prompt)


def _default_provider() -> LLMProvider:
    from dft_forge.llm import get_llm_provider
    return get_llm_provider()
