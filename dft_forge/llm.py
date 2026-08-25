"""LLM provider abstraction for DFT-Forge.

The LLM only decides WHAT to do. It outputs structured ``WorkflowIR`` or
``PlanPatch`` objects, never shell commands or full QE input.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from dft_forge.protocol.schemas import StepIR, WorkflowIR


@dataclass
class LLMRequest:
    """Request payload sent to the LLM provider."""
    prompt: str
    task_spec: Optional[Dict[str, Any]] = None
    available_materials: Optional[Sequence[str]] = None
    available_calcs: Optional[Sequence[str]] = None
    context: Optional[Dict[str, Any]] = None


@dataclass
class LLMResponse:
    """Normalized response from the LLM provider."""
    workflow: Optional[WorkflowIR] = None
    patch: Optional[Any] = None
    text: str = ""
    tokens_used: int = 0
    error: Optional[str] = None


class LLMProvider:
    """Abstract base class for LLM providers."""
    def plan(self, request: LLMRequest) -> LLMResponse:
        raise NotImplementedError


class DummyLLMProvider(LLMProvider):
    """Deterministic dummy provider for offline development and tests.

    Returns a minimal valid ``WorkflowIR`` for known materials; otherwise
    returns an empty response.
    """

    def plan(self, request: LLMRequest) -> LLMResponse:
        prompt = request.prompt.lower()
        available_materials = list(request.available_materials or ["Si", "Al", "MgO"])
        task_type = "T1"
        material = None
        # Longest names first; word-boundary match so "In" does not match
        # "optimization" and "Si" does not match "gaas".
        for candidate in sorted(available_materials, key=len, reverse=True):
            if re.search(
                rf"(?<![a-z0-9]){re.escape(candidate.lower())}(?![a-z0-9])",
                prompt,
            ):
                material = candidate
                break
        if material is None:
            return LLMResponse(error="no known material mentioned in prompt")

        if "band" in prompt:
            task_type = "T2"
            steps = [
                StepIR(step_id="scf", step_type="scf", prefix=material.lower()),
                StepIR(step_id="nscf", step_type="bands_nscf", prefix=material.lower(), depends_on=["scf"]),
            ]
        elif "dos" in prompt:
            task_type = "T2"
            steps = [
                StepIR(step_id="scf", step_type="scf", prefix=material.lower()),
                StepIR(step_id="nscf", step_type="dos_nscf", prefix=material.lower(), depends_on=["scf"]),
            ]
        else:
            steps = [
                StepIR(step_id="vc-relax", step_type="vc-relax", prefix=material.lower()),
            ]

        workflow = WorkflowIR(
            task_id=f"Dummy_{task_type}_{material}",
            task_type=task_type,
            material=material,
            steps=steps,
            reasoning="Dummy plan generated from prompt keywords.",
        )
        return LLMResponse(workflow=workflow, text=str(workflow.to_dict()))


def get_llm_provider() -> LLMProvider:
    """Return the configured LLM provider.

    ``DFT_FORGE_LLM_PROVIDER`` selects the backend:
    - ``dummy`` (default): deterministic keyword matcher, offline
    - ``openai``: any OpenAI-compatible chat API (OpenAI, StepFun, DeepSeek,
      Moonshot, local vLLM/Ollama, ...). Configure via:
        DFT_FORGE_LLM_BASE_URL  e.g. https://api.openai.com/v1
        DFT_FORGE_LLM_API_KEY   the secret key (never commit it)
        DFT_FORGE_LLM_MODEL     e.g. gpt-4o-mini / step-3.7-flash / qwen-plus
    """
    provider = os.environ.get("DFT_FORGE_LLM_PROVIDER", "dummy").lower()
    if provider == "dummy":
        return DummyLLMProvider()
    if provider in ("openai", "openai_compatible", "openai-compatible"):
        return OpenAICompatProvider()
    raise ValueError(f"Unsupported LLM provider: {provider}")


# ── OpenAI-compatible provider ────────────────────────────────────────────────

_WORKFLOW_SYSTEM_PROMPT = """You are the planning module of a DFT computation agent.
Convert the user's request into ONE JSON object describing a Quantum ESPRESSO workflow.

Output rules:
- Reply with ONLY the JSON object, no prose, no markdown fences.
- Schema:
  {
    "task_id": string,           // short ascii id like "si_vc_relax"
    "task_type": "T1" | "T2",    // T1 = vc-relax structure optimization, T2 = bands/DOS
    "material": string,          // must be one of the available materials listed below
    "reasoning": string,         // one short sentence
    "steps": [                   // ordered list, later steps may depend on earlier ones
      {
        "step_id": string,       // ascii, unique, e.g. "scf"
        "step_type": "scf" | "vc-relax" | "relax" | "bands_nscf" | "dos_nscf",
        "prefix": string,        // ascii file prefix, e.g. "si"
        "depends_on": [string],  // step_ids this step needs; [] for the first step
        "overrides": {}          // optional: ecutwfc, kpoints, nbnd, occupations...
      }
    ]
  }
- For T1 use a single vc-relax step. For T2 use scf -> bands_nscf (and/or dos_nscf).
- Never invent shell commands, QE input files, or materials outside the list.
"""


def _extract_json(text: str) -> Dict[str, Any]:
    """Extract the first JSON object from an LLM reply."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object found in LLM reply")
    return json.loads(cleaned[start : end + 1])


class OpenAICompatProvider(LLMProvider):
    """Real LLM provider for any OpenAI-compatible /chat/completions endpoint.

    Uses only the standard library so the package gains no new dependency.
    All secrets come from environment variables; nothing is persisted.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 60.0,
        max_attempts: int = 2,
    ):
        self.base_url = (
            base_url
            or os.environ.get("DFT_FORGE_LLM_BASE_URL", "https://api.openai.com/v1")
        ).rstrip("/")
        self.api_key = api_key or os.environ.get("DFT_FORGE_LLM_API_KEY", "")
        self.model = model or os.environ.get("DFT_FORGE_LLM_MODEL", "gpt-4o-mini")
        self.timeout = timeout
        self.max_attempts = max_attempts

    def _chat(self, system: str, user: str) -> str:
        import urllib.error
        import urllib.request

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise RuntimeError(f"LLM API HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM API unreachable: {exc.reason}") from exc
        try:
            return body["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected LLM API response shape: {body}") from exc

    def _build_user_message(self, request: LLMRequest) -> str:
        materials = list(request.available_materials or ["Si", "Al", "MgO"])
        calcs = list(request.available_calcs or ["scf", "vc-relax", "bands", "dos"])
        parts = [f"User request: {request.prompt}", f"Available materials: {materials}", f"Available calc types: {calcs}"]
        if request.context:
            parts.append(f"Context: {json.dumps(request.context, ensure_ascii=False)}")
        parts.append("Return the JSON workflow now.")
        return "\n".join(parts)

    def chat(self, system: str, user: str) -> str:
        """Public raw chat call for agent planners."""
        return self._chat(system, user)

    def plan(self, request: LLMRequest) -> LLMResponse:
        system = _WORKFLOW_SYSTEM_PROMPT
        user = self._build_user_message(request)
        last_error = ""
        for _ in range(self.max_attempts):
            try:
                raw = self._chat(system, user)
                data = _extract_json(raw)
                workflow = self._to_workflow(data)
                errors = workflow.validate()
                if errors:
                    raise ValueError("; ".join(errors))
                return LLMResponse(workflow=workflow, text=raw)
            except (ValueError, KeyError, TypeError, RuntimeError, json.JSONDecodeError) as exc:
                last_error = str(exc)
                user = (
                    f"Your previous reply was invalid: {last_error}\n"
                    f"Original request: {request.prompt}\n"
                    "Reply again with ONLY a valid JSON workflow object."
                )
        return LLMResponse(error=f"LLM planning failed after {self.max_attempts} attempts: {last_error}")

    @staticmethod
    def _to_workflow(data: Dict[str, Any]) -> WorkflowIR:
        steps = [
            StepIR(
                step_id=str(s["step_id"]),
                step_type=str(s["step_type"]),
                prefix=str(s.get("prefix", "run")),
                depends_on=list(s.get("depends_on", [])),
                overrides=dict(s.get("overrides", {})),
            )
            for s in data.get("steps", [])
        ]
        task_type = data.get("task_type", "")
        if task_type not in ("T1", "T2"):
            task_type = "T2" if any(s.step_type in ("bands_nscf", "dos_nscf") for s in steps) else "T1"
        return WorkflowIR(
            task_id=str(data.get("task_id", "llm_plan")),
            task_type=task_type,
            material=str(data.get("material", "")),
            steps=steps,
            reasoning=str(data.get("reasoning", "")),
        )
