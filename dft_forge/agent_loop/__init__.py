"""Agent loop backend for chat-driven multi-step tool execution.

Package layout (each module owns one concern):
- helpers.py       constants + pure helpers (formula extraction, viewers, charts)
- intent.py        message → IntentSignals (all keyword/entity detection)
- rule_planner.py  signals → deterministic steps (LLM fallback)
- llm_planner.py   LLM planner + prompt engineering
- step_prep.py     per-step guards + argument injection
- result_views.py  tool output → reply text / viewer / chart / ledger
- narration.py     LLM narration + deterministic compare block
- executor.py      the async execution loop

AgentLoop remains the single public facade — external callers are unaffected.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from dft_forge.tools.definitions import register_default_tools
from dft_forge.tools.registry import registry

from dft_forge.agent_loop.executor import run_session
from dft_forge.agent_loop.helpers import (  # noqa: F401  (re-exported for callers)
    _2D_HOST_ELEMENTS,
    _2D_KIND_NAMES,
    _ELEMENTS,
    _ZH_MATERIAL,
    _dedupe_node_log,
    _extract_charts,
    _load_env_file,
    _viewer_payload_for_material,
    _viewer_payload_from_file,
    extract_formula,
)
from dft_forge.agent_loop.llm_planner import LLMPlanner  # noqa: F401
from dft_forge.agent_loop.rule_planner import RulePlanner

register_default_tools()


class AgentLoop:
    """Planner + executor over the tool registry.

    Tries the LLM planner first (if configured via DFT_FORGE_LLM_* env);
    falls back to deterministic rules. Keeps light session context so
    follow-up messages can reuse the last structure, input file, and job id.
    """

    def __init__(self, registry_obj=None):
        self.registry = registry_obj or registry
        _load_env_file(Path(__file__).resolve().parents[2] / ".env")
        self.llm_planner = LLMPlanner(self.registry)
        self.rule_planner = RulePlanner()

    def list_tools(self) -> List[Dict[str, Any]]:
        return [t.to_dict() for t in self.registry.list_all()]

    def plan(self, message: str, ctx: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        return self.rule_planner.plan(message, ctx)

    async def run(
        self,
        message: str,
        session_dir: Path,
        ctx: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, Any]]] = None,
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        # pass self, not self.rule_planner: run() must consult self.plan so
        # subclasses/monkey-patches of the facade keep working
        return await run_session(
            message, session_dir, ctx, history, on_event,
            registry=self.registry,
            llm_planner=self.llm_planner,
            rule_planner=self,
        )

    def reset_llm(self) -> None:
        """Forget cached LLM provider so new env/config takes effect immediately."""
        self.llm_planner._provider = None
        self.llm_planner._checked = False
