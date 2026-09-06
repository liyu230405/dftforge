"""GraphEngine: create/start/pause/resume/retry facade over scheduler + store."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from dft_forge.runtime.graph import GraphTemplate
from dft_forge.runtime.run import GraphRun, NodeRun, create_run
from dft_forge.runtime.scheduler import GraphScheduler, RepairHandler, Tool
from dft_forge.runtime.states import NodeState, RunState, TERMINAL_RUN_STATES
from dft_forge.runtime.store import SQLiteStateStore

logger = logging.getLogger(__name__)


class GraphEngine:
    """Single entry point for graph-based workflows.

    Owns the tool registry, repair handlers, the state store, and template
    registry. ``base_dir`` is the workspace under which each run gets
    ``<run_id>/<node_id>/`` directories for artifacts.
    """

    def __init__(
        self,
        base_dir: Path,
        *,
        store_path: Optional[Path] = None,
        tools: Optional[Dict[str, Tool]] = None,
        repairers: Optional[Dict[str, RepairHandler]] = None,
        max_concurrency: int = 2,
        templates_dir: Optional[Path] = None,
        use_asyncio: bool = False,
    ):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.store = SQLiteStateStore(store_path or (self.base_dir / "graph_state.db"))
        self.tools: Dict[str, Tool] = dict(tools or {})
        self.repairers: Dict[str, RepairHandler] = dict(repairers or {})
        self.max_concurrency = max_concurrency
        self.templates_dir = templates_dir
        self.use_asyncio = use_asyncio
        self._templates: Dict[str, GraphTemplate] = {}

    # ── Tool / template registries ──────────────────────────────────────────

    def register_tool(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def register_template(self, template: GraphTemplate) -> None:
        errors = template.validate(known_tools=set(self.tools))
        if errors:
            raise ValueError(f"invalid template '{template.template_id}': " + "; ".join(errors))
        self._templates[template.template_id] = template

    def get_template(self, template_id: str) -> GraphTemplate:
        if template_id in self._templates:
            return self._templates[template_id]
        if self.templates_dir is not None:
            for suffix in (".json", ".yaml", ".yml"):
                candidate = self.templates_dir / f"{template_id}{suffix}"
                if candidate.exists():
                    from dft_forge.runtime.graph import load_template

                    template = load_template(candidate)
                    self.register_template(template)
                    return template
        raise ValueError(
            f"unknown template '{template_id}'. Registered: {sorted(self._templates)}"
        )

    def list_templates(self) -> List[str]:
        if self.templates_dir is not None:
            for path in sorted(self.templates_dir.glob("*.json")):
                if path.stem not in self._templates:
                    try:
                        from dft_forge.runtime.graph import load_template

                        self.register_template(load_template(path))
                    except Exception:
                        continue
        return sorted(self._templates)

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def create(self, template_id: str, inputs: Optional[Dict[str, Any]] = None) -> GraphRun:
        template = self.get_template(template_id)
        run = create_run(template, inputs)
        self.store.save_run(run)
        return run

    def _cancel_executors(self) -> int:
        """Kill in-flight subprocesses of every tool's executor; returns count.

        Duck-typed: anything on a tool (or on tool.executor) exposing
        cancel_all() participates — LocalExecutor does, FakeExecutor doesn't.
        """
        killed = 0
        for tool in self.tools.values():
            for holder in (tool, getattr(tool, "executor", None)):
                cancel_all = getattr(holder, "cancel_all", None)
                if callable(cancel_all):
                    try:
                        killed += int(cancel_all() or 0)
                    except Exception:
                        logger.exception("cancel_all failed on %s", tool)
        return killed

    def start(
        self, run_id: str, on_event: Optional[Any] = None, cancel_event: Optional[Any] = None
    ) -> GraphRun:
        """Run to a terminal state. Blocks until done (or cancelled)."""
        run = self._load(run_id)
        if run.state in TERMINAL_RUN_STATES:
            return run
        template = self.get_template(run.template_id)
        scheduler = GraphScheduler(
            self.store, self.tools, self.repairers,
            max_concurrency=self.max_concurrency,
            use_asyncio=self.use_asyncio,
            cancel_event=cancel_event,
            on_cancel=self._cancel_executors,
        )
        return scheduler.run(run, template, self.base_dir, on_event=on_event)

    def run_template(
        self,
        template_id: str,
        inputs: Optional[Dict[str, Any]] = None,
        on_event: Optional[Any] = None,
        cancel_event: Optional[Any] = None,
    ) -> GraphRun:
        run = self.create(template_id, inputs)
        return self.start(run.run_id, on_event, cancel_event=cancel_event)

    def resume(self, run_id: str) -> GraphRun:
        """Resume an interrupted run; in-flight nodes are reset to Pending."""
        template = self.get_template(self._peek_template_id(run_id))
        run = self.store.resume_run(run_id, template)
        if run is None:
            raise ValueError(f"unknown run '{run_id}'")
        if run.state in TERMINAL_RUN_STATES:
            return run
        scheduler = GraphScheduler(
            self.store, self.tools, self.repairers,
            max_concurrency=self.max_concurrency,
            use_asyncio=self.use_asyncio,
        )
        return scheduler.run(run, template, self.base_dir)

    def retry(self, run_id: str, node_ids: Optional[List[str]] = None) -> GraphRun:
        """Reset Failed/Blocked nodes to Pending and re-run."""
        run = self._load(run_id)
        targets = node_ids or [
            n.node_id for n in run.nodes.values() if n.state in (NodeState.FAILED, NodeState.BLOCKED)
        ]
        for node_id in targets:
            node = run.nodes[node_id]
            node.transition(NodeState.PENDING)
            node.attempt = 0
            node.repair_attempts = 0
            node.error = None
            node.params = {}  # re-resolve bindings on next dispatch
            node.params_resolved = False
            self.store.save_node_run(run, node)
        run.state = RunState.VALIDATED
        self.store.save_run(run)
        return self.start(run_id)

    def status(self, run_id: str) -> Dict[str, Any]:
        run = self._load(run_id)
        return {
            "run_id": run.run_id,
            "template_id": run.template_id,
            "state": run.state.value,
            "inputs": run.inputs,
            "nodes": {
                nid: {
                    "state": n.state.value,
                    "attempt": n.attempt,
                    "repair_attempts": n.repair_attempts,
                    "error": n.error,
                    "outputs": n.outputs,
                }
                for nid, n in run.nodes.items()
            },
        }

    def list_runs(self) -> List[dict]:
        return self.store.list_runs()

    # ── Internals ────────────────────────────────────────────────────────────

    def _peek_template_id(self, run_id: str) -> str:
        template_id = self.store.template_id_for(run_id)
        if template_id is None:
            raise ValueError(f"unknown run '{run_id}'")
        return template_id

    def _load(self, run_id: str) -> GraphRun:
        template = self.get_template(self._peek_template_id(run_id))
        run = self.store.load_run(run_id, template)
        if run is None:
            raise ValueError(f"unknown run '{run_id}'")
        return run
