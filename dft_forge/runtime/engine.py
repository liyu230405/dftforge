"""GraphEngine: create/start/pause/resume/retry facade over scheduler + store."""

from __future__ import annotations

import logging
import copy
import json
import time
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
        execution_identity: Optional[str] = None,
        execution_profile: Optional[Dict[str, Any]] = None,
    ):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.store = SQLiteStateStore(store_path or (self.base_dir / "graph_state.db"))
        self.tools: Dict[str, Tool] = dict(tools or {})
        self.repairers: Dict[str, RepairHandler] = dict(repairers or {})
        self.max_concurrency = max_concurrency
        self.templates_dir = templates_dir
        self.use_asyncio = use_asyncio
        self.execution_identity = execution_identity
        self.execution_profile = copy.deepcopy(execution_profile) if execution_profile else None
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
            from dft_forge.runtime.graph import load_named_template

            try:
                template = load_named_template(self.templates_dir, template_id)
            except FileNotFoundError:
                template = None
            if template is not None:
                self.register_template(template)
                return template
        raise ValueError(
            f"unknown template '{template_id}'. Registered: {sorted(self._templates)}"
        )

    def list_templates(self) -> List[str]:
        if self.templates_dir is not None:
            from dft_forge.runtime.graph import TEMPLATE_SUFFIXES, load_named_template

            root = Path(self.templates_dir)
            if root.is_dir():
                ids = {
                    path.name[: -len(path.suffix)]
                    for path in root.iterdir()
                    if path.is_file() and path.suffix in TEMPLATE_SUFFIXES
                }
                for template_id in sorted(ids):
                    if template_id in self._templates:
                        continue
                    try:
                        self.register_template(load_named_template(root, template_id))
                    except Exception:
                        continue
        return sorted(self._templates)

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def create(self, template_id: str, inputs: Optional[Dict[str, Any]] = None, *, run_metadata: Optional[Dict[str, Any]] = None) -> GraphRun:
        template = self.get_template(template_id)
        # Caller metadata is opaque JSON. Runtime-owned provenance can never
        # be overridden, and all metadata exists in the very first run write.
        if run_metadata is not None and (not isinstance(run_metadata, dict) or "execution" in run_metadata):
            raise ValueError("run_metadata must be a mapping without reserved key 'execution'")
        try:
            metadata = json.loads(json.dumps(run_metadata or {}, allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise ValueError("run_metadata must contain finite JSON values") from exc
        run = create_run(template, inputs)
        run.metadata.update(metadata)
        if self.execution_identity is not None or self.execution_profile is not None:
            run.metadata["execution"] = {
                "origin": {
                    "identity": self.execution_identity,
                    "profile": copy.deepcopy(self.execution_profile or {}),
                },
                "origin_unknown": False,
                "attempts": [],
            }
        self.store.save_run(run, template)
        for node in run.nodes.values():
            self.store.save_node_run(run, node)
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
        template = self._template_for_run(run.run_id)
        run = self.validate_execution_identity(run_id)
        return self._dispatch(run, template, "start", on_event, cancel_event)

    def _dispatch(
        self,
        run: GraphRun,
        template: GraphTemplate,
        operation: str,
        on_event: Optional[Any] = None,
        cancel_event: Optional[Any] = None,
    ) -> GraphRun:
        self._record_attempt(run, operation)
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
        *,
        run_metadata: Optional[Dict[str, Any]] = None,
    ) -> GraphRun:
        run = self.create(template_id, inputs, run_metadata=run_metadata)
        return self.start(run.run_id, on_event, cancel_event=cancel_event)

    def resume(
        self,
        run_id: str,
        on_event: Optional[Any] = None,
        cancel_event: Optional[Any] = None,
    ) -> GraphRun:
        """Resume an interrupted run; in-flight nodes are reset to Pending."""
        run = self._load(run_id)
        if run.state in TERMINAL_RUN_STATES:
            return run
        run = self.validate_execution_identity(run_id)
        template = self._template_for_run(run_id)
        run = self.store.resume_run(run_id, template)
        if run is None:
            raise ValueError(f"unknown run '{run_id}'")
        if run.state in TERMINAL_RUN_STATES:
            return run
        return self._dispatch(run, template, "resume", on_event, cancel_event)

    def retry(
        self,
        run_id: str,
        node_ids: Optional[List[str]] = None,
        on_event: Optional[Any] = None,
        cancel_event: Optional[Any] = None,
    ) -> GraphRun:
        """Reset Failed/Blocked nodes to Pending and re-run."""
        run = self._load(run_id)
        template = self._template_for_run(run_id)
        run = self.validate_execution_identity(run_id)
        if run.state not in (RunState.FAILED, RunState.PARTIALLY_SUCCEEDED):
            raise ValueError(f"run '{run_id}' is not retryable in state {run.state.value}")
        targets = (
            [n.node_id for n in run.nodes.values()
             if n.state in (NodeState.FAILED, NodeState.BLOCKED)]
            if node_ids is None else list(node_ids)
        )
        if not targets:
            raise ValueError("retry requires at least one failed or blocked node")
        if len(set(targets)) != len(targets):
            raise ValueError("retry node_ids must be unique")
        unknown = [node_id for node_id in targets if node_id not in run.nodes]
        if unknown:
            raise ValueError(f"unknown retry node(s): {unknown}")
        invalid = [
            node_id for node_id in targets
            if run.nodes[node_id].state not in (NodeState.FAILED, NodeState.BLOCKED)
        ]
        if invalid:
            raise ValueError(f"retry targets must be failed or blocked: {invalid}")

        # Include blocked descendants that become runnable once the selected
        # failed nodes recover.  Never requeue a blocked node while one of its
        # dead dependencies is left outside the retry set.
        targets_set = set(targets)
        changed = True
        while changed:
            changed = False
            for node_id, node in run.nodes.items():
                if node.state != NodeState.BLOCKED or node_id in targets_set:
                    continue
                spec = template.node(node_id)
                dead_deps = [
                    dep for dep in (spec.depends_on if spec else [])
                    if run.nodes[dep].state in (
                        NodeState.FAILED,
                        NodeState.BLOCKED,
                        NodeState.CANCELLED,
                        NodeState.SKIPPED,
                    )
                ]
                if dead_deps and all(dep in targets_set for dep in dead_deps):
                    targets_set.add(node_id)
                    changed = True
        for node_id in list(targets_set):
            node = run.nodes[node_id]
            spec = template.node(node_id)
            dead_deps = [
                dep for dep in (spec.depends_on if spec else [])
                if run.nodes[dep].state in (
                    NodeState.FAILED,
                    NodeState.BLOCKED,
                    NodeState.CANCELLED,
                    NodeState.SKIPPED,
                )
            ]
            if any(dep not in targets_set for dep in dead_deps):
                raise ValueError(
                    f"cannot retry '{node_id}' while failed dependency remains outside retry targets"
                )
        targets = list(targets_set)
        for node_id in targets:
            node = run.nodes[node_id]
            node.transition(NodeState.PENDING)
            node.attempt = 0
            node.repair_attempts = 0
            node.error = None
            node.params = {}  # re-resolve bindings on next dispatch
            node.params_resolved = False
            node.outputs = {}
            node.job_id = None
            self.store.save_node_run(run, node)
        run.state = RunState.VALIDATED
        self.store.save_run(run)
        return self._dispatch(run, template, "retry", on_event, cancel_event)

    def status(self, run_id: str) -> Dict[str, Any]:
        run = self._load(run_id)
        return {
            "run_id": run.run_id,
            "template_id": run.template_id,
            "state": run.state.value,
            "inputs": run.inputs,
            "metadata": copy.deepcopy(run.metadata),
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

    def validate_execution_identity(self, run_id: str) -> GraphRun:
        """Validate backend provenance before a recovery mutation.

        Status reads intentionally do not call this method. Legacy runs are
        marked as unknown on their first recovery and remain usable.
        """
        run = self._load(run_id)
        execution = run.metadata.get("execution")
        changed = False
        if not isinstance(execution, dict):
            execution = {"origin_unknown": True, "attempts": []}
            run.metadata["execution"] = execution
            changed = True
        if not isinstance(execution.get("attempts"), list):
            execution["attempts"] = []
            changed = True
        origin = execution.get("origin")
        if not isinstance(origin, dict):
            origin = None
        recorded_identity = origin.get("identity") if origin is not None else None
        if recorded_identity:
            if recorded_identity != self.execution_identity:
                raise ValueError(
                    f"execution identity mismatch for run '{run_id}': "
                    f"recorded origin cannot be recovered by the current backend"
                )
        elif self.execution_identity is not None:
            if origin is None:
                execution["origin"] = {
                    "identity": self.execution_identity,
                    "profile": copy.deepcopy(self.execution_profile or {}),
                }
                changed = True
        if changed:
            self.store.save_run(run)
        return run

    def _record_attempt(self, run: GraphRun, operation: str) -> None:
        if self.execution_identity is None and self.execution_profile is None and "execution" not in run.metadata:
            return
        execution = run.metadata.setdefault("execution", {})
        execution.setdefault("origin_unknown", False)
        execution.setdefault("attempts", [])
        execution["attempts"].append({
            "operation": operation,
            "profile": copy.deepcopy(self.execution_profile or {}),
            "started_at": time.time(),
        })
        self.store.save_run(run)

    # ── Internals ────────────────────────────────────────────────────────────

    def _peek_template_id(self, run_id: str) -> str:
        template_id = self.store.template_id_for(run_id)
        if template_id is None:
            raise ValueError(f"unknown run '{run_id}'")
        return template_id

    def _template_for_run(self, run_id: str) -> GraphTemplate:
        """Load a run's immutable template, with legacy registry fallback."""
        template_id = self._peek_template_id(run_id)
        snapshot = self.store.template_for(run_id)
        if snapshot is not None:
            if snapshot.template_id != template_id:
                raise ValueError(
                    f"template snapshot mismatch for run '{run_id}': "
                    f"expected '{template_id}', got '{snapshot.template_id}'"
                )
            errors = snapshot.validate(known_tools=set(self.tools))
            if errors:
                raise ValueError(
                    f"invalid template snapshot for run '{run_id}': " + "; ".join(errors)
                )
            return snapshot
        return self.get_template(template_id)

    def _load(self, run_id: str) -> GraphRun:
        template = self._template_for_run(run_id)
        run = self.store.load_run(run_id, template)
        if run is None:
            raise ValueError(f"unknown run '{run_id}'")
        return run
