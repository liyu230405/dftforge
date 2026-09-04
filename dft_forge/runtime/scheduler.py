"""Dependency-driven scheduler loop with concurrency limit and repair hooks.

Two execution modes drive the SAME state machine:
- threaded (default): ThreadPoolExecutor over blocking node tools — right for
  local QE runs that block for minutes
- asyncio (use_asyncio=True): one task per node via asyncio.to_thread — the
  seam a future asyncssh-based remote executor plugs into
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol

from dft_forge.runtime.graph import GraphTemplate, NodeSpec
from dft_forge.runtime.run import (
    ExecutionContext,
    GraphRun,
    NodeRun,
    ToolError,
    resolve_params,
)
from dft_forge.runtime.states import (
    NodeState,
    RunState,
    TERMINAL_NODE_STATES,
    determine_run_status,
)

logger = logging.getLogger(__name__)


class Tool(Protocol):
    """A node tool: execute one node, return its outputs dict, raise ToolError on failure."""

    name: str

    def execute(self, node: NodeRun, ctx: ExecutionContext) -> Dict[str, Any]: ...


class RepairHandler(Protocol):
    """Attempt to repair a failed node by mutating its params; True = requeue."""

    name: str

    def repair(self, node: NodeRun, error: ToolError) -> bool: ...


def find_ready_nodes(run: GraphRun, template: GraphTemplate) -> List[str]:
    """Pending or Ready nodes whose dependencies all succeeded.

    Ready is included so that repaired/retried nodes (which re-enter Ready)
    get re-dispatched on a later tick.
    """
    ready = []
    for spec in template.nodes:
        node = run.nodes[spec.id]
        if node.state not in (NodeState.PENDING, NodeState.READY):
            continue
        if all(run.nodes[d].state == NodeState.SUCCEEDED for d in spec.depends_on):
            ready.append(spec.id)
    return ready


def process_blocked(run: GraphRun, template: GraphTemplate) -> List[str]:
    """Pending nodes with a dead dependency become Blocked (transitively)."""
    blocked = []
    dead = {NodeState.FAILED, NodeState.CANCELLED, NodeState.BLOCKED, NodeState.SKIPPED}
    for spec in template.nodes:
        node = run.nodes[spec.id]
        if node.state != NodeState.PENDING:
            continue
        if any(run.nodes[d].state in dead for d in spec.depends_on):
            node.transition(NodeState.BLOCKED)
            blocked.append(spec.id)
    return blocked


class GraphScheduler:
    """Drives a GraphRun to a terminal state.

    Each tick: block dead-dependency nodes -> resolve params -> dispatch Ready
    nodes within max_concurrency -> collect finished futures -> apply
    success / retry / repair -> persist. Cancel yields Paused (resumable).
    """

    def __init__(
        self,
        store: Any,
        tools: Dict[str, Tool],
        repairers: Optional[Dict[str, RepairHandler]] = None,
        max_concurrency: int = 2,
        poll_interval: float = 0.05,
        cancel_event: Optional[threading.Event] = None,
        use_asyncio: bool = False,
        on_cancel: Optional[Callable[[], Any]] = None,
    ):
        self.store = store
        self.tools = tools
        self.repairers = repairers or {}
        self.max_concurrency = max(1, max_concurrency)
        self.poll_interval = poll_interval
        self.cancel_event = cancel_event or threading.Event()
        self.use_asyncio = use_asyncio
        # invoked after nodes are marked CANCELLED — the hook that kills the
        # actual subprocesses (executor.cancel_all); without it "cancel" only
        # stops scheduling while pw.x keeps burning CPU until natural exit
        self.on_cancel = on_cancel

    def run(
        self,
        run: GraphRun,
        template: GraphTemplate,
        base_dir: Path,
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> GraphRun:
        if self.use_asyncio:
            coro = self.run_async(run, template, base_dir, on_event)
            try:
                # must not be awaited from a live event loop — callers inside a
                # loop should use run_async() directly
                return asyncio.run(coro)
            except Exception:
                coro.close()  # asyncio.run leaves it unawaited on early failure
                raise
        return self._run_threaded(run, template, base_dir, on_event)

    def _run_threaded(
        self,
        run: GraphRun,
        template: GraphTemplate,
        base_dir: Path,
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> GraphRun:
        run.state = RunState.RUNNING
        self.store.save_run(run)
        ctx_base = Path(base_dir) / run.run_id
        ctx_base.mkdir(parents=True, exist_ok=True)

        def notify(node_id: str, state: str, **extra: Any) -> None:
            if on_event is None:
                return
            try:
                on_event({"type": "node", "node": node_id, "state": state, **extra})
            except Exception:
                pass

        futures: Dict[Future, str] = {}
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as pool:
            while True:
                if self.cancel_event.is_set():
                    self._cancel(run, futures)
                    return run

                process_blocked(run, template)

                ready = find_ready_nodes(run, template)
                slots = self.max_concurrency - len(futures)
                for node_id in ready[:slots]:
                    spec, node, ctx = self._prepare_dispatch(run, template, ctx_base, node_id)
                    notify(node_id, "running", attempt=node.attempt)
                    futures[pool.submit(self._execute_tool, spec, node, ctx)] = node_id

                for fut in [f for f in futures if f.done()]:
                    node_id = futures.pop(fut)
                    self._finish(run, template, node_id, fut.result, notify)

                if not futures and not find_ready_nodes(run, template):
                    states = [n.state for n in run.nodes.values()]
                    if all(s in TERMINAL_NODE_STATES for s in states):
                        run.state = determine_run_status(states)
                        self.store.save_run(run)
                        return run
                    # remaining Pending nodes can only be blocked; loop once more

                time.sleep(self.poll_interval)

    async def run_async(
        self,
        run: GraphRun,
        template: GraphTemplate,
        base_dir: Path,
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> GraphRun:
        """asyncio twin of _run_threaded: same ticks, tasks instead of futures.

        Node tools stay synchronous — each runs in a worker thread via
        asyncio.to_thread, so blocking QE runs never starve the loop.
        """
        run.state = RunState.RUNNING
        self.store.save_run(run)
        ctx_base = Path(base_dir) / run.run_id
        ctx_base.mkdir(parents=True, exist_ok=True)

        def notify(node_id: str, state: str, **extra: Any) -> None:
            if on_event is None:
                return
            try:
                on_event({"type": "node", "node": node_id, "state": state, **extra})
            except Exception:
                pass

        pending: Dict[asyncio.Task, str] = {}
        while True:
            if self.cancel_event.is_set():
                self._cancel_async(run, pending)
                return run

            process_blocked(run, template)

            ready = find_ready_nodes(run, template)
            slots = self.max_concurrency - len(pending)
            for node_id in ready[:slots]:
                spec, node, ctx = self._prepare_dispatch(run, template, ctx_base, node_id)
                notify(node_id, "running", attempt=node.attempt)
                pending[asyncio.create_task(self._execute_tool_async(spec, node, ctx))] = node_id

            if pending:
                done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    node_id = pending.pop(task)
                    self._finish(run, template, node_id, task.result, notify)
            elif not find_ready_nodes(run, template):
                states = [n.state for n in run.nodes.values()]
                if all(s in TERMINAL_NODE_STATES for s in states):
                    run.state = determine_run_status(states)
                    self.store.save_run(run)
                    return run
                # remaining Pending nodes can only be blocked; loop once more

            if not pending:
                await asyncio.sleep(self.poll_interval)

    # ── Shared by both modes ────────────────────────────────────────────────

    def _prepare_dispatch(
        self, run: GraphRun, template: GraphTemplate, ctx_base: Path, node_id: str
    ) -> tuple:
        spec = template.node(node_id)
        node = run.nodes[node_id]
        # Bindings resolve once per run (upstream outputs are immutable once
        # Succeeded); later dispatches keep params so repair-handler mutations
        # survive requeue. The explicit flag (not `if not node.params`) means
        # a node whose params legitimately resolve to {} is not re-resolved
        # on every dispatch.
        if not node.params_resolved:
            node.params = resolve_params(spec.params, run)
            node.params_resolved = True
        node.attempt += 1
        if node.state == NodeState.PENDING:
            node.transition(NodeState.READY)
        node.transition(NodeState.RUNNING)
        node.workdir = str(ctx_base / node_id)
        self.store.save_node_run(run, node)
        ctx = ExecutionContext(
            base_dir=ctx_base,
            run_id=run.run_id,
            upstream={d: run.nodes[d].outputs for d in spec.depends_on},
            timeout_seconds=spec.timeout_seconds,
        )
        return spec, node, ctx

    def _finish(
        self,
        run: GraphRun,
        template: GraphTemplate,
        node_id: str,
        result_getter: Callable[[], Dict[str, Any]],
        notify: Callable[..., None],
    ) -> None:
        node = run.nodes[node_id]
        if node.state in TERMINAL_NODE_STATES:
            # cancelled while this thread was in-flight: the result (killed
            # subprocess / partial output) must not resurrect the node
            return
        spec = template.node(node_id)
        try:
            node.outputs = result_getter()
            node.error = None
            node.transition(NodeState.SUCCEEDED)
            notify(node_id, "succeeded")
        except ToolError as exc:
            self._handle_failure(run, spec, node, exc)
            notify(node_id, node.state.value, error=node.error)
        except Exception as exc:  # tool crashed unexpectedly
            self._handle_failure(run, spec, node, ToolError(str(exc)))
            notify(node_id, node.state.value, error=node.error)
        self.store.save_node_run(run, node)

    def _execute_tool(self, spec: NodeSpec, node: NodeRun, ctx: ExecutionContext) -> Dict[str, Any]:
        tool = self.tools.get(spec.tool)
        if tool is None:
            raise ToolError(f"no tool registered for '{spec.tool}'", category="config")
        return tool.execute(node, ctx)

    async def _execute_tool_async(self, spec: NodeSpec, node: NodeRun, ctx: ExecutionContext) -> Dict[str, Any]:
        return await asyncio.to_thread(self._execute_tool, spec, node, ctx)

    def _cancel_async(self, run: GraphRun, pending: Dict[asyncio.Task, str]) -> None:
        for task in pending:
            task.cancel()
        for node_id in pending.values():
            node = run.nodes[node_id]
            if node.state == NodeState.RUNNING:
                node.transition(NodeState.CANCELLED)
                self.store.save_node_run(run, node)
        if self.on_cancel:
            self._safe_on_cancel()
        run.state = RunState.PAUSED
        self.store.save_run(run)

    def _safe_on_cancel(self) -> None:
        try:
            self.on_cancel()
        except Exception:
            logger.exception("on_cancel hook failed; subprocesses may keep running")

    def _handle_failure(self, run: GraphRun, spec: NodeSpec, node: NodeRun, exc: ToolError) -> None:
        node.error = f"[{exc.category}] {exc}"
        repairer = self._repairer_for(spec)
        if exc.repairable and repairer is not None and node.repair_attempts < spec.max_repair_attempts:
            node.transition(NodeState.REPAIRING)
            self.store.save_node_run(run, node)
            node.repair_attempts += 1
            try:
                repaired = repairer.repair(node, exc)
            except Exception as repair_exc:
                node.error += f" | repair crashed: {repair_exc}"
                repaired = False
            node.transition(NodeState.READY if repaired else NodeState.FAILED)
            return
        if node.attempt < spec.max_attempts:
            node.transition(NodeState.READY)  # plain retry
            return
        node.transition(NodeState.FAILED)

    def _repairer_for(self, spec: NodeSpec) -> Optional[RepairHandler]:
        if spec.repair == "":
            return None  # explicitly disabled
        if spec.repair is not None:
            return self.repairers.get(spec.repair)
        return next(iter(self.repairers.values())) if self.repairers else None

    def _cancel(self, run: GraphRun, futures: Dict[Future, str]) -> None:
        for fut in futures:
            fut.cancel()
        for node_id in futures.values():
            node = run.nodes[node_id]
            if node.state == NodeState.RUNNING:
                node.transition(NodeState.CANCELLED)
                self.store.save_node_run(run, node)
        if self.on_cancel:
            self._safe_on_cancel()
        run.state = RunState.PAUSED
        self.store.save_run(run)
