"""Dependency-driven scheduler loop with concurrency limit and repair hooks."""

from __future__ import annotations

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
    dead = {NodeState.FAILED, NodeState.CANCELLED, NodeState.BLOCKED}
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
    ):
        self.store = store
        self.tools = tools
        self.repairers = repairers or {}
        self.max_concurrency = max(1, max_concurrency)
        self.poll_interval = poll_interval
        self.cancel_event = cancel_event or threading.Event()

    def run(self, run: GraphRun, template: GraphTemplate, base_dir: Path) -> GraphRun:
        run.state = RunState.RUNNING
        self.store.save_run(run)
        ctx_base = Path(base_dir) / run.run_id
        ctx_base.mkdir(parents=True, exist_ok=True)

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
                    spec = template.node(node_id)
                    node = run.nodes[node_id]
                    # Bindings resolve once per run (upstream outputs are
                    # immutable once Succeeded); later dispatches keep params
                    # so repair-handler mutations survive requeue.
                    if not node.params:
                        node.params = resolve_params(spec.params, run)
                    node.attempt += 1
                    node.state = NodeState.RUNNING
                    node.workdir = str(ctx_base / node_id)
                    self.store.save_node_run(run, node)
                    ctx = ExecutionContext(
                        base_dir=ctx_base,
                        run_id=run.run_id,
                        upstream={d: run.nodes[d].outputs for d in spec.depends_on},
                    )
                    futures[pool.submit(self._execute_tool, spec, node, ctx)] = node_id

                for fut in [f for f in futures if f.done()]:
                    node_id = futures.pop(fut)
                    node = run.nodes[node_id]
                    spec = template.node(node_id)
                    try:
                        node.outputs = fut.result()
                        node.error = None
                        node.state = NodeState.SUCCEEDED
                    except ToolError as exc:
                        self._handle_failure(run, spec, node, exc)
                    except Exception as exc:  # tool crashed unexpectedly
                        self._handle_failure(run, spec, node, ToolError(str(exc)))
                    self.store.save_node_run(run, node)

                if not futures and not find_ready_nodes(run, template):
                    states = [n.state for n in run.nodes.values()]
                    if all(s in TERMINAL_NODE_STATES for s in states):
                        run.state = determine_run_status(states)
                        self.store.save_run(run)
                        return run
                    # remaining Pending nodes can only be blocked; loop once more

                time.sleep(self.poll_interval)

    def _execute_tool(self, spec: NodeSpec, node: NodeRun, ctx: ExecutionContext) -> Dict[str, Any]:
        tool = self.tools.get(spec.tool)
        if tool is None:
            raise ToolError(f"no tool registered for '{spec.tool}'", category="config")
        return tool.execute(node, ctx)

    def _handle_failure(self, run: GraphRun, spec: NodeSpec, node: NodeRun, exc: ToolError) -> None:
        node.error = f"[{exc.category}] {exc}"
        repairer = self._repairer_for(spec)
        if exc.repairable and repairer is not None and node.repair_attempts < spec.max_repair_attempts:
            node.state = NodeState.REPAIRING
            self.store.save_node_run(run, node)
            node.repair_attempts += 1
            try:
                repaired = repairer.repair(node, exc)
            except Exception as repair_exc:
                node.error += f" | repair crashed: {repair_exc}"
                repaired = False
            node.state = NodeState.READY if repaired else NodeState.FAILED
            return
        if node.attempt < spec.max_attempts:
            node.state = NodeState.READY  # plain retry
            return
        node.state = NodeState.FAILED

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
                node.state = NodeState.CANCELLED
                self.store.save_node_run(run, node)
        run.state = RunState.PAUSED
        self.store.save_run(run)
