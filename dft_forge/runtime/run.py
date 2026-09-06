"""Mutable graph runs, node runs, param binding resolution, execution context."""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from dft_forge.runtime.graph import GraphTemplate
from dft_forge.runtime.states import NodeState, RunState, can_transition


class ToolError(RuntimeError):
    """Raised by a tool when a node fails.

    `repairable` + `category` feed the repair-handler registry. A matching
    handler may mutate node params and send the node back to Ready instead of
    failing the graph.
    """

    def __init__(self, message: str, *, category: str = "unknown", repairable: bool = False):
        super().__init__(message)
        self.category = category
        self.repairable = repairable


@dataclass
class NodeRun:
    node_id: str
    state: NodeState = NodeState.PENDING
    attempt: int = 0
    repair_attempts: int = 0
    params: Dict[str, Any] = field(default_factory=dict)
    # not persisted: False after load → re-resolve on resume (resolved params
    # carry no ${} refs, so re-resolution is a no-op but keeps resume honest)
    params_resolved: bool = False
    outputs: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    job_id: Optional[str] = None
    workdir: Optional[str] = None
    updated_at: float = field(default_factory=time.time)

    def transition(self, target: NodeState) -> None:
        if not can_transition(self.state, target):
            raise ValueError(f"illegal transition {self.state.value} -> {target.value} for node '{self.node_id}'")
        self.state = target
        self.updated_at = time.time()


@dataclass
class GraphRun:
    run_id: str
    template_id: str
    state: RunState = RunState.CREATED
    nodes: Dict[str, NodeRun] = field(default_factory=dict)
    inputs: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.updated_at = time.time()

    def node_states(self) -> List[NodeState]:
        return [n.state for n in self.nodes.values()]


@dataclass
class ExecutionContext:
    """Everything a tool needs to execute one node."""

    base_dir: Path  # <workspace>/<run_id>
    run_id: str
    upstream: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # node_id -> outputs
    # per-node walltime from NodeSpec.timeout_seconds; None = executor default
    timeout_seconds: Optional[float] = None

    def node_workdir(self, node_id: str) -> Path:
        return self.base_dir / node_id


_REF_RE = re.compile(r"^\$\{(inputs|nodes)\.([A-Za-z0-9_]+)(?:\.outputs\.([A-Za-z0-9_]+))?\}$")


def resolve_value(value: Any, run: GraphRun) -> Any:
    """Resolve ${inputs.x} and ${nodes.y.outputs.z} references."""
    if isinstance(value, str):
        m = _REF_RE.match(value.strip())
        if m:
            scope, key, out_key = m.groups()
            if scope == "inputs":
                if key not in run.inputs:
                    raise ValueError(f"reference to undeclared template input: {value}")
                return run.inputs.get(key)
            if out_key is None:
                raise ValueError(f"node reference without outputs key: {value}")
            upstream = run.nodes.get(key)
            if upstream is None or upstream.state != NodeState.SUCCEEDED:
                raise ValueError(f"referenced node '{key}' has not succeeded")
            if out_key not in upstream.outputs:
                raise ValueError(
                    f"referenced output '{out_key}' not produced by node '{key}' "
                    f"(available: {sorted(upstream.outputs)})"
                )
            return upstream.outputs.get(out_key)
        return value
    if isinstance(value, dict):
        return {k: resolve_value(v, run) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_value(v, run) for v in value]
    return value


def resolve_params(params: Dict[str, Any], run: GraphRun) -> Dict[str, Any]:
    return resolve_value(params, run)


def create_run(template: GraphTemplate, inputs: Optional[Dict[str, Any]] = None) -> GraphRun:
    """Instantiate a GraphRun from a template, merging template input defaults."""
    merged_inputs = dict(template.inputs)
    if inputs:
        unknown = set(inputs) - set(merged_inputs)
        if unknown:
            raise ValueError(f"unknown template inputs: {sorted(unknown)}")
        merged_inputs.update(inputs)
    run = GraphRun(
        run_id=f"run_{uuid.uuid4().hex[:10]}",
        template_id=template.template_id,
        inputs=merged_inputs,
    )
    for spec in template.nodes:
        run.nodes[spec.id] = NodeRun(node_id=spec.id)
    return run
