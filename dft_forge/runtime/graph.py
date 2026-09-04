"""Declarative graph templates and DAG validation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

WHITE, GRAY, BLACK = 0, 1, 2

# template-level output contract: "${nodes.<id>.outputs.<key>}"
OUTPUT_REF_RE = re.compile(r"^\$\{nodes\.([A-Za-z0-9_]+)\.outputs\.([A-Za-z0-9_]+)\}$")
PARAM_REF_RE = re.compile(
    r"^\$\{(inputs|nodes)\.([A-Za-z0-9_]+)(?:\.outputs\.([A-Za-z0-9_]+))?\}$"
)
NODE_ID_RE = re.compile(r"^[A-Za-z0-9_]+$")


def _iter_param_strings(value: Any):
    if isinstance(value, str):
        yield value.strip()
    elif isinstance(value, dict):
        for nested in value.values():
            yield from _iter_param_strings(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _iter_param_strings(nested)


@dataclass
class NodeSpec:
    """Declarative node definition inside a GraphTemplate."""

    id: str
    tool: str
    params: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    max_attempts: int = 1
    max_repair_attempts: int = 2
    repair: Optional[str] = None  # repair handler name; None = default, "" = disabled
    timeout_seconds: Optional[float] = None
    execution_mode: str = "local"  # "local" | "hpc"


@dataclass
class GraphTemplate:
    """A validated DAG of NodeSpecs plus template-level inputs/outputs."""

    template_id: str
    description: str = ""
    nodes: List[NodeSpec] = field(default_factory=list)
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, str] = field(default_factory=dict)  # name -> "${nodes.x.outputs.y}"

    def node(self, node_id: str) -> Optional[NodeSpec]:
        return next((n for n in self.nodes if n.id == node_id), None)

    def validate(self, known_tools: Optional[Set[str]] = None) -> List[str]:
        """Return a list of validation errors (empty = valid).

        Checks structural constraints and parameter references before a run is
        created, so malformed templates fail at load time instead of halfway
        through an expensive calculation.
        """
        errors: List[str] = []
        if not self.template_id.strip():
            errors.append("template_id must be non-empty")

        ids = [n.id for n in self.nodes]
        if not ids:
            errors.append("template has no nodes")
        if len(set(ids)) != len(ids):
            errors.append("duplicate node ids")
        if any(not n.id.strip() for n in self.nodes):
            errors.append("node id must be non-empty")

        known = set(ids)
        for n in self.nodes:
            if n.id and not NODE_ID_RE.fullmatch(n.id):
                errors.append(f"node '{n.id}' contains unsupported characters")
            if not n.tool.strip():
                errors.append(f"node '{n.id}' tool must be non-empty")
            if n.max_attempts < 1:
                errors.append(f"node '{n.id}' max_attempts must be >= 1")
            if n.max_repair_attempts < 0:
                errors.append(f"node '{n.id}' max_repair_attempts must be >= 0")
            if n.timeout_seconds is not None and n.timeout_seconds <= 0:
                errors.append(f"node '{n.id}' timeout_seconds must be > 0")
            if n.execution_mode not in {"local", "hpc"}:
                errors.append(
                    f"node '{n.id}' execution_mode must be 'local' or 'hpc', got {n.execution_mode!r}"
                )
            if len(set(n.depends_on)) != len(n.depends_on):
                errors.append(f"node '{n.id}' has duplicate dependencies")
            for dep in n.depends_on:
                if dep not in known:
                    errors.append(f"node '{n.id}' depends on unknown node '{dep}'")
            if n.id in n.depends_on:
                errors.append(f"node '{n.id}' depends on itself")

            for raw in _iter_param_strings(n.params):
                if not raw.startswith("${"):
                    continue
                match = PARAM_REF_RE.fullmatch(raw)
                if match is None:
                    errors.append(f"node '{n.id}' has malformed parameter reference: {raw!r}")
                    continue
                scope, key, output_key = match.groups()
                if scope == "inputs" and (output_key is not None or key not in self.inputs):
                    errors.append(f"node '{n.id}' references unknown input '{key}'")
                elif scope == "nodes":
                    if output_key is None:
                        errors.append(f"node '{n.id}' node reference needs an outputs key: {raw!r}")
                    elif key not in known:
                        errors.append(f"node '{n.id}' references unknown node '{key}'")
                    elif key not in n.depends_on:
                        errors.append(
                            f"node '{n.id}' references '{key}' without declaring it in depends_on"
                        )

        cycle = self._detect_cycle()
        if cycle:
            errors.append("cycle detected: " + " -> ".join(cycle))

        for name, ref in self.outputs.items():
            if not isinstance(ref, str) or not OUTPUT_REF_RE.match(ref):
                errors.append(
                    f"template output '{name}' must be a '${{nodes.<id>.outputs.<key>}}' reference, got: {ref!r}"
                )
                continue
            node_id = OUTPUT_REF_RE.match(ref).group(1)
            if node_id not in ids:
                errors.append(f"template output '{name}' references unknown node '{node_id}'")

        if known_tools is not None:
            for n in self.nodes:
                if n.tool not in known_tools:
                    errors.append(f"node '{n.id}' uses unknown tool '{n.tool}'")
        return errors

    def _detect_cycle(self) -> Optional[List[str]]:
        adj: Dict[str, List[str]] = {n.id: list(n.depends_on) for n in self.nodes}
        color = {n.id: WHITE for n in self.nodes}
        path: List[str] = []

        def dfs(node_id: str) -> Optional[List[str]]:
            if color[node_id] == GRAY:
                start = path.index(node_id)
                return path[start:] + [node_id]
            if color[node_id] == BLACK:
                return None
            color[node_id] = GRAY
            path.append(node_id)
            for dep in adj.get(node_id, []):
                if dep in color:
                    found = dfs(dep)
                    if found:
                        return found
            path.pop()
            color[node_id] = BLACK
            return None

        for n in self.nodes:
            if color[n.id] == WHITE:
                found = dfs(n.id)
                if found:
                    return found
        return None

    # ── Serialization ────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "template_id": self.template_id,
            "description": self.description,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "nodes": [
                {
                    "id": n.id,
                    "tool": n.tool,
                    "params": n.params,
                    "depends_on": n.depends_on,
                    "max_attempts": n.max_attempts,
                    "max_repair_attempts": n.max_repair_attempts,
                    **({"repair": n.repair} if n.repair is not None else {}),
                    **({"timeout_seconds": n.timeout_seconds} if n.timeout_seconds else {}),
                    **({"execution_mode": n.execution_mode} if n.execution_mode != "local" else {}),
                }
                for n in self.nodes
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphTemplate":
        nodes = [
            NodeSpec(
                id=str(n["id"]),
                tool=str(n["tool"]),
                params=dict(n.get("params", {})),
                depends_on=[str(d) for d in n.get("depends_on", [])],
                max_attempts=int(n.get("max_attempts", 1)),
                max_repair_attempts=int(n.get("max_repair_attempts", 2)),
                repair=n.get("repair"),
                timeout_seconds=n.get("timeout_seconds"),
                execution_mode=str(n.get("execution_mode", "local")),
            )
            for n in data.get("nodes", [])
        ]
        return cls(
            template_id=str(data["template_id"]),
            description=str(data.get("description", "")),
            nodes=nodes,
            inputs=dict(data.get("inputs", {})),
            outputs=dict(data.get("outputs", {})),
        )


def load_template(path: Path) -> GraphTemplate:
    """Load a template from .json (or .yaml when PyYAML is available)."""
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        import yaml  # optional dependency

        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    return GraphTemplate.from_dict(data)
