"""Goal-first planning IR and dynamic capability graph.

The planner should describe *why* a workflow is needed before selecting a
tool.  This module is intentionally small and framework-agnostic: capabilities
are discovered from the live tool registry and graph-template output contracts,
then an IR is compiled into the existing executor step shape.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class PlanIR:
    """Validated, explainable plan produced before execution."""

    goal: str
    constraints: Dict[str, Any] = field(default_factory=dict)
    assumptions: List[str] = field(default_factory=list)
    expected_outputs: List[str] = field(default_factory=list)
    actions: List[Dict[str, Any]] = field(default_factory=list)
    validation: List[str] = field(default_factory=list)
    source: str = "rules"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: Dict[str, Any], *, source: str = "llm") -> "PlanIR":
        actions = payload.get("actions") or payload.get("steps") or []
        normalized: List[Dict[str, Any]] = []
        for action in actions:
            if not isinstance(action, dict):
                continue
            normalized.append({
                "capability": action.get("capability") or action.get("tool"),
                "tool": action.get("tool"),
                "args": action.get("args") or {},
                "description": str(action.get("description") or action.get("reason") or "")[:180],
            })
        return cls(
            goal=str(payload.get("goal") or payload.get("objective") or "").strip()[:240],
            constraints=dict(payload.get("constraints") or {}),
            assumptions=[str(x)[:180] for x in (payload.get("assumptions") or [])[:8]],
            expected_outputs=[str(x)[:120] for x in (payload.get("expected_outputs") or payload.get("outputs") or [])[:12]],
            actions=normalized,
            validation=[str(x)[:160] for x in (payload.get("validation") or payload.get("success_criteria") or [])[:8]],
            source=source,
        )

    @classmethod
    def from_steps(cls, goal: str, steps: Iterable[Dict[str, Any]], *, source: str = "rules") -> "PlanIR":
        normalized = []
        expected: List[str] = []
        for step in steps:
            args = step.get("args") or {}
            normalized.append({
                "capability": step.get("tool"),
                "tool": step.get("tool"),
                "args": args,
                "description": str(step.get("description") or "")[:180],
            })
            template = args.get("template_id")
            if template:
                expected.extend(_TEMPLATE_OUTPUT_LABELS.get(str(template), []))
        return cls(
            goal=goal[:240], actions=normalized,
            expected_outputs=list(dict.fromkeys(expected)), source=source,
        )


@dataclass(frozen=True)
class Capability:
    id: str
    tool_id: str
    category: str
    description: str
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_CAPABILITY_ALIASES = {
    "understand.structure": "structure.analyze",
    "inspect.structure": "structure.analyze",
    "build.structure": "structure.generate",
    "build.2d": "structure.build2d",
    "dope.structure": "structure.dope",
    "run.template": "graph.run",
    "inspect.run": "graph.status",
    "verify.result": "result.verify",
    "parse.result": "result.parse",
    "compare.results": "analysis.compare",
}

_TEMPLATE_OUTPUT_LABELS = {
    "t0_scf": ["总能量", "原子数"],
    "t1_vc_relax": ["总能量", "平衡晶格参数", "压力", "体积"],
    "t2_bands": ["总能量", "带隙", "费米能级", "金属性"],
    "t2_dos": ["总能量", "费米能级态密度"],
}


def build_capability_graph(registry: Any, templates_dir: Optional[Path] = None) -> List[Capability]:
    """Discover currently enabled tools and graph-template output contracts."""
    capabilities: List[Capability] = []
    for tool in registry.list_all():
        if not tool.enabled:
            continue
        props = tool.input_schema.get("properties", {}) if isinstance(tool.input_schema, dict) else {}
        capabilities.append(Capability(
            id=tool.id, tool_id=tool.id, category=tool.category,
            description=tool.description[:220], inputs=list(props),
        ))
    if templates_dir and templates_dir.exists():
        for path in sorted(templates_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            template_id = str(payload.get("template_id") or path.stem)
            outputs = list((payload.get("outputs") or {}).keys())
            capabilities.append(Capability(
                id=f"run.{template_id}", tool_id="graph.run", category="graph-template",
                description=str(payload.get("description") or template_id)[:220],
                inputs=list((payload.get("inputs") or {}).keys()), outputs=outputs,
            ))
    return capabilities


def compile_plan(plan: PlanIR, registry: Any, *, templates_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Resolve semantic capabilities to live tools and return executor steps."""
    by_id = {c.id: c for c in build_capability_graph(registry, templates_dir)}
    tool_ids = {tool.id for tool in registry.list_all() if tool.enabled}
    compiled: List[Dict[str, Any]] = []
    errors: List[str] = []
    for action in plan.actions[:8]:
        capability = str(action.get("capability") or action.get("tool") or "").strip()
        resolved = _CAPABILITY_ALIASES.get(capability, capability)
        if capability.startswith("run.") and capability in by_id:
            resolved = "graph.run"
            args = dict(action.get("args") or {})
            args.setdefault("template_id", capability[4:])
        else:
            args = dict(action.get("args") or {})
        if resolved not in tool_ids:
            errors.append(f"unknown capability: {capability}")
            continue
        compiled.append({
            "tool": resolved,
            "args": args,
            "description": str(action.get("description") or capability)[:180],
        })
    plan.validation.extend(errors)
    return compiled


def capability_catalog_text(registry: Any, templates_dir: Optional[Path] = None) -> str:
    """Compact catalog for the LLM: semantic capabilities plus output contracts."""
    lines = []
    for cap in build_capability_graph(registry, templates_dir):
        suffix = f" -> outputs: {', '.join(cap.outputs)}" if cap.outputs else ""
        lines.append(f"- {cap.id} (tool={cap.tool_id}) | inputs: {', '.join(cap.inputs)} | {cap.description}{suffix}")
    return "\n".join(lines)
