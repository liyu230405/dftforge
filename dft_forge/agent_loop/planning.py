"""Goal-first planning IR and dynamic capability graph.

The planner should describe *why* a workflow is needed before selecting a
tool.  This module is intentionally small and framework-agnostic: capabilities
are discovered from the live tool registry and graph-template output contracts,
then an IR is compiled into the existing executor step shape.
"""

from __future__ import annotations

import json
import re
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
    advisory_steps: List[Dict[str, Any]] = field(default_factory=list)
    reply: str = ""
    execution_mode: str = "execute"  # execute | plan
    support_status: str = "complete"  # complete | partial | unsupported
    required_capabilities: List[str] = field(default_factory=list)
    missing_capabilities: List[str] = field(default_factory=list)
    blocking_reasons: List[str] = field(default_factory=list)
    over_budget: bool = False
    validation: List[str] = field(default_factory=list)
    source: str = "rules"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: Dict[str, Any], *, source: str = "llm") -> "PlanIR":
        actions = payload.get("actions") or payload.get("steps") or []
        normalized: List[Dict[str, Any]] = []
        for i, action in enumerate(actions):
            if not isinstance(action, dict):
                continue
            normalized.append({
                "step_id": str(action.get("step_id") or action.get("id") or f"step_{i + 1}"),
                "depends_on": list(action.get("depends_on") or []),
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
            advisory_steps=[x for x in (payload.get("advisory_steps") or payload.get("advisory") or [])[:12] if isinstance(x, dict)],
            reply=str(payload.get("reply") or "").strip()[:4000],
            execution_mode=_normalize_execution_mode(payload.get("execution_mode") or payload.get("mode")),
            support_status=_normalize_support_status(payload.get("support_status") or payload.get("capability_status")),
            required_capabilities=[str(x)[:120] for x in (payload.get("required_capabilities") or [])[:20]],
            missing_capabilities=[str(x)[:160] for x in (payload.get("missing_capabilities") or payload.get("limitations") or [])[:20]],
            blocking_reasons=[str(x)[:200] for x in (payload.get("blocking_reasons") or [])[:12]],
            validation=[str(x)[:160] for x in (payload.get("validation") or payload.get("success_criteria") or [])[:8]],
            source=source,
        )

    @classmethod
    def from_steps(cls, goal: str, steps: Iterable[Dict[str, Any]], *, source: str = "rules") -> "PlanIR":
        normalized = []
        expected: List[str] = []
        for i, step in enumerate(steps):
            args = step.get("args") or {}
            normalized.append({
                "step_id": str(step.get("step_id") or f"step_{i + 1}"),
                "depends_on": list(step.get("depends_on") or []),
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
    side_effect: str = "unknown"
    semantic_contract: str = ""
    deferred_inputs: List[str] = field(default_factory=list)

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
    "t2_dos": ["总能量", "费米能级态密度", "元素分波态密度"],
}

MAX_PLAN_ACTIONS = 8
_PLAN_ONLY_PATTERNS = (
    r"只(?:做|给|需)?规划", r"仅(?:做|给|需)?规划", r"先规划", r"先给(?:出)?(?:方案|流程)",
    r"不要(?:实际)?运行", r"不(?:要|实际)?运行", r"不要执行", r"不提交", r"先不要提交",
    r"只给流程", r"仅说明流程", r"plan\s*-?\s*only", r"(?:do not|don't)\s+(?:actually\s+)?(?:run|execute)",
    r"without\s+running", r"just\s+plan",
)


def _normalize_execution_mode(value: Any) -> str:
    value = str(value or "execute").strip().lower().replace("-", "_")
    return "plan" if value in {"plan", "plan_only", "planning", "dry_run", "dryrun"} else "execute"


def _normalize_support_status(value: Any) -> str:
    value = str(value or "complete").strip().lower()
    if value in {"unsupported", "not_supported", "no"}:
        return "unsupported"
    if value in {"partial", "incomplete", "limited"}:
        return "partial"
    return "complete"


def requested_plan_only(message: str) -> bool:
    """Deterministic safety override: user wording may downgrade execution."""
    text = str(message or "").lower().replace("不实际执行", "不要实际执行")
    return any(re.search(pattern, text) for pattern in _PLAN_ONLY_PATTERNS)


def effective_execution_mode(
    message: str,
    planned_mode: str = "execute",
    requested_mode: Optional[str] = None,
) -> str:
    """Resolve mode with an explicit API request and a safety downgrade.

    The API/UI may force plan mode.  Natural-language no-run wording always
    wins over both the API and the model, so a model can never upgrade it.
    """
    if requested_plan_only(message):
        return "plan"
    if requested_mode is not None:
        return _normalize_execution_mode(requested_mode)
    return _normalize_execution_mode(planned_mode)


def mark_blocked(plan: PlanIR, reason: str, *, status: str = "partial") -> None:
    reason = str(reason).strip()
    if reason and reason not in plan.blocking_reasons:
        plan.blocking_reasons.append(reason)
    if status == "unsupported" or plan.support_status == "unsupported":
        plan.support_status = "unsupported"
    elif plan.support_status == "complete":
        plan.support_status = status


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
            side_effect=tool.side_effect,
            semantic_contract=tool.semantic_contract,
            deferred_inputs=list(tool.deferred_inputs),
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
    """Resolve capabilities without silently discarding unknown actions.

    The returned list contains only executable registry calls.  The original
    actions remain in ``PlanIR`` so unsupported scientific operations can be
    reported as advisory/missing capabilities rather than being mistaken for
    a complete workflow.
    """
    by_id = {c.id: c for c in build_capability_graph(registry, templates_dir)}
    tool_ids = {tool.id for tool in registry.list_all() if tool.enabled}
    compiled: List[Dict[str, Any]] = []
    errors: List[str] = []
    if len(plan.actions) > MAX_PLAN_ACTIONS:
        plan.over_budget = True
        mark_blocked(
            plan,
            f"计划包含 {len(plan.actions)} 个动作，超过安全上限 {MAX_PLAN_ACTIONS}；未静默截断",
        )
    for action in plan.actions:
        capability = str(action.get("capability") or action.get("tool") or "").strip()
        if capability and capability not in plan.required_capabilities:
            plan.required_capabilities.append(capability)
        resolved = _CAPABILITY_ALIASES.get(capability, capability)
        if capability.startswith("run.") and capability in by_id:
            resolved = "graph.run"
            args = dict(action.get("args") or {})
            args.setdefault("template_id", capability[4:])
        else:
            args = dict(action.get("args") or {})
        if resolved not in tool_ids:
            if capability and capability not in plan.missing_capabilities:
                plan.missing_capabilities.append(capability)
            mark_blocked(plan, f"当前工具注册表没有能力：{capability}", status="unsupported")
            continue
        compiled.append({
            "step_id": str(action.get("step_id") or f"step_{len(compiled) + 1}"),
            "depends_on": list(action.get("depends_on") or []),
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
        contract = f" | contract: {cap.semantic_contract}" if cap.semantic_contract else ""
        deferred = f" | deferred: {', '.join(cap.deferred_inputs)}" if cap.deferred_inputs else ""
        tool = registry.get(cap.tool_id)
        schema = (
            " | args_schema: " + json.dumps(tool.input_schema, ensure_ascii=False, separators=(",", ":"))
            if tool and cap.id == cap.tool_id else ""
        )
        lines.append(
            f"- {cap.id} (tool={cap.tool_id}) | inputs: {', '.join(cap.inputs)} | "
            f"effect: {cap.side_effect} | {cap.description}{contract}{deferred}{suffix}{schema}"
        )
    return "\n".join(lines)


def render_plan_reply(plan: PlanIR, steps: List[Dict[str, Any]]) -> str:
    """Stable, non-executing response for a plan or blocked workflow."""
    parts: List[str] = []
    if plan.execution_mode == "plan":
        parts.append("执行模式：仅规划，本轮未执行任何工具。")
    if plan.reply:
        parts.append(plan.reply)
    if plan.support_status != "complete":
        label = {"partial": "部分支持", "unsupported": "当前不支持"}.get(plan.support_status, plan.support_status)
        parts.append(f"能力判断：{label}。")
    if plan.blocking_reasons:
        parts.append("阻断原因：" + "；".join(plan.blocking_reasons[:4]) + "。")
    if plan.missing_capabilities:
        parts.append("缺少能力：" + "、".join(plan.missing_capabilities[:8]) + "。")
    if plan.advisory_steps:
        advisory = []
        for action in plan.advisory_steps[:8]:
            name = action.get("capability") or action.get("tool") or "框架外步骤"
            desc = str(action.get("description") or "").strip()
            advisory.append(f"{name}{'：' + desc if desc else ''}")
        parts.append("框架外科研建议：\n" + "\n".join(f"- {x}" for x in advisory))
    actions = plan.actions or [
        {"capability": s.get("tool"), "description": s.get("description", "")}
        for s in steps
    ]
    if actions:
        rendered = []
        for action in actions[:MAX_PLAN_ACTIONS]:
            name = action.get("capability") or action.get("tool") or "未命名动作"
            desc = str(action.get("description") or "").strip()
            rendered.append(f"{name}{'：' + desc if desc else ''}")
        if len(actions) > MAX_PLAN_ACTIONS:
            rendered.append(f"（其余 {len(actions) - MAX_PLAN_ACTIONS} 个动作未展开）")
        parts.append("规划步骤：\n" + "\n".join(f"{i + 1}. {x}" for i, x in enumerate(rendered)))
    if plan.expected_outputs:
        parts.append("预期输出：" + "、".join(plan.expected_outputs) + "。")
    return "\n\n".join(p for p in parts if p) or "已生成计划，但当前没有可执行动作。"
