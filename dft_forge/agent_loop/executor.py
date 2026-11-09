"""Execution loop: planned steps → tool calls → streamed events → reply."""

from __future__ import annotations

import asyncio
import copy
import json
import re
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from dft_forge.agent_loop.helpers import _dedupe_node_log
from dft_forge.agent_loop.execution_policy import check_execution_policy
from dft_forge.agent_loop.method_contract import MethodContractError, check_method_contract, method_for_step
from dft_forge.agent_loop.narration import compare_block, narrate
from dft_forge.agent_loop.provenance import result_provenance, provenance_warning
from dft_forge.agent_loop.planning import (
    PlanIR,
    effective_execution_mode,
    mark_blocked,
    render_plan_reply,
    requested_plan_only,
)
from dft_forge.agent_loop.result_views import RunState, handle_tool_result
from dft_forge.agent_loop.step_prep import align_formation_compounds, check_self_doping, prepare_step_args
from dft_forge.agent_loop.step_validator import semantic_plan_errors
from dft_forge.agent_loop.scientific_inputs import guard_scientific_inputs, trusted_thermodynamic_inputs


_STOP_REPLY = "计算已停止（进行中的步骤已终止；已完成的结果保留在会话中）。"


def _cancel_requested(ctx: Dict[str, Any]) -> bool:
    """Read the process-local cancellation side channel without serializing it."""
    event = ctx.get("cancel_event")
    is_set = getattr(event, "is_set", None)
    return bool(callable(is_set) and is_set())


def _references_prior_result(message: str, prior: dict) -> bool:
    run_id = prior.get("run_id")
    if isinstance(run_id, str) and run_id and re.search(r"(?<![A-Za-z0-9_])" + re.escape(run_id) + r"(?![A-Za-z0-9_])", message):
        return True
    earlier = re.search(r"上次|刚才|之前|上一条|刚算的|\b(?:previous|last|earlier)\b", message, re.I)
    result = re.search(r"结果|计算|数据|图|曲线|能量|带隙|结论|\b(?:result|calculation|data|plot|chart|curve|energy|gap|conclusion)s?\b", message, re.I)
    return bool((earlier and result) or re.search(r"这个结果.*(?:靠谱|可信)|它可信吗", message))


async def run_session(
    message: str,
    session_dir: Path,
    ctx: Optional[Dict[str, Any]] = None,
    history: Optional[List[Dict[str, Any]]] = None,
    on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
    *,
    registry,
    llm_planner,
    rule_planner,
) -> Dict[str, Any]:
    ctx = dict(ctx or {})
    confirmed_plan = ctx.pop("_confirmed_method_plan", None)
    if confirmed_plan and confirmed_plan.get("thermodynamic_inputs"):
        ctx["thermodynamic_inputs"] = copy.deepcopy(confirmed_plan["thermodynamic_inputs"])

    def emit(ev: Dict[str, Any]) -> None:
        if on_event is None:
            return
        try:
            on_event(ev)
        except Exception:
            pass

    emit({"type": "phase", "phase": "planning", "label": "理解需求，规划步骤…"})
    if ctx.get("last_run_id") or ctx.get("last_formula") or ctx.get("last_structure_source"):
        context_bits = []
        if ctx.get("last_formula"):
            context_bits.append(f"结构 {ctx['last_formula']}")
        if ctx.get("last_run_id"):
            context_bits.append(f"上次运行 {ctx['last_run_id']}")
        emit({"type": "context", "label": "已加载会话上下文：" + " · ".join(context_bits)})
    if confirmed_plan:
        llm_steps, llm_reply = copy.deepcopy(confirmed_plan["steps"]), None
    else:
        llm_steps, llm_reply = await asyncio.to_thread(llm_planner.plan, message, ctx, history)
    planned_ir = getattr(llm_planner, "last_plan_ir", None)
    structured_reply = isinstance(planned_ir, PlanIR) and (
        requested_plan_only(message)
        or planned_ir.execution_mode == "plan"
        or planned_ir.support_status != "complete"
        or planned_ir.blocking_reasons
        or planned_ir.advisory_steps
    )
    if llm_reply is not None and not structured_reply:
        prior = ctx.get("last_scientific_result") or {}
        if isinstance(prior, dict) and _references_prior_result(message, prior):
            warning = provenance_warning(prior.get("provenance"))
            if warning and not llm_reply.startswith(warning):
                llm_reply = warning + "\n" + llm_reply
        emit({"type": "reply", "text": llm_reply})
        return {
            "reply": llm_reply,
            "commands": [],
            "results": [],
            "chain": [],
            "session_id": ctx.get("session_id", str(uuid.uuid4())),
            "planner": "llm",
        }
    steps = llm_steps if llm_steps is not None else rule_planner.plan(message, ctx)
    planner_name = confirmed_plan["planner"] if confirmed_plan else ("llm" if llm_steps is not None else "rules")
    plan_ir = PlanIR(**copy.deepcopy(confirmed_plan["plan_ir"])) if confirmed_plan else getattr(llm_planner, "last_plan_ir", None)
    if not isinstance(plan_ir, PlanIR) or (planner_name != "llm" and not confirmed_plan):
        plan_ir = PlanIR.from_steps(message, steps, source=planner_name)
    # User intent is a security downgrade: a model can choose to be more
    # conservative, but it can never turn an explicit plan-only request into
    # an execution request.
    plan_ir.execution_mode = effective_execution_mode(
        message,
        plan_ir.execution_mode,
        ctx.get("requested_execution_mode"),
    )
    for semantic_error in semantic_plan_errors(message, steps):
        mark_blocked(plan_ir, semantic_error)
    guard_scientific_inputs(plan_ir, steps, message, ctx)
    if ctx.get("run_config"):
        for step in steps:
            invalid_structure = check_self_doping(step.get("tool"), step.get("args") or {})
            if invalid_structure:
                mark_blocked(plan_ir, invalid_structure)
    # Persist only explainable planning metadata; execution handles and raw
    # prompts never become part of the session context.
    ctx["last_plan"] = plan_ir.to_dict()
    emit({
        "type": "plan",
        "planner": planner_name,
        "steps": [{"tool": s.get("tool"), "description": s.get("description", "")} for s in steps],
        "summary": _plan_summary(message, steps),
        "goal": plan_ir.goal,
        "constraints": plan_ir.constraints,
        "assumptions": plan_ir.assumptions,
        "expected_outputs": plan_ir.expected_outputs,
        "execution_mode": plan_ir.execution_mode,
        "support_status": plan_ir.support_status,
        "missing_capabilities": plan_ir.missing_capabilities,
        "blocking_reasons": plan_ir.blocking_reasons,
    })

    # Planning and execution are separate product operations.  Partial or
    # unsupported plans are also non-executable: running their known prefix
    # would silently produce an answer to a different scientific question.
    if (
        plan_ir.execution_mode == "plan"
        or plan_ir.support_status != "complete"
        or plan_ir.over_budget
        or plan_ir.blocking_reasons
    ):
        reply = render_plan_reply(plan_ir, steps)
        chain = [
            {
                "tool": s.get("tool"),
                "description": s.get("description", ""),
                "status": "planned",
                "summary": "未执行",
                "nodes": [],
            }
            for s in steps
        ]
        emit({"type": "reply", "text": reply})
        return {
            "reply": reply,
            "commands": [],
            "results": [],
            "chain": chain,
            "ctx": {key: value for key, value in ctx.items() if key != "thermodynamic_inputs"},
            "planner": planner_name,
        }
    # Authorize the whole plan before running even its preparation prefix.
    # Text matching in the browser is only a convenience for opening a dialog.
    for note in align_formation_compounds(steps, session_dir):
        plan_ir.assumptions.append(note)
    automatic_method = bool((ctx.get("run_config") or {}).get("auto_parameters"))
    if not automatic_method:
        check_execution_policy(steps, ctx, registry)
    try:
        method_contract = check_method_contract(steps, ctx, session_dir)
    except MethodContractError as exc:
        if exc.code == "method_confirmation_required":
            exc.pending_plan = {"steps": copy.deepcopy(steps), "plan_ir": plan_ir.to_dict(), "planner": planner_name}
            exc.pending_plan["thermodynamic_inputs"] = trusted_thermodynamic_inputs(message, ctx)
        raise
    if automatic_method:
        check_execution_policy(steps, ctx, registry)
    if method_contract:
        ctx["last_method_contract"] = method_contract  # audit copy, never execution authority
        emit({"type": "method", "method_contract": method_contract})
    commands: List[Dict[str, Any]] = []
    results: List[Dict[str, Any]] = []
    reply_parts: List[str] = []
    chain: List[Dict[str, Any]] = []
    state = RunState(ctx, session_dir)
    cancelled = False
    structure_outputs = {}

    for idx, step in enumerate(steps):
        tool_id = step.get("tool")
        step_id = str(step.get("step_id") or f"step_{idx + 1}")
        args = dict(step.get("args") or {})
        description = step.get("description", "")

        if _cancel_requested(ctx):
            cancelled = True
            chain.append({
                "tool": tool_id,
                "step_id": step_id,
                "description": description,
                "status": "cancelled",
                "summary": "已停止",
                "nodes": [],
            })
            emit({
                "type": "step", "index": idx, "step_id": step_id,
                "status": "cancelled", "summary": "已停止",
            })
            break

        emit({"type": "step", "index": idx, "step_id": step_id, "status": "running", "tool": tool_id, "description": description})
        node_log: List[Dict[str, Any]] = []
        part_count = len(reply_parts)

        def _summary() -> str:
            new = [p for p in reply_parts[part_count:] if p]
            if not new:
                return ""
            line = new[0].splitlines()[0] if new[0].splitlines() else ""
            return line[:120]

        if not tool_id:
            commands.append({"command": "noop", "step_id": step_id, "description": description})
            results.append({
                "tool_id": None,
                "returncode": -1,
                "json": None,
                "error": "No matching tool",
                "traceback": None,
            })
            reply_parts.append(
                "未匹配到可执行工具，请尝试：导入结构、构建输入、提交任务、查看状态、查询账本、环境诊断。"
            )
            chain.append({"tool": None, "step_id": step_id, "description": description, "status": "error", "summary": _summary(), "nodes": []})
            emit({"type": "step", "index": idx, "step_id": step_id, "status": "error", "summary": "未匹配到可执行工具"})
            continue

        if tool_id == "help":
            commands.append({"command": "help", "step_id": step_id, "description": description, "args": args})
            results.append({
                "tool_id": "help",
                "returncode": 0,
                "json": None,
                "error": None,
                "traceback": None,
            })
            # a planner-provided Chinese notice (E_ads limits, compare
            # guidance...) beats the canned capability list; the English
            # fallback description ("Show help") does not
            notice = (description or "").strip()
            has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in notice)
            reply_parts.append(
                notice
                if notice and has_cjk
                else (
                    "我可以帮你做这些：\n"
                    "- 计算键长/分析结构：'计算nacl的键长'、'看看Si的结构'\n"
                    "- 生成标准结构：'生成NaCl结构'\n"
                    "- 构建QE输入：'构建vc-relax输入'、'build a relax input for Si'\n"
                    "- 提交/运行计算：'提交计算'、'run'\n"
                    "- 查看状态：'查看状态'、'status'\n"
                    "- 解析/验证结果：'解析结果'、'verify'\n"
                    "- 查询历史：'查询账本'、'history'\n"
                    "- 环境诊断：'环境诊断'、'doctor'"
                )
            )
            chain.append({"tool": "help", "step_id": step_id, "description": description, "status": "done", "summary": "能力说明", "nodes": []})
            emit({"type": "step", "index": idx, "step_id": step_id, "status": "done", "summary": "能力说明"})
            continue

        # ── guard: self-doping is physically meaningless ──────────────
        _nonsense_dopant = check_self_doping(tool_id, args)
        if _nonsense_dopant:
            commands.append({"command": "noop", "step_id": step_id, "description": description})
            results.append({
                "tool_id": tool_id,
                "returncode": -1,
                "json": None,
                "error": "self-doping rejected (dopant == host element)",
                "traceback": None,
            })
            reply_parts.append(_nonsense_dopant)
            chain.append({
                "tool": tool_id, "step_id": step_id, "description": description,
                "status": "error", "summary": "无意义掺杂已拦截", "nodes": [],
            })
            emit({"type": "step", "index": idx, "step_id": step_id, "status": "error", "summary": "无意义掺杂已拦截"})
            continue

        # Provenance is a runtime side channel, never authority supplied by
        # the planner/user. Preparation may populate it from stored results.
        args.pop("_input_provenance", None)
        args.pop("_method_contract", None)
        step_method = method_for_step(method_contract, idx)
        _abort_error = prepare_step_args(
            tool_id, args,
            session_dir=session_dir, ctx=ctx,
            built_structure=state.built_structure, scf_energies=state.scf_energies,
            node_log=node_log, emit=emit, idx=idx,
            method=step_method, structure_outputs=structure_outputs,
        )
        if _abort_error:
            commands.append({"command": "noop", "step_id": step_id, "description": description})
            results.append({
                "tool_id": tool_id,
                "returncode": -1,
                "json": None,
                "error": _abort_error,
                "traceback": None,
            })
            reply_parts.append(f"步骤失败: {description} — {_abort_error}")
            chain.append({"tool": tool_id, "step_id": step_id, "description": description, "status": "error",
                          "summary": _summary(), "nodes": []})
            emit({"type": "step", "index": idx, "step_id": step_id, "status": "error",
                  "summary": "无结构可链接（前置构建失败）"})
            continue

        # Tool preparation injects process-local callbacks/events into args.
        # Keep those handles out of the persisted/session-visible command log.
        logged_args = {key: value for key, value in args.items() if not str(key).startswith("_")}
        input_provenance = args.pop("_input_provenance", None)
        commands.append({
            "command": tool_id, "step_id": step_id, "depends_on": step.get("depends_on", []),
            "description": description, "args": logged_args,
        })

        try:
            result = await registry.call(tool_id, args, execution_mode=plan_ir.execution_mode)
            inner = None
            if isinstance(result.data, dict):
                if "json" in result.data:
                    inner = result.data["json"]  # _call_cmd wrapper
                elif "returncode" not in result.data:
                    inner = result.data  # graph tools return raw payloads
            if inner is None and str(args.get("output", "")).endswith(".json"):
                # commands with --output write JSON to file only; stdout is empty
                try:
                    inner = json.loads(Path(args["output"]).read_text())
                except (OSError, ValueError):
                    pass
            result_dict: Dict[str, Any] = {
                "tool_id": result.tool_id,
                "step_id": step_id,
                "step_index": idx,
                "returncode": 0 if result.error is None else 1,
                "json": inner,
                "error": result.error,
                "traceback": result.traceback,
                "provenance": result_provenance(tool_id, inner, input_provenance),
            }
            data = inner or {}
            if result.error:
                reply_parts.append(f"{description}: 失败 - {result.error}")
                ctx.setdefault("last_failures", []).append(f"{description}: {result.error}")
            elif isinstance(data, dict) and data.get("error") and tool_id not in ("structure.analyze",):
                # tool-level failure returned as payload (e.g. graph.run)
                reply_parts.append(f"{description}: 失败 - {data['error']}")
                ctx.setdefault("last_failures", []).append(f"{description}: {data['error']}")
            else:
                handle_tool_result(tool_id, data, args, description, state, result_dict, reply_parts)
                if any(result_dict.get(key) for key in ("facts", "chart", "metrics")):
                    ctx["last_scientific_result"] = {
                        "tool_id": tool_id, "run_id": data.get("run_id"),
                        "provenance": result_dict["provenance"],
                    }
        except Exception as exc:  # noqa: BLE001
            result_dict = {
                "tool_id": tool_id,
                "returncode": 1,
                "json": None,
                "error": str(exc),
                "traceback": None,
            }
            reply_parts.append(f"{description}: 异常 - {exc}")

        results.append(result_dict)

        step_failed = bool(result_dict.get("error")) or (
            isinstance(result_dict.get("json"), dict)
            and bool(result_dict["json"].get("error"))
            and tool_id not in ("structure.analyze",)
        )
        step_cancelled = _cancel_requested(ctx)
        if not step_failed and not step_cancelled and tool_id.startswith("structure."):
            output = (result_dict.get("json") or {}).get("output")
            if isinstance(output, str) and Path(output).is_file():
                structure_outputs[idx] = output
        status = "cancelled" if step_cancelled else ("error" if step_failed else "done")
        chain.append({
            "tool": tool_id, "step_id": step_id,
            "description": description,
            "status": status,
            "summary": "已停止" if step_cancelled else _summary(),
            "nodes": _dedupe_node_log(node_log),
        })
        emit({
            "type": "step", "index": idx, "step_id": step_id, "status": status,
            "summary": "已停止" if step_cancelled else _summary(),
        })
        if result_dict.get("viewer"):
            emit({"type": "figure", "viewer": result_dict["viewer"], "provenance": result_dict.get("provenance")})
        if result_dict.get("chart"):
            emit({"type": "figure", "chart": result_dict["chart"], "provenance": result_dict.get("provenance")})
        if step_cancelled:
            cancelled = True
            break

    reply = _STOP_REPLY if cancelled else ("\n".join(reply_parts) if reply_parts else "我已收到你的消息。")
    # LLM narration: replace template-concatenated reply with natural language
    # when a provider is configured; deterministic fallback keeps templates.
    provider = None if cancelled else llm_planner._get_provider()
    if provider is not None and commands:
        emit({"type": "phase", "phase": "narrating", "label": "整理计算结果…"})
        try:
            narrated = await asyncio.to_thread(narrate, provider, message, commands, results, ctx)
        except Exception:
            narrated = ""
        if narrated:
            reply = narrated
    # deterministic comparison numbers must survive LLM narration —
    # the narrator has paraphrased them into vague promises before
    if not cancelled:
        compare_blocks = []
        for c, r in zip(commands, results):
            data = r.get("json") if isinstance(r.get("json"), dict) else {}
            if c.get("command") == "analysis.compare" and isinstance(data, dict) and data.get("ok"):
                block = compare_block(data)
                if block not in reply:
                    compare_blocks.append(block)
        if compare_blocks:
            reply = reply.rstrip() + "\n" + "\n".join(compare_blocks)
        # Narration is allowed to be natural language, but it must not erase
        # deterministic values that the graph/status tools returned.
        fact_blocks = [str(r["facts"]) for r in results if r.get("facts")]
        for block in fact_blocks:
            if block not in reply:
                reply = reply.rstrip() + "\n" + block
    # Model narration cannot upgrade offline simulation into a scientific
    # result or remove the deterministic warning attached to these numbers.
    warnings = {provenance_warning(result.get("provenance")) for result in results}
    for warning in sorted(warnings - {"", None}):
        if not reply.startswith(warning):
            reply = warning + "\n" + reply
    if ctx.get("last_failures"):
        ctx["last_failures"] = ctx["last_failures"][-5:]
    emit({"type": "reply", "text": reply})
    return {
        "reply": reply,
        "commands": commands,
        "results": results,
        "chain": chain,
        "ctx": {key: value for key, value in ctx.items() if key != "thermodynamic_inputs"},
        "planner": planner_name,
    }


def _plan_summary(message: str, steps: List[Dict[str, Any]]) -> str:
    """Explain the chosen execution contract without exposing hidden CoT."""
    template_labels = {
        "t0_scf": "单点能",
        "t1_vc_relax": "变胞结构优化",
        "t2_bands": "能带结构",
        "t2_dos": "态密度",
    }
    templates = [
        template_labels.get(str((step.get("args") or {}).get("template_id")), str((step.get("args") or {}).get("template_id")))
        for step in steps
        if (step.get("args") or {}).get("template_id")
    ]
    tools = [str(step.get("tool")) for step in steps if step.get("tool")]
    if templates:
        return f"目标：{message[:80]} · 选择{ '、'.join(templates) }流程 · 预期回传能量/结构参数"
    if tools:
        return f"目标：{message[:80]} · 执行链：{' → '.join(tools)}"
    return f"目标：{message[:80]} · 无需调用计算工具"
