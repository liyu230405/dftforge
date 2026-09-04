"""Execution loop: planned steps → tool calls → streamed events → reply."""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from dft_forge.agent_loop.helpers import _dedupe_node_log
from dft_forge.agent_loop.narration import compare_block, narrate
from dft_forge.agent_loop.result_views import RunState, handle_tool_result
from dft_forge.agent_loop.step_prep import check_self_doping, prepare_step_args


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

    def emit(ev: Dict[str, Any]) -> None:
        if on_event is None:
            return
        try:
            on_event(ev)
        except Exception:
            pass

    emit({"type": "phase", "phase": "planning", "label": "理解需求，规划步骤…"})
    llm_steps, llm_reply = await asyncio.to_thread(llm_planner.plan, message, ctx, history)
    if llm_reply is not None:
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
    planner_name = "llm" if llm_steps is not None else "rules"
    emit({
        "type": "plan",
        "planner": planner_name,
        "steps": [{"tool": s.get("tool"), "description": s.get("description", "")} for s in steps],
    })
    commands: List[Dict[str, Any]] = []
    results: List[Dict[str, Any]] = []
    reply_parts: List[str] = []
    chain: List[Dict[str, Any]] = []
    state = RunState(ctx, session_dir)

    for idx, step in enumerate(steps):
        tool_id = step.get("tool")
        args = dict(step.get("args") or {})
        description = step.get("description", "")
        emit({"type": "step", "index": idx, "status": "running", "tool": tool_id, "description": description})
        node_log: List[Dict[str, Any]] = []
        part_count = len(reply_parts)

        def _summary() -> str:
            new = [p for p in reply_parts[part_count:] if p]
            if not new:
                return ""
            line = new[0].splitlines()[0] if new[0].splitlines() else ""
            return line[:120]

        if not tool_id:
            commands.append({"command": "noop", "description": description})
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
            chain.append({"tool": None, "description": description, "status": "error", "summary": _summary(), "nodes": []})
            emit({"type": "step", "index": idx, "status": "error", "summary": "未匹配到可执行工具"})
            continue

        if tool_id == "help":
            commands.append({"command": "help", "description": description, "args": args})
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
            chain.append({"tool": "help", "description": description, "status": "done", "summary": "能力说明", "nodes": []})
            emit({"type": "step", "index": idx, "status": "done", "summary": "能力说明"})
            continue

        # ── guard: self-doping is physically meaningless ──────────────
        _nonsense_dopant = check_self_doping(tool_id, args)
        if _nonsense_dopant:
            commands.append({"command": "noop", "description": description})
            results.append({
                "tool_id": tool_id,
                "returncode": -1,
                "json": None,
                "error": "self-doping rejected (dopant == host element)",
                "traceback": None,
            })
            reply_parts.append(_nonsense_dopant)
            chain.append({
                "tool": tool_id, "description": description,
                "status": "error", "summary": "无意义掺杂已拦截", "nodes": [],
            })
            emit({"type": "step", "index": idx, "status": "error", "summary": "无意义掺杂已拦截"})
            continue

        _abort_error = prepare_step_args(
            tool_id, args,
            session_dir=session_dir, ctx=ctx,
            built_structure=state.built_structure, scf_energies=state.scf_energies,
            node_log=node_log, emit=emit, idx=idx,
        )
        if _abort_error:
            commands.append({"command": "noop", "description": description})
            results.append({
                "tool_id": tool_id,
                "returncode": -1,
                "json": None,
                "error": _abort_error,
                "traceback": None,
            })
            reply_parts.append(f"步骤失败: {description} — 前置结构构建未成功，无法链接结构。")
            chain.append({"tool": tool_id, "description": description, "status": "error",
                          "summary": _summary(), "nodes": []})
            emit({"type": "step", "index": idx, "status": "error",
                  "summary": "无结构可链接（前置构建失败）"})
            continue

        commands.append({"command": tool_id, "description": description, "args": args})

        try:
            result = await registry.call(tool_id, args)
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
                "returncode": 0 if result.error is None else 1,
                "json": inner,
                "error": result.error,
                "traceback": result.traceback,
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
        status = "error" if step_failed else "done"
        chain.append({
            "tool": tool_id,
            "description": description,
            "status": status,
            "summary": _summary(),
            "nodes": _dedupe_node_log(node_log),
        })
        emit({"type": "step", "index": idx, "status": status, "summary": _summary()})
        if result_dict.get("viewer"):
            emit({"type": "figure", "viewer": result_dict["viewer"]})
        if result_dict.get("chart"):
            emit({"type": "figure", "chart": result_dict["chart"]})

    reply = "\n".join(reply_parts) if reply_parts else "我已收到你的消息。"
    # LLM narration: replace template-concatenated reply with natural language
    # when a provider is configured; deterministic fallback keeps templates.
    provider = llm_planner._get_provider()
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
    compare_blocks = []
    for c, r in zip(commands, results):
        data = r.get("json") if isinstance(r.get("json"), dict) else {}
        if c.get("command") == "analysis.compare" and isinstance(data, dict) and data.get("ok"):
            block = compare_block(data)
            if block not in reply:
                compare_blocks.append(block)
    if compare_blocks:
        reply = reply.rstrip() + "\n" + "\n".join(compare_blocks)
    if ctx.get("last_failures"):
        ctx["last_failures"] = ctx["last_failures"][-5:]
    emit({"type": "reply", "text": reply})
    return {
        "reply": reply,
        "commands": commands,
        "results": results,
        "chain": chain,
        "ctx": ctx,
        "planner": planner_name,
    }
