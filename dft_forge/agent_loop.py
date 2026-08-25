"""Agent loop backend for chat-driven multi-step tool execution."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from dft_forge.tools.definitions import register_default_tools
from dft_forge.tools.registry import registry

register_default_tools()


class AgentLoop:
    """Rule-based planner + executor over the tool registry.

    Keeps light session context so follow-up messages can reuse the
    last structure, input file, and job id.
    """

    def __init__(self, registry_obj=None):
        self.registry = registry_obj or registry

    def list_tools(self) -> List[Dict[str, Any]]:
        return [t.to_dict() for t in self.registry.list_all()]

    def plan(self, message: str, ctx: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        lower = message.lower()
        ctx = ctx or {}
        steps: List[Dict[str, Any]] = []

        has_structure_ctx = bool(
            ctx.get("last_structure_file") or ctx.get("last_structure_source") or ctx.get("last_input_file")
        )
        has_job_ctx = bool(ctx.get("last_job_id"))

        material_hint = None
        pair_types = None
        if any(k in lower for k in ["nacl", "氯化钠", "clna"]):
            material_hint = "NaCl"
            pair_types = "Na-Cl"
        elif any(k in lower for k in ["mgo", "氧化镁", "gomg"]):
            material_hint = "MgO"
            pair_types = "Mg-O"
        elif any(k in lower for k in ["si", "硅"]):
            material_hint = "Si"
            pair_types = "Si-Si"
        elif any(k in lower for k in ["al", "铝"]):
            material_hint = "Al"
            pair_types = "Al-Al"

        # Detect compound intents
        wants_analyze = any(
            k in lower
            for k in [
                "键长", "bond", "distance", "距离", "分析", "analyze", "info",
                "看看", "查看结构", "结构信息", "原子", "atoms", "体积", "volume",
                "晶胞", "cell", "neighbors", "邻居", "结构", "structure",
            ]
        )
        wants_import = any(k in lower for k in ["导入", "import", "读取", "load", "打开", "open", "加载", "读取结构", "导入结构"])
        wants_generate = any(k in lower for k in ["生成", "generate", "创建", "create", "做个", "建一个"]) and "build a" not in lower
        wants_build = any(
            k in lower
            for k in [
                "构建", "build", "生成", "generate", "输入", "input", "松弛", "relax",
                "vc-relax", "scf", "bands", "dos", "态密度", "带", "计算类型", "in文件",
                "qe输入", "输入文件",
            ]
        ) and not wants_generate
        wants_submit = any(k in lower for k in ["提交", "submit", "运行", "run", "执行", "execute", "开始", "提交计算"]) and "计算" not in lower or ("计算" in lower and any(k in lower for k in ["提交", "运行", "run", "execute"]))
        wants_status = any(k in lower for k in ["状态", "status", "进度", "progress", "查看", "check", "任务", "job"])
        wants_parse = any(k in lower for k in ["解析", "parse", "结果", "result", "输出", "output", "读取结果"])
        wants_verify = any(k in lower for k in ["验证", "verify", "校验", "校验结果"])
        wants_ledger = any(k in lower for k in ["账本", "ledger", "历史", "history", "record", "查询", "query", "记录", "日志"])
        wants_doctor = any(k in lower for k in ["环境", "env", "doctor", "诊断", "diagnose", "依赖", "dependencies", "安装", "software"])

        # If analyze is requested, do not auto-trigger submit/build unless explicitly requested
        if wants_analyze:
            wants_submit = False
            wants_build = False
            wants_parse = False
            wants_verify = False

        # Intent: analyze / bond length / structure info
        if wants_analyze:
            source = ctx.get("last_structure_file") or ctx.get("last_structure_source")
            if not source:
                if wants_generate or material_hint:
                    steps.append({
                        "tool": "structure.generate",
                        "args": {"source": material_hint or "Si"},
                        "description": f"Generate {material_hint or 'Si'} structure",
                    })
                    source = "__generated__"
                else:
                    source = "tests/fixtures/POSCAR_Si"
            steps.append({
                "tool": "structure.analyze",
                "args": {"source": source, "pair_types": pair_types},
                "description": "Analyze structure",
            })

        # Intent: import / load structure
        elif wants_import and not wants_analyze:
            if not has_structure_ctx:
                source = material_hint or "tests/fixtures/POSCAR_Si"
                steps.append({
                    "tool": "structure.import",
                    "args": {"source": source},
                    "description": "Import structure",
                })

        # Intent: generate structure
        elif wants_generate and not wants_analyze:
            steps.append({
                "tool": "structure.generate",
                "args": {"source": material_hint or "Si"},
                "description": f"Generate {material_hint or 'Si'} structure",
            })

        # Intent: build QE input
        if wants_build:
            calc = "scf"
            if any(k in lower for k in ["vc-relax", "relax", "松弛", "vc_relax"]):
                calc = "vc-relax"
            elif any(k in lower for k in ["bands", "带"]):
                calc = "bands"
            elif any(k in lower for k in ["dos", "态密度"]):
                calc = "dos"

            material = material_hint or "Si"
            source = ctx.get("last_structure_file") or ctx.get("last_structure_source")
            if source:
                steps.append({
                    "tool": "input.build",
                    "args": {"type": calc, "structure": source, "structure_format": "auto"},
                    "description": f"Build {calc} input from structure",
                })
            else:
                steps.append({
                    "tool": "input.build",
                    "args": {"type": calc, "material": material},
                    "description": f"Build {calc} input for {material}",
                })

        # Intent: submit / run calculation
        if wants_submit:
            input_file = ctx.get("last_input_file") or "input.in"
            steps.append({
                "tool": "job.submit",
                "args": {"input": input_file, "backend": "local"},
                "description": "Submit job",
            })

        # Intent: job status
        if wants_status:
            job_id = ctx.get("last_job_id") or (message.split()[-1] if message.split() else "local_default")
            steps.append({
                "tool": "job.status",
                "args": {"job_id": job_id},
                "description": "Check job status",
            })

        # Intent: parse result
        if wants_parse:
            output_file = ctx.get("last_output_file") or "input.in"
            steps.append({
                "tool": "result.parse",
                "args": {"input": output_file, "type": ctx.get("last_result_type", "vc-relax")},
                "description": "Parse result",
            })

        # Intent: verify
        if wants_verify:
            output_file = ctx.get("last_output_file") or "input.in"
            steps.append({
                "tool": "result.verify",
                "args": {"input": output_file, "task_type": ctx.get("last_task_type", "T1")},
                "description": "Verify result",
            })

        # Intent: ledger / history
        if wants_ledger:
            steps.append({
                "tool": "ledger.query",
                "args": {"limit": 20},
                "description": "Query ledger",
            })

        # Intent: doctor / environment
        if wants_doctor:
            steps.append({
                "tool": "doctor",
                "args": {},
                "description": "Check environment",
            })

        if not steps:
            steps.append({
                "tool": "help",
                "args": {},
                "description": "Show help",
            })

        return steps

    async def run(self, message: str, session_dir: Path, ctx: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        ctx = dict(ctx or {})
        steps = self.plan(message, ctx)
        commands: List[Dict[str, Any]] = []
        results: List[Dict[str, Any]] = []
        reply_parts: List[str] = []

        for step in steps:
            tool_id = step.get("tool")
            args = dict(step.get("args") or {})
            description = step.get("description", "")

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
                reply_parts.append(
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
                continue

            if tool_id == "structure.import":
                args["output"] = str(session_dir / "structure.json")
            elif tool_id == "structure.generate":
                args.setdefault("output", str(session_dir / "generated.cif"))
            elif tool_id == "structure.analyze" and args.get("source") == "__generated__":
                args["source"] = ctx.get("last_structure_source") or args.get("source")
            elif tool_id == "input.build":
                args["output"] = str(session_dir / "input.in")
            elif tool_id == "job.submit":
                args["input"] = str(session_dir / "input.in")
                args["output"] = str(session_dir / "job_result.json")
            elif tool_id == "result.parse":
                args.setdefault("input", str(session_dir / "input.in"))
                args.setdefault("type", ctx.get("last_result_type", "vc-relax"))
            elif tool_id == "result.verify":
                args.setdefault("input", str(session_dir / "input.in"))
                args.setdefault("task_type", ctx.get("last_task_type", "T1"))
            elif tool_id == "ledger.query":
                args["database"] = str(session_dir.parent / "ledger.db")

            commands.append({"command": tool_id, "description": description, "args": args})

            try:
                result = await self.registry.call(tool_id, args)
                inner = result.data.get("json") if isinstance(result.data, dict) else result.data
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
                elif tool_id == "structure.import" and data.get("ok") and data.get("success"):
                    ctx["last_structure_file"] = str(session_dir / "structure.json")
                    ctx["last_structure_source"] = data.get("source") or args.get("source")
                    ctx["last_formula"] = data.get("formula")
                    viewer_payload = {
                        "formula": data.get("formula"),
                        "cif": data.get("cif"),
                        "source": data.get("source"),
                    }
                    reply_parts.append(
                        f"已导入结构: {data.get('formula')} | atoms={data.get('natoms')} species={data.get('nspecies')}"
                    )
                    result_dict["viewer"] = viewer_payload
                elif tool_id == "structure.generate" and data.get("ok") and data.get("output"):
                    ctx["last_structure_source"] = data.get("output")
                    ctx["last_formula"] = data.get("formula")
                    reply_parts.append(f"已生成结构: {data.get('formula')} -> {data.get('output')}")
                elif tool_id == "input.build" and data.get("input_file"):
                    ctx["last_input_file"] = data.get("input_file")
                    ctx["last_calc_type"] = data.get("calc_type")
                    reply_parts.append(f"已构建输入: {data.get('input_file')} ({data.get('calc_type')})")
                elif tool_id == "job.submit" and data.get("ok") and data.get("success"):
                    ctx["last_job_id"] = data.get("job_id")
                    ctx["last_output_file"] = data.get("output_files", [None])[0] if data.get("output_files") else str(session_dir / "input.in")
                    reply_parts.append(f"任务已提交: job_id={data.get('job_id')} walltime={data.get('walltime_sec')}s")
                elif tool_id == "job.status":
                    reply_parts.append(f"任务状态: {data.get('state', 'unknown')}")
                elif tool_id == "result.parse":
                    ctx["last_output_file"] = args.get("input")
                    ctx["last_result_type"] = args.get("type")
                    reply_parts.append(f"解析完成: {args.get('type')}")
                elif tool_id == "result.verify":
                    reply_parts.append(f"验证完成: passed={data.get('passed')}")
                elif tool_id == "ledger.query":
                    reply_parts.append(f"账本记录数: {data.get('count', 0)}")
                elif tool_id == "doctor":
                    reply_parts.append(
                        f"环境诊断完成: QE={data.get('qe_binary')} bands.x={data.get('bands_x_binary')} dos.x={data.get('dos_x_binary')}"
                    )
                elif tool_id == "structure.analyze":
                    analysis = data.get("analysis") or {}
                    formula = analysis.get("formula") or data.get("formula") or ctx.get("last_formula")
                    pair_distances = analysis.get("pair_distances") or {}
                    min_pair = analysis.get("minimum_distance_pair")
                    min_dist = analysis.get("minimum_distance_angstrom")
                    lines = [f"结构分析: {formula} | atoms={analysis.get('natoms')} species={analysis.get('nspecies')}"]
                    if min_pair and min_dist is not None:
                        lines.append(f"最短距离: {min_dist:.4f} Å ({'-'.join(min_pair)})")
                    for pair, info in pair_distances.items():
                        lines.append(f"{pair}: count={info.get('count')} min={info.get('min_angstrom')} Å max={info.get('max_angstrom')} Å")
                    reply_parts.append("\n".join(lines))
                else:
                    reply_parts.append(f"{description}: 完成")
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

        reply = "\n".join(reply_parts) if reply_parts else "我已收到你的消息。"
        return {
            "reply": reply,
            "commands": commands,
            "results": results,
            "ctx": ctx,
        }
