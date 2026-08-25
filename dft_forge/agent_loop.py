"""Agent loop backend for chat-driven multi-step tool execution."""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from dft_forge.tools.definitions import register_default_tools
from dft_forge.tools.registry import registry

register_default_tools()

_CHAT_SKIP_TOOLS = {"help"}

_ELEMENTS = {
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S",
    "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga",
    "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd",
    "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm",
    "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os",
    "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa",
    "U", "Np", "Pu",
}

# Chinese aliases for quick material lookup in chat text
_ZH_MATERIAL = {"硅": "Si", "铝": "Al", "氯化钠": "NaCl", "食盐": "NaCl", "氧化镁": "MgO"}


def extract_formula(message: str) -> Optional[str]:
    """Pull a chemical formula (e.g. CaTiO3, GaAs, NaCl) out of chat text.

    Accepts any casing (catio3 → CaTiO3) as long as tokens map to real
    element symbols. Longest match wins so GaAs is preferred over Ga/As.
    """
    text = message
    for zh, en in _ZH_MATERIAL.items():
        if zh in text:
            return en
    tokens = re.findall(r"[A-Za-z]{1,2}\d{0,3}(?:\s?[A-Za-z]{1,2}\d{0,3})*", text)
    best = None
    for tok in tokens:
        parts = re.findall(r"[A-Za-z]{1,2}\d{0,3}", tok)
        syms = []
        ok = True
        for p in parts:
            m = re.match(r"([A-Za-z]{1,2})(\d*)", p)
            sym, num = m.group(1), m.group(2)
            cand = None
            for probe in (sym.capitalize(), sym.upper()):
                if probe in _ELEMENTS:
                    cand = probe
                    break
            if cand is None:
                ok = False
                break
            syms.append(cand + num)
        if not ok or not syms or len(syms) > 4:
            continue
        formula = "".join(syms)
        if best is None or len(formula) > len(best):
            best = formula
    return best


def _read_text(path: Any) -> Optional[str]:
    try:
        return Path(str(path)).read_text()
    except OSError:
        return None


def _viewer_payload_for_material(material: Any) -> Optional[Dict[str, Any]]:
    """CIF viewer payload for a material key or raw formula (e.g. CaTiO3)."""
    if not material or not str(material).strip():
        return None
    try:
        import tempfile

        from ase.io import write as ase_write

        from dft_forge.compiler import MATERIAL_DB, build_atoms, formula_atoms

        name = str(material).strip()
        atoms = build_atoms(name) if name in MATERIAL_DB else formula_atoms(name)[0]
        with tempfile.NamedTemporaryFile(suffix=".cif", mode="w+", delete=False) as tf:
            ase_write(tf.name, atoms, format="cif")
            cif = Path(tf.name).read_text()
        Path(tf.name).unlink(missing_ok=True)
        return {
            "formula": atoms.get_chemical_formula(),
            "cif": cif,
            "natoms": len(atoms),
            "source": name,
        }
    except Exception:
        return None


def _extract_charts(nodes: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Pull plottable curves + headline numbers from graph node outputs."""
    chart: Dict[str, Any] = {}
    for info in nodes.values():
        outs = info.get("outputs") or {}
        if outs.get("eigenvalues_ev"):
            bands = {
                "kind": "bands",
                "eigenvalues_ev": outs["eigenvalues_ev"],
                "fermi_ev": outs.get("fermi_ev"),
                "band_gap_ev": outs.get("band_gap_ev"),
                "is_metal": outs.get("is_metal"),
                "n_bands": outs.get("n_bands"),
                "n_kpoints": outs.get("n_kpoints"),
            }
            for key in ("k_axis", "k_ticks", "k_labels"):
                if outs.get(key):
                    bands[key] = outs[key]
            chart["bands"] = bands
        if outs.get("dos_curve"):
            chart["dos"] = {"kind": "dos", **outs["dos_curve"], "fermi_ev": outs.get("fermi_ev")}
        for k in ("energy_ry", "a_angstrom", "pressure_kbar", "max_force_ev_ang"):
            v = outs.get(k)
            if v is None:
                continue
            metrics = chart.setdefault("metrics", {})
            if k not in metrics or (not metrics[k] and v):  # nonzero wins over nscf placeholder zeros
                metrics[k] = v
    return chart or None


def _load_env_file(path: Path) -> None:
    """Load 'export KEY=VALUE' lines from .env without overwriting real env."""
    if not path.exists():
        return
    import os
    import re

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$", line)
        if not m:
            continue
        key, val = m.group(1), m.group(2).strip().strip("'\"")
        if key not in os.environ:
            os.environ[key] = val


class LLMPlanner:
    """LLM-based planner: natural language → tool-call steps.

    Falls back (returns None) on any error so AgentLoop can use rules.
    """

    def __init__(self, registry_obj):
        self.registry = registry_obj
        self._provider = None
        self._checked = False

    def _get_provider(self):
        if self._checked:
            return self._provider
        self._checked = True
        import os

        if os.environ.get("DFT_FORGE_LLM_PROVIDER", "dummy").lower() not in ("openai", "openai_compatible", "openai-compatible"):
            return None
        if not os.environ.get("DFT_FORGE_LLM_API_KEY"):
            return None
        try:
            from dft_forge.llm import get_llm_provider

            provider = get_llm_provider()
            self._provider = provider if hasattr(provider, "chat") else None
        except Exception:
            self._provider = None
        return self._provider

    def _catalog(self) -> str:
        from dft_forge.compiler import MATERIAL_DB

        lines = []
        for t in self.registry.list_all():
            if t.id in _CHAT_SKIP_TOOLS:
                continue
            props = t.input_schema.get("properties", {}) if isinstance(t.input_schema, dict) else {}
            args = ", ".join(f"{k}:{v.get('type', '?')}" for k, v in props.items())
            lines.append(f"- {t.id} | args: {args} | {t.description}")
        lines.append(f"Available bulk materials: {', '.join(sorted(MATERIAL_DB))}")
        return "\n".join(lines)

    def plan(self, message: str, ctx: Dict[str, Any]):
        """Returns (steps, direct_reply) or (None, None) to fall back to rules."""
        provider = self._get_provider()
        if provider is None:
            return None, None
        system = f"""You are the planner of DFT-Forge, a DFT computation agent (Quantum ESPRESSO).
Convert the user's request into a JSON plan of tool calls.

Rules:
- Reply with ONLY one JSON object, no markdown fences:
  {{"reply": string, "steps": [{{"tool": string, "args": {{}}, "description": string}}]}}
- Use ONLY the tool ids listed below; args keys must match exactly.
- ANY chemical formula works directly: pass it as inputs.material of graph.run
  (e.g. CaTiO3, SrTiO3, GaAs). Unknown formulas are auto-built into prototype
  structures — NEVER ask the user to import a file for a plain bulk crystal.
- For a full verified calculation prefer graph.run with a template_id
  (t1_vc_relax = structure optimization, t2_bands = band structure, t2_dos = density of states).
  "算 X 的能带" → one graph.run t2_bands step; combining several targets emits one step each.
- For 2D materials (graphene/h-BN, doping, adsorption sites) use structure.build2d,
  then optionally input.build with the produced CIF path and type.
- Site scan ("different sites/positions"): emit one structure.build2d step per site
  (top/bridge/hollow), each with its own output path.
- Chit-chat / questions about you: empty steps, answer in "reply".
- descriptions in Chinese. At most 8 steps.

Tools:
{self._catalog()}"""
        ctx_summary = {k: ctx.get(k) for k in ("last_structure_source", "last_formula", "last_input_file", "last_job_id") if ctx.get(k)}
        user = f"Context: {json.dumps(ctx_summary, ensure_ascii=False)}\nUser: {message}"
        try:
            raw = provider.chat(system, user)
            data = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
        except Exception:
            return None, None
        if not isinstance(data, dict):
            return None, None
        known = {t.id for t in self.registry.list_all()}
        steps = []
        for s in (data.get("steps") or [])[:8]:
            if not isinstance(s, dict) or s.get("tool") not in known:
                return None, None
            steps.append({
                "tool": s["tool"],
                "args": s.get("args") or {},
                "description": str(s.get("description", ""))[:120],
            })
        reply = str(data.get("reply") or "").strip()
        return steps, (reply if reply and not steps else None)


class AgentLoop:
    """Planner + executor over the tool registry.

    Tries the LLM planner first (if configured via DFT_FORGE_LLM_* env);
    falls back to deterministic rules. Keeps light session context so
    follow-up messages can reuse the last structure, input file, and job id.
    """

    def __init__(self, registry_obj=None):
        self.registry = registry_obj or registry
        _load_env_file(Path(__file__).resolve().parent.parent / ".env")
        self.llm_planner = LLMPlanner(self.registry)

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
        formula_hint = material_hint or extract_formula(message)

        # Detect compound intents
        wants_analyze = any(
            k in lower
            for k in [
                "键长", "键能", "bond", "distance", "距离", "分析", "analyze", "info",
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

        # Strong analysis words beat calculation; bare "结构" does not (能带结构 etc.)
        strong_analyze = any(k in lower for k in ("键长", "键能", "bond", "distance", "距离", "分析", "analyze"))

        # Full-calculation intent: one-shot graph.run for ANY formula — the
        # engine auto-builds a prototype structure when the material is unknown.
        graph_targets: List[str] = []
        if any(k in lower for k in ("能带", "带结构", "带隙", "bands", "band")):
            graph_targets.append("t2_bands")
        if any(k in lower for k in ("态密度", "dos")):
            graph_targets.append("t2_dos")
        if any(k in lower for k in ("结构优化", "晶格常数", "弛豫", "松弛", "优化", "vc-relax", "vcrelax", "vc_relax")):
            graph_targets.append("t1_vc_relax")
        wants_generic_calc = any(k in lower for k in ("算", "计算", "跑", "仿真", "simulation"))
        if (
            not graph_targets
            and wants_generic_calc
            and not strong_analyze
            and not (wants_import or wants_generate or wants_build or wants_status or wants_ledger or wants_doctor)
        ):
            graph_targets.append("t1_vc_relax")

        if graph_targets and formula_hint and not strong_analyze and not (wants_import or wants_generate):
            label = {"t2_bands": "能带结构", "t2_dos": "态密度", "t1_vc_relax": "结构优化"}
            for tid in graph_targets:
                steps.append({
                    "tool": "graph.run",
                    "args": {"template_id": tid, "inputs": {"material": formula_hint}},
                    "description": f"{formula_hint} {label[tid]}",
                })
            return steps

        # Intent: analyze / bond length / structure info
        if wants_analyze:
            source = ctx.get("last_structure_file") or ctx.get("last_structure_source")
            if not source:
                if wants_generate or material_hint or formula_hint:
                    steps.append({
                        "tool": "structure.generate",
                        "args": {"source": material_hint or formula_hint or "Si"},
                        "description": f"Generate {material_hint or formula_hint or 'Si'} structure",
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
        llm_steps, llm_reply = self.llm_planner.plan(message, ctx)
        if llm_reply is not None:
            return {
                "reply": llm_reply,
                "commands": [],
                "results": [],
                "session_id": ctx.get("session_id", str(uuid.uuid4())),
                "planner": "llm",
            }
        steps = llm_steps if llm_steps is not None else self.plan(message, ctx)
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
                args["output"] = str(session_dir / "generated.cif")
            elif tool_id == "structure.build2d":
                kind = str(args.get("kind", "2d")).replace("/", "_")
                sc = str(args.get("supercell", "1x1")).replace("x", "_")
                site = str((args.get("adsorb") or {}).get("site", "")).replace("/", "_")
                args["output"] = str(session_dir / f"{kind}_{sc}{'_' + site if site else ''}.cif")
            elif tool_id == "structure.analyze" and str(args.get("output", "")).endswith(".json"):
                args["output"] = str(session_dir / "structure_analysis.json")
            elif tool_id == "structure.analyze" and args.get("source") == "__generated__":
                args["source"] = ctx.get("last_structure_source") or args.get("source")
            elif tool_id == "input.build":
                args["output"] = str(session_dir / "input.in")
            elif tool_id == "graph.run":
                if not str(args.get("workdir") or "").strip():
                    args["workdir"] = str(session_dir / "graphs")
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
                elif isinstance(data, dict) and data.get("error") and tool_id not in ("structure.analyze",):
                    # tool-level failure returned as payload (e.g. graph.run)
                    reply_parts.append(f"{description}: 失败 - {data['error']}")
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
                elif tool_id == "structure.generate" and data.get("output"):
                    ctx["last_structure_source"] = data.get("output")
                    ctx["last_formula"] = data.get("formula")
                    ctx["last_structure_file"] = data.get("output")
                    reply_parts.append(f"已生成结构: {data.get('formula')}（{len(data.get('species', []))} 种元素，{data.get('natoms')} 原子）\n文件: {data.get('output')}")
                    result_dict["viewer"] = {
                        "formula": data.get("formula"),
                        "cif_path": data.get("output"),
                        "cif": _read_text(data.get("output")),
                        "natoms": data.get("natoms"),
                    }
                elif tool_id == "structure.build2d" and data.get("output"):
                    ctx["last_structure_source"] = data.get("output")
                    ctx["last_structure_file"] = data.get("output")
                    ctx["last_formula"] = data.get("formula")
                    lines = [
                        f"已构建2D结构: {data.get('formula')} — {data.get('description')}",
                        f"原子数: {data.get('natoms')} | 最小原子间距: {data.get('min_distance')} Å | 真空层已含",
                        f"可用吸附位点: top / bridge / hollow（坐标见 JSON）",
                        f"文件: {data.get('output')}",
                    ]
                    reply_parts.append("\n".join(lines))
                    result_dict["viewer"] = {
                        "formula": data.get("formula"),
                        "cif_path": data.get("output"),
                        "cif": _read_text(data.get("output")),
                        "cell": data.get("cell"),
                        "sites": data.get("sites"),
                        "natoms": data.get("natoms"),
                    }
                elif tool_id == "graph.run" and data.get("run_id"):
                    ctx["last_run_id"] = data.get("run_id")
                    lines = [f"图计算完成: {data.get('state')}  (运行ID: {data.get('run_id')})"]
                    for nid, info in (data.get("nodes") or {}).items():
                        line = f"  节点 {nid}: {info.get('state')}"
                        if info.get("error"):
                            line += f" | {str(info['error'])[:100]}"
                        outs = info.get("outputs") or {}
                        bulky = ("eigenvalues_ev", "k_axis", "k_ticks", "dos_curve")
                        shown = {
                            k: (round(v, 4) if isinstance(v, float) else v)
                            for k, v in outs.items()
                            if not str(k).endswith("stdout") and k not in bulky and not isinstance(v, list)
                        }
                        if shown:
                            line += f" | {shown}"
                        lines.append(line)
                    reply_parts.append("\n".join(lines))
                    chart = _extract_charts(data.get("nodes") or {})
                    if chart:
                        result_dict["chart"] = chart
                    viewer = _viewer_payload_for_material((args.get("inputs") or {}).get("material"))
                    if viewer:
                        result_dict["viewer"] = viewer
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
                    if not analysis:
                        why = data.get("error") or data.get("message") or "结构解析失败（格式或校验问题）"
                        reply_parts.append(f"结构分析失败: {why}")
                    else:
                        pair_distances = analysis.get("pair_distances") or {}
                        min_pair = analysis.get("minimum_distance_pair")
                        min_dist = analysis.get("minimum_distance_angstrom")
                        lines = [f"结构分析: {formula} | atoms={analysis.get('natoms')} species={analysis.get('nspecies')}"]
                        if min_pair and min_dist is not None:
                            lines.append(f"最短距离: {min_dist:.4f} Å ({'-'.join(min_pair)})")
                        for pair, info in pair_distances.items():
                            lines.append(f"{pair}: count={info.get('count')} min={info.get('min_angstrom')} Å max={info.get('max_angstrom')} Å")
                        src = args.get("source") if args.get("source") != "__generated__" else ctx.get("last_structure_source")
                        if src:
                            lines.append(f"文件: {src}")
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
        # LLM narration: replace template-concatenated reply with natural language
        # when a provider is configured; deterministic fallback keeps templates.
        provider = self.llm_planner._get_provider()
        if provider is not None and commands:
            try:
                narrated = self._narrate(provider, message, commands, results, ctx)
            except Exception:
                narrated = ""
            if narrated:
                reply = narrated
        return {
            "reply": reply,
            "commands": commands,
            "results": results,
            "ctx": ctx,
            "planner": "llm" if llm_steps is not None else "rules",
        }

    def _narrate(self, provider, message, commands, results, ctx) -> str:
        """Ask the LLM to summarize executed steps in natural Chinese."""
        steps = []
        for c, r in zip(commands, results):
            data = r.get("json") if isinstance(r.get("json"), dict) else {}
            brief = {"tool": c.get("command"), "desc": c.get("description")}
            if r.get("error"):
                brief["error"] = str(r["error"])[:200]
            elif isinstance(data, dict) and data.get("error"):
                brief["error"] = str(data["error"])[:200]
            else:
                flat = {}
                for k, v in (data or {}).items():
                    if isinstance(v, (int, float, str, bool)) and k not in ("cif", "stdout"):
                        flat[k] = v if not isinstance(v, str) or len(v) < 80 else v[:80] + "…"
                    elif isinstance(v, list) and k in ("species", "output_files"):
                        flat[k] = v[:6]
                brief["out"] = flat
            steps.append(brief)
        system = (
            "你是 DFT-Forge 计算代理的播报员。根据用户请求和已执行的工具步骤，"
            "用自然的中文回复：说明做了什么、关键数值结果（能量/带隙/原子数等）、"
            "或失败原因与建议。铁律：所有数值必须逐字来自执行步骤数据，"
            "禁止用教科书/记忆中的理论值替换计算值（如计算带隙 0.57 eV 就说 0.57，"
            "绝不说成实验值或理论值）；数据里没有的数值一个都不许出现。"
            "不要罗列工具名，结构/图表已自动展示在界面右侧无需赘述。"
            "2-5 句话，直接输出文本。"
        )
        user = f"用户请求: {message}\n执行步骤: {json.dumps(steps, ensure_ascii=False, default=str)[:4000]}"
        out = str(provider.chat(system, user)).strip()
        return out

    def reset_llm(self) -> None:
        """Forget cached LLM provider so new env/config takes effect immediately."""
        self.llm_planner._provider = None
        self.llm_planner._checked = False
