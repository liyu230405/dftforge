"""Agent loop backend for chat-driven multi-step tool execution."""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

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


def _viewer_payload_from_file(path: Any) -> Optional[Dict[str, Any]]:
    """CIF viewer payload from a structure file path."""
    cif = _read_text(path)
    if not cif:
        return None
    natoms = None
    formula = None
    try:
        from ase.io import read as ase_read

        atoms = ase_read(str(path))
        natoms = len(atoms)
        formula = atoms.get_chemical_formula()
    except Exception:
        pass
    return {
        "formula": formula or Path(str(path)).stem,
        "cif": cif,
        "natoms": natoms,
        "source": str(path),
    }


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


def _dedupe_node_log(log: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collapse a node event stream to one entry per node (latest state, first-seen order)."""
    latest: Dict[str, Any] = {}
    order: List[str] = []
    for e in log:
        n = e.get("node")
        if n not in latest:
            order.append(n)
        latest[n] = e.get("state")
    return [{"node": n, "state": latest[n]} for n in order]


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

    def plan(self, message: str, ctx: Dict[str, Any], history: Optional[List[Dict[str, Any]]] = None):
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
  (e.g. CaTiO3, SrTiO3, GaAs, MoS2). Unknown formulas are auto-built into
  prototype structures — NEVER ask the user to import a file for a plain
  bulk crystal. Bulk MoS2/WS2/MoSe2/WSe2 build the real 2H layered crystal.
- For a full verified calculation prefer graph.run with a template_id
  (t1_vc_relax = structure optimization, t2_bands = band structure, t2_dos = density of states).
  "算 X 的能带" → one graph.run t2_bands step; combining several targets emits one step each.
- 2D monolayers (单层/monolayer/二维/2D intent, or 石墨烯/graphene, 氮化硼/h-BN,
  MoS2/WS2/MoSe2/WSe2 单层): FIRST structure.build2d with kind 'graphene'|'bn'|
  'mos2'|'ws2'|'mose2'|'wse2' plus supercell/dopants/adsorb as requested,
  THEN graph.run with template_id and empty inputs — omit inputs.material so the
  built structure chains automatically. Kind matches the material named.
- Bulk doping ("P掺杂Si", "B-doped diamond", X掺杂Y): structure.dope
  with source (host material/formula), element (dopant), supercell '2x2x2'
  (default; use '3x3x3' for lower concentration), then graph.run with
  empty inputs — the doped structure chains automatically.
- 2D doping/adsorption (N掺杂石墨烯, O吸附在MoS2): structure.build2d with
  dopants/adsorb args (supercell '3x3' typical for doping), then graph.run
  with empty inputs.
- Site scan ("different sites/positions"): emit one structure.build2d step per site
  (top/bridge/hollow), each with its own output path.
- Chit-chat / questions about you: empty steps, answer in "reply".
- Use the conversation history for context: pronouns like "它/这个/再算一次/继续"
  refer to the material or calculation mentioned earlier.
- descriptions in Chinese. At most 8 steps.

Tools:
{self._catalog()}"""
        ctx_summary = {k: ctx.get(k) for k in ("last_structure_source", "last_formula", "last_input_file", "last_job_id") if ctx.get(k)}
        user = f"Context: {json.dumps(ctx_summary, ensure_ascii=False)}"
        if history:
            lines = []
            for h in history[-8:]:
                who = "用户" if h.get("role") == "user" else "助手"
                text = str(h.get("text", ""))[:200]
                if text:
                    lines.append(f"{who}: {text}")
            if lines:
                user += "\n对话历史（旧→新）:\n" + "\n".join(lines)
        user += f"\nUser: {message}"
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
        if not formula_hint:
            # follow-up like "再算一下它的能带": reuse the session's last material
            formula_hint = ctx.get("last_formula") or ctx.get("last_material")

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

        # ── 2D monolayer / doping / adsorption intents ────────────────────
        # Graphene/h-BN are monolayers by name; TMDs (MoS2...) have bulk
        # forms too, so they need an explicit 单层/2D keyword to build 2D.
        _tmd_map = {
            "mos2": "mos2", "二硫化钼": "mos2", "ws2": "ws2",
            "mose2": "mose2", "wse2": "wse2", "mote2": "mote2", "wte2": "wte2",
        }
        wants_2d = any(k in lower for k in ("单层", "monolayer", "二维", "2d"))
        build2d_kind = None
        if any(k in lower for k in ("石墨烯", "graphene")):
            build2d_kind = "graphene"
        elif any(k in lower for k in ("氮化硼", "h-bn", "hbn")) or re.search(r"\bbn\b", lower):
            build2d_kind = "bn"
        elif wants_2d or "吸附" in message or "adsorb" in lower:
            for key, kind in _tmd_map.items():
                if key in lower:
                    build2d_kind = kind
                    break

        dopant_el = None
        m = re.search(r"([A-Z][a-z]?)\s*掺杂", message) or re.search(r"掺杂\s*([A-Z][a-z]?)", message)
        if m is None:
            m = re.search(r"([A-Z][a-z]?)[-\s]doped", lower)
        if m:
            dopant_el = m.group(1)

        ads_el = None
        ads_site = None
        m = re.search(r"([A-Z][a-z]?)\s*吸附", message)
        if m:
            ads_el = m.group(1)
            if "bridge" in lower or "桥" in message:
                ads_site = "bridge"
            elif "hollow" in lower or "洞" in message or "六元环" in message:
                ads_site = "hollow"
            else:
                ads_site = "top"

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

        # ── 2D / doped structures: build first, then a chained graph.run ──
        # (the execution loop injects the built CIF into graph.run steps
        #  that name no material)
        if (
            (build2d_kind or dopant_el)
            and not strong_analyze
            and not wants_import
            and not wants_generate
            and not (wants_analyze and not graph_targets)
        ):
            build_steps: List[Dict[str, Any]] = []
            if build2d_kind:
                bargs: Dict[str, Any] = {"kind": build2d_kind}
                if dopant_el or ads_el:
                    bargs["supercell"] = "3x3"
                if dopant_el:
                    bargs["dopants"] = [{"index": 0, "element": dopant_el}]
                if ads_el:
                    bargs["adsorb"] = {"element": ads_el, "site": ads_site or "top"}
                kind_label = {"graphene": "石墨烯", "bn": "h-BN", "mos2": "MoS2", "ws2": "WS2",
                              "mose2": "MoSe2", "wse2": "WSe2"}.get(build2d_kind, build2d_kind)
                mods = []
                if dopant_el:
                    mods.append(f"{dopant_el}掺杂")
                if ads_el:
                    mods.append(f"{ads_el}吸附({ads_site})")
                desc = f"构建{kind_label}单层" + ("".join(mods) if mods else "")
                build_steps.append({"tool": "structure.build2d", "args": bargs, "description": desc})
            elif dopant_el:
                host = formula_hint or material_hint
                if host:
                    build_steps.append({
                        "tool": "structure.dope",
                        "args": {"source": host, "element": dopant_el, "supercell": "2x2x2"},
                        "description": f"构建{dopant_el}掺杂{host}超胞",
                    })
            if build_steps:
                if graph_targets:
                    label = {"t2_bands": "能带结构", "t2_dos": "态密度", "t1_vc_relax": "结构优化"}
                    name = build2d_kind or f"{dopant_el}掺杂{formula_hint or ''}"
                    for tid in graph_targets:
                        build_steps.append({
                            "tool": "graph.run",
                            "args": {"template_id": tid, "inputs": {}},
                            "description": f"{name} {label[tid]}",
                        })
                return build_steps

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
                if build2d_kind:
                    kind_label = {"graphene": "石墨烯", "bn": "h-BN", "mos2": "MoS2", "ws2": "WS2",
                                  "mose2": "MoSe2", "wse2": "WSe2"}.get(build2d_kind, build2d_kind)
                    steps.append({
                        "tool": "structure.build2d",
                        "args": {"kind": build2d_kind},
                        "description": f"构建{kind_label}单层",
                    })
                    source = "__generated__"
                elif wants_generate or material_hint or formula_hint:
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
            if build2d_kind:
                kind_label = {"graphene": "石墨烯", "bn": "h-BN", "mos2": "MoS2", "ws2": "WS2",
                              "mose2": "MoSe2", "wse2": "WSe2"}.get(build2d_kind, build2d_kind)
                steps.append({
                    "tool": "structure.build2d",
                    "args": {"kind": build2d_kind},
                    "description": f"构建{kind_label}单层",
                })
            else:
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

    async def run(
        self,
        message: str,
        session_dir: Path,
        ctx: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, Any]]] = None,
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
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
        llm_steps, llm_reply = await asyncio.to_thread(self.llm_planner.plan, message, ctx, history)
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
        steps = llm_steps if llm_steps is not None else self.plan(message, ctx)
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
        # structure built earlier in this plan (build2d/dope/generate) —
        # auto-chained into a following graph.run that names no material
        built_structure: Optional[str] = None

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
                chain.append({"tool": "help", "description": description, "status": "done", "summary": "能力说明", "nodes": []})
                emit({"type": "step", "index": idx, "status": "done", "summary": "能力说明"})
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
            elif tool_id == "structure.dope":
                src = re.sub(r"[^A-Za-z0-9]", "_", str(args.get("source", "mat")))[:20]
                el = re.sub(r"[^A-Za-z0-9]", "", str(args.get("element", "X")))[:3]
                args["output"] = str(session_dir / f"doped_{src}_{el}.cif")
            elif tool_id == "structure.analyze" and str(args.get("output", "")).endswith(".json"):
                args["output"] = str(session_dir / "structure_analysis.json")
            elif tool_id == "structure.analyze" and args.get("source") == "__generated__":
                args["source"] = ctx.get("last_structure_source") or args.get("source")
            elif tool_id == "input.build":
                args["output"] = str(session_dir / "input.in")
                if not args.get("structure") and built_structure:
                    args["structure"] = built_structure
            elif tool_id == "graph.run":
                if not str(args.get("workdir") or "").strip():
                    args["workdir"] = str(session_dir / "graphs")
                inputs = dict(args.get("inputs") or {})
                if built_structure and not inputs.get("structure") and not inputs.get("material"):
                    # chain the structure built by an earlier step in this plan
                    inputs["structure"] = built_structure
                    args["inputs"] = inputs

                def _node_cb(ev, _idx=idx, _log=node_log):
                    _log.append({"node": ev.get("node"), "state": ev.get("state")})
                    emit({"type": "node", "step": _idx, "node": ev.get("node"), "state": ev.get("state")})

                args["_on_event"] = _node_cb
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
                    built_structure = data.get("output")
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
                    built_structure = data.get("output")
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
                elif tool_id == "structure.dope" and data.get("output"):
                    ctx["last_structure_source"] = data.get("output")
                    ctx["last_structure_file"] = data.get("output")
                    ctx["last_formula"] = data.get("formula")
                    built_structure = data.get("output")
                    reply_parts.append(
                        f"已构建掺杂结构: {data.get('formula')} — {data.get('description')}\n"
                        f"原子数: {data.get('natoms')} | 最小原子间距: {data.get('min_distance')} Å\n"
                        f"文件: {data.get('output')}"
                    )
                    result_dict["viewer"] = {
                        "formula": data.get("formula"),
                        "cif_path": data.get("output"),
                        "cif": _read_text(data.get("output")),
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
                    _inputs = args.get("inputs") or {}
                    viewer = _viewer_payload_for_material(_inputs.get("material"))
                    if viewer is None and _inputs.get("structure"):
                        viewer = _viewer_payload_from_file(_inputs["structure"])
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
        provider = self.llm_planner._get_provider()
        if provider is not None and commands:
            emit({"type": "phase", "phase": "narrating", "label": "整理计算结果…"})
            try:
                narrated = await asyncio.to_thread(self._narrate, provider, message, commands, results, ctx)
            except Exception:
                narrated = ""
            if narrated:
                reply = narrated
        emit({"type": "reply", "text": reply})
        return {
            "reply": reply,
            "commands": commands,
            "results": results,
            "chain": chain,
            "ctx": ctx,
            "planner": planner_name,
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
