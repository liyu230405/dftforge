"""Intent detection: raw chat message → typed signals for the rule planner.

Vocabulary lives in the data tables below — adding a new keyword or material
alias is a one-line data change; only genuinely new detection LOGIC needs a
new code branch. The rule planner decides what to DO with the signals.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from dft_forge.agent_loop.helpers import extract_formula


@dataclass
class IntentSignals:
    """Everything the rule planner may need, detected up front."""

    material_hint: Optional[str] = None
    pair_types: Optional[str] = None
    formula_hint: Optional[str] = None
    build2d_kind: Optional[str] = None
    dopant_el: Optional[str] = None
    ads_el: Optional[str] = None
    ads_site: Optional[str] = None

    wants_analyze: bool = False
    wants_import: bool = False
    wants_generate: bool = False
    wants_build: bool = False
    wants_submit: bool = False
    wants_status: bool = False
    wants_parse: bool = False
    wants_verify: bool = False
    wants_ledger: bool = False
    wants_doctor: bool = False
    wants_compare: bool = False
    wants_eads: bool = False
    wants_formation: bool = False
    wants_2d: bool = False
    strong_analyze: bool = False

    graph_targets: List[str] = field(default_factory=list)
    has_structure_ctx: bool = False
    has_job_ctx: bool = False


# ── Data tables ───────────────────────────────────────────────────────────────

# 简繁归一：繁体提问（能帶、態密度、導入…）命中简体关键词表
_TRAD_TO_SIMP = str.maketrans({
    "態": "态", "帶": "带", "結": "结", "構": "构", "導": "导", "計": "计",
    "摻": "掺", "對": "对", "較": "较", "區": "区", "驗": "验", "釋": "释",
    "優": "优", "鬆": "松", "馳": "弛", "網": "网", "歷": "历", "錄": "录",
    "診": "诊", "斷": "断", "環": "环", "層": "层", "維": "维", "們": "们",
})

# (规范材料名, 键对类型, 命中关键词, 需词边界) —— 新材料 = 加一行
# 短元素符号（si/al）在英文句子里是常见子串（ba**si**c, **al**so），
# 必须整词匹配；中文关键词（硅/铝）无此问题
_MATERIAL_HINTS: List[tuple] = [
    ("NaCl", "Na-Cl", ("nacl", "氯化钠", "clna"), False),
    ("MgO", "Mg-O", ("mgo", "氧化镁", "gomg"), False),
    ("Si", "Si-Si", ("si", "硅"), True),
    ("Al", "Al-Al", ("al", "铝"), True),
]

_WORD_BOUNDARY = r"(?<![a-z0-9]){}(?![a-z0-9])"

# 简单标志位意图：flag → 关键词表 —— 新词汇/新意图 = 加一行
_INTENT_KEYWORDS: Dict[str, tuple] = {
    "wants_analyze": (
        "键长", "键能", "bond", "distance", "距离", "分析", "analyze", "info",
        "看看", "查看结构", "结构信息", "原子", "atoms", "体积", "volume",
        "晶胞", "cell", "neighbors", "邻居", "结构", "structure",
    ),
    "wants_import": ("导入", "import", "读取", "load", "打开", "open", "加载", "读取结构", "导入结构"),
    "wants_generate": ("生成", "generate", "创建", "create", "做个", "建一个"),
    "wants_build": (
        "构建", "build", "生成", "generate", "输入", "input", "松弛", "relax",
        "vc-relax", "scf", "bands", "dos", "态密度", "带", "计算类型", "in文件",
        "qe输入", "输入文件",
    ),
    "wants_submit": ("提交", "submit", "运行", "run", "执行", "execute", "开始", "提交计算"),
    "wants_status": ("状态", "status", "进度", "progress", "查看", "check", "任务", "job"),
    "wants_parse": ("解析", "parse", "结果", "result", "输出", "output", "读取结果"),
    "wants_verify": ("验证", "verify", "校验", "校验结果"),
    "wants_ledger": ("账本", "ledger", "历史", "history", "record", "查询", "query", "记录", "日志"),
    "wants_doctor": ("环境", "env", "doctor", "诊断", "diagnose", "依赖", "dependencies", "安装", "software"),
    "wants_compare": ("对比", "比较", "区别", "差异", "差距", "compare", " vs ", "vs.", "哪个更"),
    "wants_eads": ("吸附能", "adsorption energy", "eads", "e_ads"),
    "wants_formation": ("形成能", "formation energy", "formation enthalpy", "生成能"),
    "strong_analyze": ("键长", "键能", "bond", "distance", "距离", "分析", "analyze"),
}

# 2D 材料识别 —— 天然单层材料：提到名字即 build2d
_ALWAYS_2D: Dict[str, str] = {
    "石墨烯": "graphene", "graphene": "graphene",
    "氮化硼": "bn", "h-bn": "bn", "hbn": "bn",
}
# 有体相的层状材料（TMD）：需显式 单层/2D/吸附 意图才 build2d
_TMD_2D: Dict[str, str] = {
    "mos2": "mos2", "二硫化钼": "mos2", "ws2": "ws2",
    "mose2": "mose2", "wse2": "wse2", "mote2": "mote2", "wte2": "wte2",
}

# 计算意图：关键词组 → graph 模板 —— 新计算类型 = 加一行
_GRAPH_TARGETS: List[tuple] = [
    (("能带", "带结构", "带隙", "bands", "band"), "t2_bands"),
    (("态密度", "dos"), "t2_dos"),
    (("结构优化", "晶格常数", "弛豫", "松弛", "优化", "vc-relax", "vcrelax", "vc_relax"), "t1_vc_relax"),
]

_GENERIC_CALC_WORDS = ("算", "计算", "跑", "仿真", "simulation")
_2D_WORDS = ("单层", "monolayer", "二维", "2d")


def detect_intents(message: str, ctx: Optional[Dict[str, Any]] = None) -> IntentSignals:
    ctx = ctx or {}
    lower = message.lower().translate(_TRAD_TO_SIMP)
    sig = IntentSignals()

    sig.has_structure_ctx = bool(
        ctx.get("last_structure_file") or ctx.get("last_structure_source") or ctx.get("last_input_file")
    )
    sig.has_job_ctx = bool(ctx.get("last_job_id"))

    for material, pair_types, keys, needs_boundary in _MATERIAL_HINTS:
        for k in keys:
            hit = re.search(_WORD_BOUNDARY.format(re.escape(k)), lower) if needs_boundary else (k in lower)
            if hit:
                sig.material_hint = material
                sig.pair_types = pair_types
                break
        if sig.material_hint:
            break
    sig.formula_hint = sig.material_hint or extract_formula(message)
    if not sig.formula_hint:
        # follow-up like "再算一下它的能带": reuse the session's last material
        sig.formula_hint = ctx.get("last_formula") or ctx.get("last_material")

    # Simple keyword-driven flags
    for flag, keys in _INTENT_KEYWORDS.items():
        setattr(sig, flag, any(k in lower for k in keys))

    # flag-specific exceptions the keyword tables cannot express
    if sig.wants_generate and "build a" in lower:
        sig.wants_generate = False
    if sig.wants_build and sig.wants_generate:
        sig.wants_build = False
    if sig.wants_submit and "计算" in lower and not any(
        k in lower for k in ("提交", "运行", "run", "execute")
    ):
        sig.wants_submit = False
    # comparison intent must be checked BEFORE status/submit keywords so
    # "对比一下任务结果" never falls into a job-status lookup dead end
    if sig.wants_compare:
        sig.wants_status = False
        sig.wants_submit = False
        sig.wants_parse = False

    # ── 2D monolayer / doping / adsorption intents ────────────────────
    # Graphene/h-BN are monolayers by name; TMDs (MoS2...) have bulk
    # forms too, so they need an explicit 单层/2D keyword to build 2D.
    sig.wants_2d = any(k in lower for k in _2D_WORDS)
    for key, kind in _ALWAYS_2D.items():
        if key in lower:
            sig.build2d_kind = kind
            break
    else:
        if re.search(r"\bbn\b", lower):
            sig.build2d_kind = "bn"
        elif sig.wants_2d or "吸附" in message or "adsorb" in lower:
            for key, kind in _TMD_2D.items():
                if key in lower:
                    sig.build2d_kind = kind
                    break

    m = re.search(r"([A-Z][a-z]?)\s*掺杂", message) or re.search(r"掺杂\s*([A-Z][a-z]?)", message)
    if m is None:
        m = re.search(r"([A-Z][a-z]?)[-\s]doped", lower)
    if m is None:
        # Chinese prose often leaves single-letter dopants lowercase: 掺杂n和c
        m = re.search(r"掺杂\s*([a-z])\b", message)
    if m:
        sig.dopant_el = m.group(1)
        if len(sig.dopant_el) == 1:
            sig.dopant_el = sig.dopant_el.upper()

    m = re.search(r"([A-Z][a-z]?[A-Za-z0-9]*)\s*吸附", message) or re.search(r"([a-z0-9]{2,3})\s*(?:on|@)\s*\w+", lower)
    if m is None:
        m = re.search(r"\b(o2|n2|h2|cl2|co|oh|no|h2o|co2|nh3)\s*吸附", lower)
    if m:
        sig.ads_el = m.group(1)
        if "bridge" in lower or "桥" in message:
            sig.ads_site = "bridge"
        elif "hollow" in lower or "洞" in message or "六元环" in message:
            sig.ads_site = "hollow"
        else:
            sig.ads_site = "top"

    # If analyze is requested, do not auto-trigger submit/build unless explicitly requested
    if sig.wants_analyze:
        sig.wants_submit = False
        sig.wants_build = False
        sig.wants_parse = False
        sig.wants_verify = False

    # Full-calculation intent: one-shot graph.run for ANY formula — the
    # engine auto-builds a prototype structure when the material is unknown.
    for keys, template in _GRAPH_TARGETS:
        if any(k in lower for k in keys):
            sig.graph_targets.append(template)
    wants_generic_calc = any(k in lower for k in _GENERIC_CALC_WORDS)
    if (
        not sig.graph_targets
        and wants_generic_calc
        and not sig.strong_analyze
        and not (sig.wants_import or sig.wants_generate or sig.wants_build or sig.wants_status or sig.wants_ledger or sig.wants_doctor)
    ):
        sig.graph_targets.append("t1_vc_relax")

    return sig
