"""Intent detection: raw chat message → typed signals for the rule planner.

Vocabulary lives in the data tables below — adding a new keyword or material
alias is a one-line data change; only genuinely new detection LOGIC needs a
new code branch. The rule planner decides what to DO with the signals.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from dft_forge.agent_loop.helpers import _ELEMENTS, extract_formula


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
    ads_sites: List[str] = field(default_factory=list)
    vacancy_el: Optional[str] = None
    vacancy_charges: List[int] = field(default_factory=list)

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
    wants_vacancy: bool = False
    wants_phonon: bool = False
    wants_magnetism: bool = False
    wants_convergence: bool = False
    wants_chemical_potential: bool = False
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
    "wants_vacancy": ("空位", "vacancy", "缺陷"),
    "wants_phonon": ("声子", "phonon", "dfpt", "虚频", "dynamical stability"),
    "wants_magnetism": ("磁性", "磁矩", "铁磁", "反铁磁", "ferromagnetic", "antiferromagnetic", "spin-polarized"),
    "wants_convergence": ("收敛测试", "收敛扫描", "收敛性", "convergence test", "convergence study", "convergence scan", "ecutwfc", "截断能", "k 点收敛"),
    "wants_chemical_potential": ("化学势", "chemical potential", "稳定区间", "convex hull", "相图"),
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


def _ordered_adsorption_sites(message: str) -> List[str]:
    """Return every explicitly requested adsorption site in message order."""
    text = str(message or "")
    lower = text.lower()
    hits: List[tuple[int, str]] = []
    patterns = {
        "top": (r"(?<![a-z])top(?![a-z])", r"顶位", r"顶点位"),
        "bridge": (r"(?<![a-z])bridge(?![a-z])", r"桥位", r"桥接位"),
        "hollow": (r"(?<![a-z])hollow(?![a-z])", r"空心位", r"洞位", r"六元环"),
    }
    for site, site_patterns in patterns.items():
        for pattern in site_patterns:
            for match in re.finditer(pattern, lower):
                hits.append((match.start(), site))
    sites: List[str] = []
    for _, site in sorted(hits):
        if site not in sites:
            sites.append(site)
    all_three = any(key in lower for key in (
        "三个位点", "三种位点", "三个吸附位点", "三种吸附位点",
        "three adsorption sites", "all three sites",
    ))
    generic_multi = any(key in lower for key in (
        "多位点", "不同位点", "multiple sites", "different sites",
    ))
    if all_three or (generic_multi and not sites):
        for site in ("top", "bridge", "hollow"):
            if site not in sites:
                sites.append(site)
    return sites


_CHARGE_MARKER = re.compile(
    r"(?:电荷(?:态)?|charge(?:\s*states?)?|(?<![A-Za-z])q(?![A-Za-z]))",
    re.IGNORECASE,
)
_CHARGE_LIST = re.compile(
    r"\s*(?:=|:|为|是)?\s*"
    r"([+−＋－-]?\s*\d+"
    r"(?:\s*(?:[/,，、;；]|和|与|\band\b|\bor\b)\s*[+−＋－-]?\s*\d+)*)",
    re.IGNORECASE,
)
_CHARGE_TOKEN = re.compile(r"([+−＋－-]?)\s*(\d+)")
_SIGNED_CHARGE = re.compile(r"(?<![\w.])([+−＋－-])\s*(\d+)(?![\d.])")
_SIGNED_CHARGE_SEQUENCE = re.compile(
    r"[+−＋－-]\s*\d+"
    r"(?:\s*(?:[/,，、;；]|和|与|\band\b|\bor\b)\s*[+−＋－-]\s*\d+)+",
    re.IGNORECASE,
)
_NON_CHARGE_QUANTITY = re.compile(
    r"(?:\beV\b|\bRy\b|化学势|chemical\s+potential|费米|fermi|势对齐|alignment|介电|dielectric)",
    re.IGNORECASE,
)


def _explicit_vacancy_charges(message: str) -> List[int]:
    """Extract explicit charge states without treating unsigned prose numbers as charge."""
    text = str(message or "")
    positioned: List[tuple[int, int]] = []
    for marker in _CHARGE_MARKER.finditer(text):
        match = _CHARGE_LIST.match(text, marker.end())
        if not match:
            continue
        for token in _CHARGE_TOKEN.finditer(match.group(1)):
            sign, magnitude = token.groups()
            value = int(magnitude)
            if sign in ("-", "−", "－"):
                value = -value
            positioned.append((match.start(1) + token.start(), value))
    # Compact charge-state lists often omit a label: "+1/+2/-1 vacancy".
    # Do not reinterpret signed chemical potentials, Fermi levels, alignment
    # values, or dielectric observations as defect charge states.
    covered: List[tuple[int, int]] = []
    for sequence in _SIGNED_CHARGE_SEQUENCE.finditer(text):
        covered.append(sequence.span())
        for token in _SIGNED_CHARGE.finditer(sequence.group(0)):
            sign, magnitude = token.groups()
            value = int(magnitude)
            if sign in ("-", "−", "－"):
                value = -value
            positioned.append((sequence.start() + token.start(), value))
    for match in _SIGNED_CHARGE.finditer(text):
        if any(start <= match.start() < end for start, end in covered):
            continue
        nearby = text[max(0, match.start() - 40):match.end() + 40]
        if not re.search(r"空位|vacancy|缺陷", nearby, re.IGNORECASE):
            continue
        before = text[max(0, match.start() - 40):match.start()]
        after = text[match.end():match.end() + 8]
        if _NON_CHARGE_QUANTITY.search(before) or re.match(r"\s*(?:eV|Ry)\b", after, re.IGNORECASE):
            continue
        sign, magnitude = match.groups()
        value = int(magnitude)
        if sign in ("-", "−", "－"):
            value = -value
        positioned.append((match.start(), value))
    charges: List[int] = []
    for _, value in sorted(positioned):
        if value not in charges:
            charges.append(value)
    return charges


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

    vacancy_match = re.search(
        r"(?<![A-Za-z])([A-Z][a-z]?)(?:原子)?\s*(?:空位|vacancy)",
        message or "",
        re.IGNORECASE,
    )
    if vacancy_match:
        symbol = vacancy_match.group(1)
        sig.vacancy_el = symbol[:1].upper() + symbol[1:].lower()
    if sig.wants_vacancy and sig.wants_formation:
        sig.vacancy_charges = _explicit_vacancy_charges(message)

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
        elif sig.wants_2d or "吸附" in message or "adsorp" in lower:
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

    m = re.search(r"(?<![A-Za-z])([A-Z][a-z]?[A-Za-z0-9]*)\s*吸附", message)
    if m is None:
        # Chinese requests also commonly put the adsorbate after the verb:
        # "石墨烯吸附N的吸附能".  Do not substitute a default element.
        m = re.search(r"吸附\s*([A-Z][a-z]?[A-Za-z0-9]*)", message)
    if m is None:
        m = re.search(r"(?<![A-Za-z])([A-Z][a-z]?)\s*(?:atom\s*)?adsorp", message, re.IGNORECASE)
    if m is None and sig.wants_eads:
        m = re.search(r"(?<![A-Za-z])([A-Z][a-z]?)\s*(?:原子\s*)?(?:在|于|on\b|@)", message, re.IGNORECASE)
    if m is None:
        m = re.search(r"\b([a-z][a-z0-9]{0,2})\b\s*(?:on|@)\s*\w+", lower)
    if m is None:
        m = re.search(r"\b(o2|n2|h2|cl2|co|oh|no|h2o|co2|nh3)\s*吸附", lower)
    if m is None and sig.wants_eads:
        # The species may be specified as a labelled input rather than next
        # to the verb ("吸附物种为 N", "吸附能（O 原子）").
        m = re.search(r"吸附物种\s*(?:是|为|:|：|=)?\s*([A-Z][a-z]?)\b", message)
        if m is None:
            m = re.search(r"吸附能\s*[（(]\s*([A-Z][a-z]?)\s*原子", message)
    if m:
        sig.ads_el = m.group(1)
        # Uppercase CO/NO/OH are molecular adsorbates, not Co/No or an atom.
        molecular = {"CO", "NO", "OH", "H2O", "CO2", "NH3", "O2", "N2", "H2", "F2", "CL2"}
        if sig.ads_el.upper() in molecular and sig.ads_el not in _ELEMENTS:
            sig.ads_el = sig.ads_el.upper()
        elif len(sig.ads_el) <= 2:
            sig.ads_el = sig.ads_el[:1].upper() + sig.ads_el[1:].lower()
        sig.ads_sites = _ordered_adsorption_sites(message)
        if not sig.ads_sites:
            sig.ads_sites = ["top"]
        sig.ads_site = sig.ads_sites[0]

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
