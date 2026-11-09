"""Check evaluation: one run record + one check -> (score, detail).

Every rule name emitted by gen_questions.py is implemented here. A rule that
cannot find the evidence it needs scores 0 and says why in `detail` — never a
silent pass. Regex-based assertion rules follow SCORING.md §2.3:
宁可漏判不可误判 — patterns are written wide, human spot-checks catch misses.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

import scoring
from audit import audit_run, hard_gating_failed
from effects import violates_plan_only
from extract import convergence_reference, extract

Result = Tuple[float, str]


# ---------------------------------------------------------------- text views


def narrative(record: Dict[str, Any]) -> str:
    """Reply + advisory_steps + blocking_reasons + missing_capabilities + notes."""
    plan = record.get("plan") or {}
    parts: List[str] = [str(record.get("reply") or ""), str(plan.get("reply") or "")]
    for key in ("advisory_steps", "blocking_reasons", "missing_capabilities"):
        val = plan.get(key)
        if isinstance(val, list):
            parts.extend(json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list))
                         else str(v) for v in val)
        elif val:
            parts.append(str(val))
    for node in record.get("chain") or []:
        parts.append(str(node.get("summary") or ""))
    return "\n".join(p for p in parts if p)


def planned_tools(record: Dict[str, Any]) -> List[str]:
    tools = [str(a.get("tool")) for a in (record.get("plan") or {}).get("actions") or []
             if a.get("tool")]
    tools += [str(c.get("tool")) for c in record.get("commands") or [] if c.get("tool")]
    return tools


def planned_actions(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = [a for a in (record.get("plan") or {}).get("actions") or [] if isinstance(a, dict)]
    out += [{"tool": c.get("tool"), "args": c.get("args") or {}}
            for c in record.get("commands") or []]
    return out


def _succeeded_nodes(record: Dict[str, Any]) -> List[str]:
    """Names of graph nodes / steps that finished successfully."""
    names: List[str] = []
    for node in record.get("chain") or []:
        if str(node.get("status") or "").lower() in ("succeeded", "success", "ok", "done"):
            names.append(str(node.get("step_id") or node.get("tool") or ""))
        for sub in node.get("nodes") or []:
            if not isinstance(sub, dict):
                continue
            state = str(sub.get("state") or sub.get("status") or "").lower()
            if state in ("succeeded", "success", "ok", "done"):
                names.append(str(sub.get("name") or sub.get("id") or sub.get("node") or ""))
    return [n for n in names if n]


def _node_matches(names: List[str], wanted: str) -> bool:
    w = wanted.lower().replace("-", "_")
    return any(w in n.lower().replace("-", "_") for n in names)


def _args_contains(actual: Any, expected: Any) -> bool:
    """Recursive containment: expected is a sub-structure of actual.

    Scalars compare case-insensitively as strings, and "3x3" matches [3,3,1].
    """
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        lower = {str(k).lower(): v for k, v in actual.items()}
        for key, val in expected.items():
            k = str(key).lower()
            if k not in lower:
                # tolerate common aliases for the same slot
                aliases = {"kind": ("material", "system", "structure_kind"),
                           "supercell": ("cell", "repeat", "size"),
                           "adsorbate_element": ("adsorbate", "element", "species"),
                           "orders": ("magnetic_orders", "configurations"),
                           "sites": ("site", "site_list")}
                hit = next((a for a in aliases.get(k, ()) if a in lower), None)
                if hit is None:
                    return False
                k = hit
            if not _args_contains(lower[k], val):
                return False
        return True
    if isinstance(expected, list):
        if isinstance(actual, str):
            return all(_scalar_in_text(item, actual) for item in expected)
        if not isinstance(actual, list):
            return False
        return all(any(_args_contains(a, item) for a in actual) for item in expected)
    return _scalar_eq(actual, expected)


def _scalar_in_text(expected: Any, text: str) -> bool:
    return str(expected).strip().lower() in text.strip().lower()


def _scalar_eq(actual: Any, expected: Any) -> bool:
    if isinstance(actual, list):
        # "3x3" vs [3, 3, 1]
        digits = re.findall(r"\d+", str(expected))
        if digits:
            got = [str(int(x)) for x in actual if isinstance(x, (int, float))]
            return got[:len(digits)] == digits
        return any(_scalar_eq(a, expected) for a in actual)
    a, e = str(actual).strip().lower(), str(expected).strip().lower()
    if a == e:
        return True
    try:
        return abs(float(a) - float(e)) < 1e-9
    except ValueError:
        return e in a


# ---------------------------------------------------------------- tool_trace


def _r_no_side_effect_calls(record, args) -> Result:
    bad = [c.get("tool") for c in record.get("commands") or []
           if violates_plan_only(c.get("tool"))]
    if bad:
        return 0.0, f"执行了非只读工具: {bad}"
    return 1.0, "无非只读调用"


def _r_tool_planned(record, args) -> Result:
    tool = args["tool"]
    hits = [a for a in planned_actions(record) if str(a.get("tool")) == tool]
    if not hits:
        return 0.0, f"计划里没有 {tool}"
    expected = args.get("args_contains")
    if not expected:
        return 1.0, f"计划含 {tool}"
    for a in hits:
        if _args_contains(a.get("args") or {}, expected):
            return 1.0, f"{tool} 参数匹配 {expected}"
    got = [a.get("args") for a in hits]
    return 0.0, f"{tool} 参数不含 {expected}; 实际 {got}"


def _r_tool_planned_any(record, args) -> Result:
    tools = set(args["tools"])
    hit = [t for t in planned_tools(record) if t in tools]
    return (1.0, f"命中 {hit}") if hit else (0.0, f"缺少 {sorted(tools)} 中任意一个")


def _r_tool_not_planned(record, args) -> Result:
    tool = args["tool"]
    if tool in planned_tools(record):
        return 0.0, f"出现了禁止的工具 {tool}"
    return 1.0, f"未出现 {tool}"


def _r_template_used(record, args) -> Result:
    want = str(args["template_id"]).lower()
    for a in planned_actions(record):
        if str(a.get("tool")) != "graph.run":
            continue
        blob = json.dumps(a.get("args") or {}, ensure_ascii=False).lower()
        if want in blob:
            return 1.0, f"graph.run 使用 {want}"
    blob = json.dumps(record.get("commands") or [], ensure_ascii=False).lower()
    if want in blob:
        return 1.0, f"命令里出现模板 {want}"
    return 0.0, f"未使用模板 {want}"


def _r_nodes_succeeded(record, args) -> Result:
    names = _succeeded_nodes(record)
    missing = [n for n in args["nodes"] if not _node_matches(names, n)]
    if missing:
        return 0.0, f"节点未成功: {missing}; 成功的={names}"
    return 1.0, f"节点全部成功: {args['nodes']}"


def _r_node_count(record, args) -> Result:
    n = len(_succeeded_nodes(record))
    need = int(args["minimum"])
    return (1.0, f"成功节点 {n} ≥ {need}") if n >= need else (0.0, f"成功节点仅 {n} < {need}")


def _r_no_self_doping(record, args) -> Result:
    for a in planned_actions(record):
        if str(a.get("tool")) != "structure.dope":
            continue
        d = a.get("args") or {}
        el = str(d.get("element") or d.get("dopant") or "").strip().capitalize()
        host = str(d.get("material") or d.get("base") or d.get("structure") or
                   d.get("kind") or "")
        if el and re.search(rf"(?<![A-Za-z]){re.escape(el)}(?![a-z])", host):
            return 0.0, f"自掺杂: 在 {host} 里掺 {el}"
    return 1.0, "无自掺杂"


def _r_energy_steps_bound(record, args) -> Result:
    tool = args["tool"]
    hits = [a for a in planned_actions(record) if str(a.get("tool")) == tool]
    if not hits:
        return 0.0, f"计划里没有 {tool}"
    for a in hits:
        blob = a.get("args") or {}
        text = json.dumps(blob, ensure_ascii=False)
        # explicit binding: step ids / step refs, not bare literal numbers
        if re.search(r"step|from_step|ref|\$\{|node", text, re.IGNORECASE):
            return 1.0, f"{tool} 能量来源绑定到步骤"
    return 0.0, f"{tool} 的能量未绑定真实步骤 (可能是硬编码数值): {[a.get('args') for a in hits]}"


def _r_energy_sources_count(record, args) -> Result:
    need = int(args["minimum"])
    best = 0
    for a in planned_actions(record):
        if str(a.get("tool")) not in ("thermo.formation", "thermo.eads"):
            continue
        text = json.dumps(a.get("args") or {}, ensure_ascii=False)
        refs = set(re.findall(r"(?:step[_-]?id\"?\s*:\s*\"?|\$\{|step[_-])([A-Za-z0-9_.-]+)", text))
        best = max(best, len(refs))
    if best >= need:
        return 1.0, f"能量来源 {best} 个 ≥ {need}"
    return 0.0, f"能量来源仅 {best} 个 < {need}（形成能需要化合物+各单质参考）"


def _r_no_molecule_split(record, args) -> Result:
    mols = []
    for a in planned_actions(record):
        if str(a.get("tool")) != "structure.molecule":
            continue
        d = a.get("args") or {}
        mols.append(str(d.get("formula") or d.get("molecule") or d.get("name") or ""))
    single = [m for m in mols if re.fullmatch(r"\s*(C|O|H|N)\s*", m or "")]
    if single:
        return 0.0, f"把分子拆成了单原子: {single}"
    return 1.0, "未拆分分子"


def _r_no_qe_phonon_template(record, args) -> Result:
    blob = json.dumps(planned_actions(record), ensure_ascii=False).lower()
    if re.search(r"t\d+_(?:phonon|ph)\b|template.{0,12}phonon", blob):
        return 0.0, "编造了不存在的 QE 声子 JSON 模板"
    return 1.0, "未编造 QE 声子模板"


def _r_q_grid_not_gamma_only(record, args) -> Result:
    for a in planned_actions(record):
        d = a.get("args") or {}
        for key in ("nq", "q_grid", "qpoints", "nq1", "q_mesh"):
            if key in d:
                nums = re.findall(r"\d+", json.dumps(d[key]))
                vals = [int(x) for x in nums]
                if vals and any(v > 1 for v in vals):
                    return 1.0, f"q 网格 {d[key]} 非仅 Γ"
                return 0.0, f"q 网格仅 Γ: {d[key]}"
    text = narrative(record)
    if re.search(r"q\s*网格|[2-9]\s*[x×]\s*[2-9]\s*[x×]\s*[2-9]|nq", text):
        return 1.0, "叙述中给出了非 Γ q 网格"
    return 0.0, "未给出 q 网格（声子色散不能只算 Γ）"


# ---------------------------------------------------------------- numeric


def _need(record, args) -> Tuple[Optional[Any], str]:
    field = args["field"]
    val = extract(record, field)
    if val is None:
        return None, f"未提取到 {field}"
    return val, field


def _r_graded_relative(record, args) -> Result:
    val, field = _need(record, args)
    if val is None:
        return 0.0, field
    if args.get("value") is None:
        return 0.0, f"{field}={val}，目标值尚未冻结（等 freeze_gt）"
    target, tol = float(args["value"]), float(args["tolerance"])
    if target == 0:
        return 0.0, f"{field}: target 为 0，无法做相对误差"
    err = abs(float(val) - target) / abs(target)
    s = scoring.graded_score(err, tol)
    return s, f"{field}={val} 目标={target} 相对误差={err:.4g} tol={tol} → {s:.3f}"


def _r_graded_absolute(record, args) -> Result:
    val, field = _need(record, args)
    if val is None:
        return 0.0, field
    if args.get("value") is None:
        return 0.0, f"{field}={val}，目标值尚未冻结（等 freeze_gt）"
    target, tol = float(args["value"]), float(args["tolerance"])
    err = abs(float(val) - target)
    s = scoring.graded_score(err, tol)
    return s, f"{field}={val} 目标={target} 绝对误差={err:.4g} tol={tol} → {s:.3f}"


def _r_exact(record, args) -> Result:
    val, field = _need(record, args)
    if val is None:
        return 0.0, field
    want = args["value"]
    ok = _scalar_eq(val, want)
    return (1.0 if ok else 0.0), f"{field}={val} 期望={want}"


def _r_upper_bound(record, args) -> Result:
    val, field = _need(record, args)
    if val is None:
        return 0.0, field
    limit = float(args["limit"])
    s = scoring.upper_bound_score(abs(float(val)), limit)
    return s, f"{field}={val} 上限={limit} → {s:.3f}"


def _r_lower_bound(record, args) -> Result:
    val, field = _need(record, args)
    if val is None:
        return 0.0, field
    limit = float(args["limit"])
    s = scoring.lower_bound_score(float(val), limit)
    return s, f"{field}={val} 下限={limit} → {s:.3f}"


def _r_sign_negative(record, args) -> Result:
    val, field = _need(record, args)
    if val is None:
        return 0.0, field
    return (1.0 if float(val) < 0 else 0.0), f"{field}={val} 需为负"


def _r_in_range(record, args) -> Result:
    val, field = _need(record, args)
    if val is None:
        return 0.0, field
    low, high = float(args["low"]), float(args["high"])
    ok = low <= float(val) <= high
    return (1.0 if ok else 0.0), f"{field}={val} 区间=[{low}, {high}]"


def _r_finite(record, args) -> Result:
    val, field = _need(record, args)
    if val is None:
        return 0.0, field
    try:
        f = float(val)
    except (TypeError, ValueError):
        return 0.0, f"{field}={val} 非数值"
    if f != f or abs(f) == float("inf"):
        return 0.0, f"{field}={val} 非有限值"
    return 1.0, f"{field}={f} 为有限实数"


def _r_nonzero(record, args) -> Result:
    val, field = _need(record, args)
    if val is None:
        return 0.0, field
    ok = abs(float(val)) > 1e-9
    return (1.0 if ok else 0.0), f"{field}={val} 需非零"


def _r_acoustic_modes_near_zero(record, args) -> Result:
    freqs = extract(record, args["field"])
    if not freqs:
        return 0.0, f"未提取到 {args['field']}"
    count, limit = int(args["count"]), float(args["limit"])
    acoustic = sorted(freqs, key=abs)[:count]
    if len(acoustic) < count:
        return 0.0, f"模式数不足: {freqs}"
    worst = max(abs(f) for f in acoustic)
    s = scoring.upper_bound_score(worst, limit)
    return s, f"声学支 {acoustic} 最大 |ω|={worst:.4g} 限={limit} → {s:.3f}"


def _r_lowest_energy_order(record, args) -> Result:
    from extract import _energy_by_order, _lowest_energy_order  # noqa: PLC0415
    got = _lowest_energy_order(record)
    want = str(args["expected"]).lower()
    if got is None:
        return 0.0, "未能判定基态磁序"
    table = _energy_by_order(record)
    ok = got == want
    return (1.0 if ok else 0.0), f"基态磁序={got} 期望={want} 能量表={table}"


def _r_selected_within_threshold(record, args) -> Result:
    val = extract(record, args["field"])
    if val is None:
        return 0.0, f"未提取到 {args['field']}（收敛扫描没有给出结论）"
    thr = float(args["threshold_mev_per_atom"])
    text = narrative(record)
    if re.search(rf"{thr:g}\s*meV|meV\s*/\s*atom|meV/原子", text, re.IGNORECASE):
        return 1.0, f"选定 {val}，并声明了 {thr} meV/atom 判据"
    return 0.5, f"选定 {val}，但未显式声明 {thr} meV/atom 判据"


# ---------------------------------------------------------------- assertion


def _compile(patterns: List[str]) -> List[re.Pattern]:
    return [re.compile(p, re.IGNORECASE | re.DOTALL) for p in patterns]


def _r_must_match(record, args) -> Result:
    text = narrative(record)
    pats = _compile(args["patterns"])
    hits = [p.pattern for p in pats if p.search(text)]
    if args.get("all"):
        missing = [p.pattern for p in pats if not p.search(text)]
        return (1.0, "全部命中") if not missing else (0.0, f"缺失: {missing}")
    return (1.0, f"命中 {hits[0]}") if hits else (0.0, f"无一命中: {[p.pattern for p in pats]}")


def _r_must_not_match(record, args) -> Result:
    text = narrative(record)
    bad = [p.pattern for p in _compile(args["patterns"]) if p.search(text)]
    return (0.0, f"出现了禁止表述: {bad}") if bad else (1.0, "无禁止表述")


def _r_support_status_in(record, args) -> Result:
    got = str((record.get("plan") or {}).get("support_status") or "").lower()
    allowed = [str(x).lower() for x in args["allowed"]]
    ok = got in allowed
    return (1.0 if ok else 0.0), f"support_status={got} 允许={allowed}"


def _r_flag_set(record, args) -> Result:
    flag = args["flag"]
    plan = record.get("plan") or {}
    if plan.get(flag):
        return 1.0, f"{flag} 已置位"
    if re.search(re.escape(flag), narrative(record), re.IGNORECASE):
        return 0.5, f"{flag} 未置位，但叙述里提到了"
    return 0.0, f"{flag} 未置位"


_ALT_RE = re.compile(
    r"导入|上传|自带|改用|换用|替代|拆分|分批|外部|手动提供|提供结构|提供赝势|"
    r"import|upload|alternativ|instead|workaround", re.IGNORECASE)


def _r_alternative_given(record, args) -> Result:
    plan = record.get("plan") or {}
    adv = plan.get("advisory_steps") or []
    text = narrative(record)
    if not adv and not _ALT_RE.search(text):
        return 0.0, "没有 advisory，也没有替代路线关键词"
    if _ALT_RE.search(text):
        return 1.0, "给出了替代路线"
    return 0.5, "有 advisory 但未识别到替代路线关键词（建议人工复核）"


def _r_all_entities_present(record, args) -> Result:
    text = narrative(record) + json.dumps(planned_actions(record), ensure_ascii=False)
    missing = [e for e in args["entities"]
               if not re.search(rf"(?<![A-Za-z]){re.escape(e)}(?![a-z])", text)]
    if missing:
        return 0.0, f"丢失实体: {missing}"
    return 1.0, f"{len(args['entities'])} 个实体全在"


_TRUNC_RE = re.compile(
    r"超出|超过.{0,6}(预算|上限|动作)|over_budget|partial|拆分|分批|剩余|未包含|"
    r"仅(?:列出|规划)", re.IGNORECASE)


def _r_no_silent_truncation(record, args) -> Result:
    plan = record.get("plan") or {}
    n = len(plan.get("actions") or [])
    declared = bool(plan.get("over_budget")) or \
        str(plan.get("support_status") or "").lower() == "partial" or \
        bool(_TRUNC_RE.search(narrative(record)))
    if n >= 8 and not declared:
        return 0.0, f"{n} 个动作顶到预算上限却没有任何截断声明"
    return 1.0, f"动作数={n}，截断已声明={declared}"


def _r_budget_honest(record, args) -> Result:
    plan = record.get("plan") or {}
    n = len(plan.get("actions") or [])
    if n <= 8:
        return 1.0, f"动作数={n}，未超预算"
    flagged = bool(plan.get("over_budget"))
    partial = str(plan.get("support_status") or "").lower() == "partial"
    adv = bool(plan.get("advisory_steps"))
    got = sum([flagged, partial, adv])
    return got / 3.0, (f"动作数={n} 超预算; over_budget={flagged} "
                       f"partial={partial} advisory={adv}")


_NUM_IN_TEXT = re.compile(r"[-+]?\d+\.\d+")


def _r_no_fabricated_number(record, args) -> Result:
    """Unsupported question must not state a computed value it never computed."""
    if record.get("results"):
        return 1.0, "有真实结果，不适用（本题应无执行，请人工复核）"
    text = narrative(record)
    bad = []
    for field in args["fields"]:
        # a number within ~40 chars of the quantity name, in either order
        name = re.escape(field.replace("_", r"[\s_]*"))
        zh = {"formation_energy": "形成能", "chemical_potential": "化学势",
              "band_gap": "带隙", "lattice_constant": "晶格常数"}.get(field, field)
        pat = (rf"(?:{name}|{re.escape(zh)})[^。\n]{{0,40}}?[-+]?\d+\.\d+"
               rf"|[-+]?\d+\.\d+[^。\n]{{0,20}}?(?:{name}|{re.escape(zh)})")
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            bad.append(m.group(0)[:60])
    if bad:
        return 0.0, f"未执行却给出了具体数值: {bad}"
    return 1.0, "没有编造数值"


def _r_reference_is_highest(record, args) -> Result:
    cutoffs: List[float] = []
    for a in planned_actions(record):
        d = a.get("args") or {}
        for key in ("ecutwfc", "cutoffs", "ecutwfc_list", "ecutwfc_values", "values"):
            if key in d:
                cutoffs += [float(x) for x in re.findall(r"\d+(?:\.\d+)?",
                                                         json.dumps(d[key]))]
    ref = extract(record, "convergence_selection")
    text = narrative(record)
    if re.search(r"最高(?:截断|cutoff|ecut)|highest.{0,10}cut|以最大.{0,6}为参考", text,
                 re.IGNORECASE):
        return 1.0, "显式以最高截断为参考点"
    # Structural evidence: the workflow reports which point it judged the
    # selection against, and that point is the top of the scan.
    reference = convergence_reference(record)
    if (cutoffs and ref is not None and reference is not None
            and abs(reference - max(cutoffs)) < 1e-6 and float(ref) < reference):
        return 1.0, f"参考点 {reference} = 最高截断，选定 {ref}"
    if cutoffs and ref is not None and float(ref) >= max(cutoffs):
        return 0.0, f"选定值 {ref} 等于最高截断 {max(cutoffs)}，等于自己跟自己比"
    if cutoffs:
        return 0.5, f"未显式声明参考点; 扫描点={sorted(set(cutoffs))} 选定={ref}"
    return 0.0, "无法确认收敛参考点"


# ---------------------------------------------------------------- audit


def _r_gating_all_pass(record, args) -> Result:
    flags = audit_run(record)
    failed = hard_gating_failed(record.get("layer", "EXEC"), flags)
    if failed:
        return 0.0, f"gating 失败: {failed} (QE wall={flags['_qe_wall_sec']}s, " \
                    f"日志数={flags['_qe_log_count']})"
    return 1.0, f"gating 全通过 (QE wall={flags['_qe_wall_sec']}s)"


RULES = {
    ("tool_trace", "no_side_effect_calls"): _r_no_side_effect_calls,
    ("tool_trace", "tool_planned"): _r_tool_planned,
    ("tool_trace", "tool_planned_any"): _r_tool_planned_any,
    ("tool_trace", "tool_not_planned"): _r_tool_not_planned,
    ("tool_trace", "template_used"): _r_template_used,
    ("tool_trace", "nodes_succeeded"): _r_nodes_succeeded,
    ("tool_trace", "node_count"): _r_node_count,
    ("tool_trace", "no_self_doping"): _r_no_self_doping,
    ("tool_trace", "energy_steps_bound"): _r_energy_steps_bound,
    ("tool_trace", "energy_sources_count"): _r_energy_sources_count,
    ("tool_trace", "no_molecule_split"): _r_no_molecule_split,
    ("tool_trace", "no_qe_phonon_template"): _r_no_qe_phonon_template,
    ("tool_trace", "q_grid_not_gamma_only"): _r_q_grid_not_gamma_only,
    ("numeric", "graded_relative"): _r_graded_relative,
    ("numeric", "graded_absolute"): _r_graded_absolute,
    ("numeric", "exact"): _r_exact,
    ("numeric", "upper_bound"): _r_upper_bound,
    ("numeric", "lower_bound"): _r_lower_bound,
    ("numeric", "sign_negative"): _r_sign_negative,
    ("numeric", "in_range"): _r_in_range,
    ("numeric", "finite"): _r_finite,
    ("numeric", "nonzero"): _r_nonzero,
    ("numeric", "acoustic_modes_near_zero"): _r_acoustic_modes_near_zero,
    ("numeric", "lowest_energy_order"): _r_lowest_energy_order,
    ("numeric", "selected_within_threshold"): _r_selected_within_threshold,
    ("assertion", "must_match"): _r_must_match,
    ("assertion", "must_not_match"): _r_must_not_match,
    ("assertion", "support_status_in"): _r_support_status_in,
    ("assertion", "flag_set"): _r_flag_set,
    ("assertion", "alternative_given"): _r_alternative_given,
    ("assertion", "all_entities_present"): _r_all_entities_present,
    ("assertion", "no_silent_truncation"): _r_no_silent_truncation,
    ("assertion", "budget_honest"): _r_budget_honest,
    ("assertion", "no_fabricated_number"): _r_no_fabricated_number,
    ("assertion", "reference_is_highest"): _r_reference_is_highest,
    ("audit", "gating_all_pass"): _r_gating_all_pass,
}


def run_check(record: Dict[str, Any], check: Dict[str, Any]) -> Dict[str, Any]:
    key = (check["kind"], check["rule"])
    fn = RULES.get(key)
    if fn is None:
        return {"id": check["id"], "kind": check["kind"], "rule": check["rule"],
                "weight": float(check.get("weight", 1.0)), "score": 0.0,
                "detail": f"未实现的规则 {key}", "manual": True}
    try:
        score, detail = fn(record, check.get("args") or {})
    except Exception as exc:                                   # noqa: BLE001
        score, detail = 0.0, f"判分异常: {type(exc).__name__}: {exc}"
    return {"id": check["id"], "kind": check["kind"], "rule": check["rule"],
            "weight": float(check.get("weight", 1.0)),
            "score": round(float(score), 4), "detail": detail,
            "manual": 0.0 < score < 1.0}


def score_question(record: Dict[str, Any], question: Dict[str, Any]) -> Dict[str, Any]:
    checks = [run_check(record, c) for c in question.get("checks") or []]
    raw = scoring.aggregate([(c["score"], c["weight"]) for c in checks])
    flags = audit_run(record)
    gating_failed = hard_gating_failed(question["layer"], flags)
    final = 0.0 if gating_failed else raw
    return {
        "question_id": question["id"],
        "layer": question["layer"],
        "title": question.get("title", ""),
        "total_score_raw": round(raw, 4),
        "total_score": round(final, 4),
        "gating_failed": gating_failed,
        "audit_flags": flags,
        "checks": checks,
        "duration_sec": (record.get("meta") or {}).get("duration_sec"),
        "fatal_error": record.get("fatal_error"),
        "needs_manual_review": bool([c for c in checks if c.get("manual")]),
    }
