"""Reply narration: LLM natural-language summary + deterministic compare block."""

from __future__ import annotations

import json
from typing import Any, Dict, List


def compare_block(data: dict) -> str:
    """Deterministic comparison summary: numbers the narrator must not lose."""
    out_lines = ["对比结果:"]
    if data.get("metric") == "band_gap":
        for it in data.get("items") or []:
            out_lines.append(
                f"  {it.get('label')}: 带隙 {it.get('band_gap_ev')} eV"
                + (f" ({it.get('type')})" if it.get("type") else "")
            )
        for d in data.get("differences") or []:
            out_lines.append(
                f"  {d['between'][0]} vs {d['between'][1]}: "
                f"带隙差 {d['delta_band_gap_ev']:+.4f} eV（{d['wider']} 更宽）"
            )
    else:
        for it in data.get("items") or []:
            per = f"，{it['energy_ev_per_atom']} eV/原子" if it.get("energy_ev_per_atom") is not None else ""
            out_lines.append(f"  {it['label']}: E = {it['energy_ev']} eV{per}")
        for d in data.get("differences") or []:
            per = f"，{d['delta_energy_ev_per_atom']:+.4f} eV/原子" if d.get("delta_energy_ev_per_atom") is not None else ""
            out_lines.append(
                f"  {d['between'][0]} vs {d['between'][1]}: "
                f"能量差 {d['delta_energy_ev']:+.4f} eV{per}（{d['lower']} 更低）"
            )
    out_lines.append(f"  {data.get('note', '')}")
    return "\n".join(out_lines)


def narrate(provider, message: str, commands: List[dict], results: List[dict], ctx: dict) -> str:
    """Ask the LLM to summarize executed steps in natural Chinese."""
    steps = []
    for c, r in zip(commands, results):
        data = r.get("json") if isinstance(r.get("json"), dict) else {}
        brief = {"tool": c.get("command"), "desc": c.get("description")}
        contract = (data.get("metadata") or {}).get("method_contract") or {}
        if contract:
            brief["confirmed_method"] = (contract.get("step") or {}).get("effective")
        brief["provenance"] = r.get("provenance") or {"execution_class": "unknown"}
        if r.get("error"):
            brief["error"] = str(r["error"])[:200]
        elif isinstance(data, dict) and data.get("error"):
            brief["error"] = str(data["error"])[:200]
        else:
            flat = {}
            for name in ("formation", "chemical_potential_reference", "comparison", "stability", "selection"):
                if isinstance(data.get(name), dict):
                    flat[name] = data[name]
            for k, v in (data or {}).items():
                if isinstance(v, (int, float, str, bool)) and k not in ("cif", "stdout"):
                    flat[k] = v if not isinstance(v, str) or len(v) < 80 else v[:80] + "…"
                elif isinstance(v, list) and k in (
                    "species", "output_files", "runs", "formation_lines",
                    "transition_levels", "site_results",
                ):
                    flat[k] = v[:6]
                elif k in ("items", "differences") and isinstance(v, list):
                    # comparison detail rows carry the numbers the narrator
                    # must quote verbatim — never drop them in flattening
                    flat[k] = v[:8]
            # Graph tools wrap headline values in outputs/nodes. Include those
            # scalars in the narrator prompt so it cannot honestly claim that
            # a successful calculation returned only a status.
            for k, v in (data.get("outputs") or {}).items() if isinstance(data.get("outputs"), dict) else []:
                if isinstance(v, (int, float, str, bool)) or v is None:
                    flat[f"output.{k}"] = v
                elif k == "pdos_elements" and isinstance(v, list):
                    flat[f"output.{k}"] = v
            for node_id, node in (data.get("nodes") or {}).items() if isinstance(data.get("nodes"), dict) else []:
                for k, v in (node.get("outputs") or {}).items():
                    if isinstance(v, (int, float, str, bool)) or v is None:
                        flat[f"{node_id}.{k}"] = v
                    elif k == "frequencies_thz" and isinstance(v, list):
                        flat[f"{node_id}.{k}"] = v[:24]
                    elif k == "elements" and isinstance(v, list):
                        flat[f"{node_id}.{k}"] = v
                    elif k == "element_dos_at_fermi" and isinstance(v, dict):
                        flat[f"{node_id}.{k}"] = v
            brief["out"] = flat
        steps.append(brief)
    system = (
        "你是 DFT-Forge 计算代理的播报员。根据用户请求和已执行的工具步骤，"
        "用自然的中文回复：说明做了什么、关键数值结果（能量/带隙/原子数等）、"
        "或失败原因与建议。铁律：所有数值必须逐字来自执行步骤数据，"
        "禁止用教科书/记忆中的理论值替换计算值（如计算带隙 0.57 eV 就说 0.57，"
        "绝不说成实验值或理论值）；数据里没有的数值一个都不许出现。"
        "来源铁律：provenance 标为 simulated 或 contains_simulated=true 的结果必须明确称为模拟演示，"
        "不能写成科研结论；compute 仅表示计算输出，并不证明科学有效性。"
        "parsed/user_input/unknown 不能声称已执行真实计算，也不能根据当前连接设置猜测历史来源。"
        "confirmed_method 是实际确认的参数，不得称为未指定的默认值。"
        "is_metal=true 时，正的小 band_gap_ev 只能称为当前采样/展宽下不可分辨的数值间隔，不能宣称有限带隙半导体。"
        "声子稳定性只适用于给定 q_grid 的采样，Γ 点无显著虚频不能证明全布里渊区动力学稳定。"
        "带电缺陷只能引用数据中的显式介电、势对齐和有限尺寸修正，不能称为尺寸收敛或带隙修正后的结果。"
        "吸附位点最低只表示本次请求且成功验证的初始构型集合，不能称为全局最稳定构型。"
        "对比类结果必须逐体系给出数值和差值，结果已在手就立即汇报，"
        "禁止说'后续将输出/稍后给出'。"
        "不要罗列工具名，结构/图表已自动展示在界面右侧无需赘述。"
        "2-5 句话，直接输出文本。"
    )
    user = f"用户请求: {message}\n执行步骤: {json.dumps(steps, ensure_ascii=False, default=str)[:12000]}"
    out = str(provider.chat(system, user)).strip()
    return out
