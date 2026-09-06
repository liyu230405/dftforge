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
            for node_id, node in (data.get("nodes") or {}).items() if isinstance(data.get("nodes"), dict) else []:
                for k, v in (node.get("outputs") or {}).items():
                    if isinstance(v, (int, float, str, bool)) or v is None:
                        flat[f"{node_id}.{k}"] = v
            brief["out"] = flat
        steps.append(brief)
    system = (
        "你是 DFT-Forge 计算代理的播报员。根据用户请求和已执行的工具步骤，"
        "用自然的中文回复：说明做了什么、关键数值结果（能量/带隙/原子数等）、"
        "或失败原因与建议。铁律：所有数值必须逐字来自执行步骤数据，"
        "禁止用教科书/记忆中的理论值替换计算值（如计算带隙 0.57 eV 就说 0.57，"
        "绝不说成实验值或理论值）；数据里没有的数值一个都不许出现。"
        "对比类结果必须逐体系给出数值和差值，结果已在手就立即汇报，"
        "禁止说'后续将输出/稍后给出'。"
        "不要罗列工具名，结构/图表已自动展示在界面右侧无需赘述。"
        "2-5 句话，直接输出文本。"
    )
    user = f"用户请求: {message}\n执行步骤: {json.dumps(steps, ensure_ascii=False, default=str)[:4000]}"
    out = str(provider.chat(system, user)).strip()
    return out
