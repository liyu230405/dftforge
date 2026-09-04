"""Per-tool result handling: tool output → reply text / viewer / chart / ledger.

Mutates RunState (ctx, chained structure, SCF energies, reply parts) and
enriches result_dict with viewer/chart/metrics payloads for the frontend.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from dft_forge.agent_loop.helpers import (
    _extract_charts,
    _read_text,
    _viewer_payload_for_material,
    _viewer_payload_from_file,
)
from dft_forge.agent_loop.narration import compare_block


class RunState:
    """Mutable per-run session state shared across steps."""

    def __init__(self, ctx: Dict[str, Any], session_dir: Path):
        self.ctx = ctx
        self.session_dir = session_dir
        self.built_structure: Optional[str] = None
        self.scf_energies: List[Dict[str, Any]] = []


def handle_tool_result(
    tool_id: str,
    data: Dict[str, Any],
    args: Dict[str, Any],
    description: str,
    state: RunState,
    result_dict: Dict[str, Any],
    reply_parts: List[str],
) -> None:
    ctx = state.ctx

    if tool_id == "structure.import" and data.get("ok") and data.get("success"):
        ctx["last_structure_file"] = str(state.session_dir / "structure.json")
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
        state.built_structure = data.get("output")
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
        state.built_structure = data.get("output")
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
        state.built_structure = data.get("output")
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
    elif tool_id in ("structure.molecule", "structure.reference") and data.get("output"):
        state.built_structure = data.get("output")
        ctx["last_structure_source"] = data.get("output")
        ctx["last_structure_file"] = data.get("output")
        ctx["last_formula"] = data.get("formula")
        reply_parts.append(
            f"已构建: {data.get('formula')} — {data.get('description')}\n文件: {data.get('output')}"
        )
        result_dict["viewer"] = {
            "formula": data.get("formula"),
            "cif_path": data.get("output"),
            "cif": _read_text(data.get("output")),
            "natoms": data.get("natoms"),
        }
    elif tool_id == "graph.run" and data.get("run_id"):
        _handle_graph_run(data, args, description, state, result_dict, reply_parts)
    elif tool_id == "input.build" and data.get("input_file"):
        ctx["last_input_file"] = data.get("input_file")
        ctx["last_calc_type"] = data.get("calc_type")
        reply_parts.append(f"已构建输入: {data.get('input_file')} ({data.get('calc_type')})")
    elif tool_id == "job.submit" and data.get("ok") and data.get("success"):
        ctx["last_job_id"] = data.get("job_id")
        ctx["last_output_file"] = data.get("output_files", [None])[0] if data.get("output_files") else str(state.session_dir / "input.in")
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
        _handle_structure_analyze(data, args, ctx, reply_parts)
    elif tool_id == "thermo.eads" and isinstance(data.get("e_ads_ev"), (int, float)):
        ev = data.get("e_ads_ev")
        strength = "强吸附（放热）" if ev < -1.0 else ("中等吸附" if ev < -0.3 else "弱吸附/近似不吸附")
        reply_parts.append(
            f"吸附能计算完成: E_ads = {ev} eV/原子\n"
            f"判定: {strength}" + ("（负值=放热，体系稳定）" if ev < 0 else "（正值=吸热，不稳定吸附）")
        )
        result_dict["metrics"] = {"e_ads_ev": ev}
    elif tool_id == "thermo.formation" and isinstance(data.get("formation_energy_ev_per_atom"), (int, float)):
        evpa = data.get("formation_energy_ev_per_atom")
        stability = "相对单质稳定（负形成能）" if evpa < 0 else "相对单质不稳定（正形成能）"
        reply_parts.append(
            f"形成能计算完成: {data.get('formula')} E_form = {evpa} eV/原子"
            f"（{data.get('formation_energy_ev_per_formula')} eV/化学式）\n判定: {stability}"
        )
        result_dict["metrics"] = {"formation_energy_ev_per_atom": evpa}
    elif tool_id == "analysis.compare" and data.get("ok"):
        reply_parts.append(compare_block(data))
        result_dict["metrics"] = {"compare": data.get("differences")}
    else:
        reply_parts.append(f"{description}: 完成")


def _handle_graph_run(
    data: Dict[str, Any],
    args: Dict[str, Any],
    description: str,
    state: RunState,
    result_dict: Dict[str, Any],
    reply_parts: List[str],
) -> None:
    ctx = state.ctx
    ctx["last_run_id"] = data.get("run_id")
    _nodes = data.get("nodes") or {}
    lines = [f"图计算完成: {data.get('state')}  (运行ID: {data.get('run_id')})"]
    # ── session computation ledger: every finished graph run ──
    # remembered across turns so "对比/比较" and follow-ups can
    # find earlier results instead of claiming no records exist
    if data.get("state") == "succeeded":
        _bands_out = next(
            ((v.get("outputs") or {}) for k, v in _nodes.items()
             if k == "bands" and v.get("state") == "succeeded"),
            {},
        )
        _scf_out = next(
            ((v.get("outputs") or {}) for k, v in _nodes.items()
             if k in ("scf", "vc_relax") and v.get("state") == "succeeded"
             and (v.get("outputs") or {}).get("energy_ry") is not None),
            {},
        )
        ledger = ctx.setdefault("ledger", [])
        ledger.append({
            "label": description or args.get("template_id") or "计算",
            "template": args.get("template_id"),
            "material": (args.get("inputs") or {}).get("material"),
            "run_id": data.get("run_id"),
            "band_gap_ev": _bands_out.get("band_gap_ev"),
            "is_metal": _bands_out.get("is_metal"),
            "fermi_ev": _bands_out.get("fermi_ev"),
            "energy_ry": _scf_out.get("energy_ry"),
            "natoms": _scf_out.get("natoms") or _bands_out.get("natoms"),
        })
        ctx["ledger"] = ledger[-20:]
    for nid, info in (data.get("nodes") or {}).items():
        line = f"  节点 {nid}: {info.get('state')}"
        if info.get("error"):
            line += f" | {str(info['error'])[:100]}"
        outs = info.get("outputs") or {}
        bulky = ("eigenvalues_ev", "k_axis", "k_ticks", "dos_curve", "pdos_curve", "analysis")
        shown = {
            k: (round(v, 4) if isinstance(v, float) else v)
            for k, v in outs.items()
            if not str(k).endswith("stdout") and k not in bulky and not isinstance(v, list) and not isinstance(v, dict)
        }
        if shown:
            line += f" | {shown}"
        lines.append(line)
        # collect SCF-class energies for thermo workflows
        if info.get("state") == "succeeded" and outs.get("energy_ry") is not None:
            state.scf_energies.append({
                "energy_ry": outs["energy_ry"],
                "natoms": outs.get("natoms"),
                "label": f"{description}:{nid}",
            })
        # surface post-relax geometry facts directly in the reply
        an = outs.get("analysis")
        if isinstance(an, dict) and an.get("symmetry", {}).get("available"):
            lines.append(
                f"  结构分析: 空间群 {an['symmetry'].get('space_group')} "
                f"(No.{an['symmetry'].get('space_group_number')}) | "
                f"常规胞 {an['symmetry'].get('conventional_cell', {}).get('formula', '')}"
            )
            bl = an.get("bond_lengths") or {}
            for pair, st in list(bl.items())[:3]:
                lines.append(
                    f"  键长 {pair}: {st['mean_angstrom']} Å (均值, {st['count']} 条)"
                )
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


def _handle_structure_analyze(
    data: Dict[str, Any],
    args: Dict[str, Any],
    ctx: Dict[str, Any],
    reply_parts: List[str],
) -> None:
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
        sym = analysis.get("symmetry") or {}
        if sym.get("available"):
            conv = sym.get("conventional_cell") or {}
            lines.append(
                f"空间群: {sym.get('space_group')} (No.{sym.get('space_group_number')})"
                + (f" | 常规胞 {conv.get('formula')} (a={conv.get('a_angstrom')} Å)" if conv else "")
            )
        bl = analysis.get("bond_lengths") or {}
        for pair, st in list(bl.items())[:4]:
            lines.append(f"键长 {pair}: 均值 {st['mean_angstrom']} Å ×{st['count']}")
        ba = analysis.get("bond_angles") or {}
        for trip, st in list(ba.items())[:3]:
            lines.append(f"键角 {trip}: 均值 {st['mean_deg']}° ×{st['count']}")
        src = args.get("source") if args.get("source") != "__generated__" else ctx.get("last_structure_source")
        if src:
            lines.append(f"文件: {src}")
        reply_parts.append("\n".join(lines))
