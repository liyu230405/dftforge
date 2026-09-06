"""Rule-based planner: intent signals → tool-call steps.

Deterministic fallback when the LLM planner is unavailable or failed.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from dft_forge.agent_loop.helpers import _2D_KIND_NAMES, _ELEMENTS
from dft_forge.agent_loop.intent import IntentSignals, detect_intents

_ADSORBATE_MOLECULE = {
    "O": "o2", "N": "n2", "H": "h2", "Cl": "cl2", "F": "f2",
    "CO": "co", "OH": "oh", "NO": "no", "H2O": "h2o", "CO2": "co2", "NH3": "nh3",
}


class RulePlanner:
    """Deterministic planner: keyword signals → steps."""

    def plan(self, message: str, ctx: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        ctx = ctx or {}
        if ctx.get("last_run_id") and re.fullmatch(
            r"\s*(?:需要|可以|好的|好|继续|是的|要|请|提取|拿出来)\s*[。！!,.，]?\s*",
            message or "",
        ):
            return [{
                "tool": "graph.status",
                "args": {"run_id": ctx["last_run_id"]},
                "description": "读取上一次计算的详细输出并提取关键数值",
            }]
        sig = detect_intents(message, ctx)
        lower = message.lower()
        steps: List[Dict[str, Any]] = []

        # ── comparison intent: session ledger first, never a dead end ──
        if sig.wants_compare:
            ledger = ctx.get("ledger") or []
            metric = (
                "energy" if any(k in lower for k in ("能量", "energy"))
                else "band_gap" if any(k in lower for k in ("能带", "带隙", "带结构", "band"))
                else "auto"
            )
            if len(ledger) >= 2:
                return [
                    {"tool": "analysis.compare",
                     "args": {"use_last": 2, "metric": metric},
                     "description": "对比本会话最近两次计算"},
                ]
            # nothing to compare yet — if the message itself names a system
            # worth computing, plan that normally instead of a dead-end notice
            if not (sig.build2d_kind or sig.dopant_el or sig.formula_hint or sig.material_hint):
                known = "、".join(
                    str(e.get("label") or e.get("material") or "计算") for e in ledger
                ) or "还没有任何计算记录"
                return [{
                    "tool": "help",
                    "args": {"topic": "compare"},
                    "description": (
                        f"对比需要先有两个体系的计算结果。本会话已有: {known}。"
                        "请先补算另一个体系（例如'算纯石墨烯的能带'），再说'对比'即可。"
                    ),
                }]

        # ── composite thermochemistry intents: E_ads / formation energy ──
        # These assemble multi-calculation workflows (agent-level value-add).
        if sig.wants_eads:
            ads_species = sig.ads_el or "O"
            molecule_kind = _ADSORBATE_MOLECULE.get(ads_species.upper()) or _ADSORBATE_MOLECULE.get(ads_species.lower())
            if molecule_kind is None:
                for mk in ("co2", "h2o", "nh3", "cl2", "o2", "n2", "h2", "co"):
                    if mk in lower.replace(" ", "").replace("-", ""):
                        molecule_kind = mk
                        break
            is_single_atom = len(ads_species) <= 2 and ads_species.capitalize() in _ELEMENTS
            if sig.build2d_kind and molecule_kind:
                if not is_single_atom:
                    notice = (
                        f"分子吸附（{ads_species}）的多原子建模暂不支持；当前 E_ads 闭环支持"
                        "单原子吸附（O/H/N/Cl，气相参考 O₂/H₂/N₂/Cl₂）。可先算原子吸附能。"
                    )
                    return [{"tool": "help", "args": {"topic": "eads"}, "description": notice}]
                sc = "3x3"  # standard surface supercell for adsorption studies
                steps_eads: List[Dict[str, Any]] = [
                    {
                        "tool": "structure.build2d",
                        "args": {"kind": sig.build2d_kind, "supercell": sc,
                                 "adsorb": {"element": ads_species, "site": sig.ads_site or "top"}},
                        "description": f"构建吸附体系: {ads_species}@{sig.build2d_kind} {sc} 超胞({sig.ads_site or 'top'}位)",
                    },
                    {"tool": "graph.run", "args": {"template_id": "t0_scf", "inputs": {}},
                     "description": "吸附体系单点能 E(surf+ads)"},
                    {
                        "tool": "structure.build2d",
                        "args": {"kind": sig.build2d_kind, "supercell": sc},
                        "description": f"构建纯表面: {sig.build2d_kind} {sc} 超胞",
                    },
                    {"tool": "graph.run", "args": {"template_id": "t0_scf", "inputs": {}},
                     "description": "纯表面单点能 E(surf)"},
                    {"tool": "structure.molecule", "args": {"kind": molecule_kind},
                     "description": f"构建气体分子参考 {molecule_kind.upper()}"},
                    {"tool": "graph.run", "args": {"template_id": "t0_scf", "inputs": {}},
                     "description": "分子单点能 E(mol)"},
                    {"tool": "thermo.eads", "args": {},
                     "description": "计算吸附能 E_ads = E(surf+ads) − E(surf) − E(mol)"},
                ]
                return steps_eads

        if sig.wants_formation and (sig.formula_hint or sig.material_hint):
            comp = sig.formula_hint or sig.material_hint
            from dft_forge.structure_builder import REFERENCE_PHASES

            try:
                from dft_forge.structure_builder import reference_elements
                els = reference_elements(str(comp))
            except Exception:
                els = []
            unsupported = [e for e in els if e not in REFERENCE_PHASES]
            if unsupported:
                reply = (f"暂无法计算 {comp} 的形成能：缺少元素参考态（{', '.join(unsupported)}）。"
                         f"当前支持的元素: {', '.join(sorted(REFERENCE_PHASES))}")
                return [{"tool": "help", "args": {"topic": "formation"}, "description": reply}]
            steps_f: List[Dict[str, Any]] = [
                {"tool": "graph.run", "args": {"template_id": "t0_scf", "inputs": {"material": str(comp)}},
                 "description": f"{comp} 化合物单点能"},
            ]
            for el in els:
                steps_f.append({"tool": "structure.reference", "args": {"element": el},
                                "description": f"构建 {el} 元素参考态"})
                steps_f.append({"tool": "graph.run", "args": {"template_id": "t0_scf", "inputs": {}},
                                "description": f"{el} 参考态单点能"})
            steps_f.append({"tool": "thermo.formation", "args": {"formula": str(comp)},
                            "description": f"计算 {comp} 形成能"})
            return steps_f

        # ── 2D / doped structures: build first, then a chained graph.run ──
        # (the execution loop injects the built CIF into graph.run steps
        #  that name no material)
        if (
            (sig.build2d_kind or sig.dopant_el)
            and not sig.strong_analyze
            and not sig.wants_import
            and not sig.wants_generate
            and not (sig.wants_analyze and not sig.graph_targets)
        ):
            build_steps: List[Dict[str, Any]] = []
            if sig.build2d_kind:
                bargs: Dict[str, Any] = {"kind": sig.build2d_kind}
                if sig.dopant_el or sig.ads_el:
                    bargs["supercell"] = "3x3"
                if sig.dopant_el:
                    bargs["dopants"] = [{"index": 0, "element": sig.dopant_el}]
                if sig.ads_el:
                    bargs["adsorb"] = {"element": sig.ads_el, "site": sig.ads_site or "top"}
                kind_label = {"graphene": "石墨烯", "bn": "h-BN", "mos2": "MoS2", "ws2": "WS2",
                              "mose2": "MoSe2", "wse2": "WSe2"}.get(sig.build2d_kind, sig.build2d_kind)
                mods = []
                if sig.dopant_el:
                    mods.append(f"{sig.dopant_el}掺杂")
                if sig.ads_el:
                    mods.append(f"{sig.ads_el}吸附({sig.ads_site})")
                desc = f"构建{kind_label}单层" + ("".join(mods) if mods else "")
                build_steps.append({"tool": "structure.build2d", "args": bargs, "description": desc})
            elif sig.dopant_el:
                host = sig.formula_hint or sig.material_hint
                if host:
                    build_steps.append({
                        "tool": "structure.dope",
                        "args": {"source": host, "element": sig.dopant_el, "supercell": "2x2x2"},
                        "description": f"构建{sig.dopant_el}掺杂{host}超胞",
                    })
            if build_steps:
                if sig.graph_targets:
                    label = {"t2_bands": "能带结构", "t2_dos": "态密度", "t1_vc_relax": "结构优化"}
                    if sig.build2d_kind:
                        # include the modification in the name so ledger entries
                        # distinguish 掺杂/吸附 systems from the pristine host
                        name = f"{sig.dopant_el or ''}掺杂{_2D_KIND_NAMES.get(sig.build2d_kind, sig.build2d_kind)}" if sig.dopant_el else (
                            f"{_2D_KIND_NAMES.get(sig.build2d_kind, sig.build2d_kind)}({sig.ads_el}吸附)" if sig.ads_el
                            else _2D_KIND_NAMES.get(sig.build2d_kind, sig.build2d_kind)
                        )
                    else:
                        name = f"{sig.dopant_el}掺杂{sig.formula_hint or ''}"
                    for tid in sig.graph_targets:
                        build_steps.append({
                            "tool": "graph.run",
                            "args": {"template_id": tid, "inputs": {}},
                            "description": f"{name} {label[tid]}",
                        })
                return build_steps

        if sig.graph_targets and sig.formula_hint and not sig.strong_analyze and not (sig.wants_import or sig.wants_generate):
            label = {"t2_bands": "能带结构", "t2_dos": "态密度", "t1_vc_relax": "结构优化"}
            for tid in sig.graph_targets:
                steps.append({
                    "tool": "graph.run",
                    "args": {"template_id": tid, "inputs": {"material": sig.formula_hint}},
                    "description": f"{sig.formula_hint} {label[tid]}",
                })
            return steps

        # Intent: analyze / bond length / structure info
        if sig.wants_analyze:
            source = ctx.get("last_structure_file") or ctx.get("last_structure_source")
            if not source:
                if sig.build2d_kind:
                    kind_label = {"graphene": "石墨烯", "bn": "h-BN", "mos2": "MoS2", "ws2": "WS2",
                                  "mose2": "MoSe2", "wse2": "WSe2"}.get(sig.build2d_kind, sig.build2d_kind)
                    steps.append({
                        "tool": "structure.build2d",
                        "args": {"kind": sig.build2d_kind},
                        "description": f"构建{kind_label}单层",
                    })
                    source = "__generated__"
                elif sig.wants_generate or sig.material_hint or sig.formula_hint:
                    steps.append({
                        "tool": "structure.generate",
                        "args": {"source": sig.material_hint or sig.formula_hint or "Si"},
                        "description": f"Generate {sig.material_hint or sig.formula_hint or 'Si'} structure",
                    })
                    source = "__generated__"
                else:
                    source = "tests/fixtures/POSCAR_Si"
            steps.append({
                "tool": "structure.analyze",
                "args": {"source": source, "pair_types": sig.pair_types},
                "description": "Analyze structure",
            })

        # Intent: import / load structure
        elif sig.wants_import and not sig.wants_analyze:
            if not sig.has_structure_ctx:
                source = sig.material_hint or "tests/fixtures/POSCAR_Si"
                steps.append({
                    "tool": "structure.import",
                    "args": {"source": source},
                    "description": "Import structure",
                })

        # Intent: generate structure
        elif sig.wants_generate and not sig.wants_analyze:
            if sig.build2d_kind:
                kind_label = {"graphene": "石墨烯", "bn": "h-BN", "mos2": "MoS2", "ws2": "WS2",
                              "mose2": "MoSe2", "wse2": "WSe2"}.get(sig.build2d_kind, sig.build2d_kind)
                steps.append({
                    "tool": "structure.build2d",
                    "args": {"kind": sig.build2d_kind},
                    "description": f"构建{kind_label}单层",
                })
            else:
                steps.append({
                    "tool": "structure.generate",
                    "args": {"source": sig.material_hint or "Si"},
                    "description": f"Generate {sig.material_hint or 'Si'} structure",
                })

        # Intent: build QE input
        if sig.wants_build:
            calc = "scf"
            if any(k in lower for k in ["vc-relax", "relax", "松弛", "vc_relax"]):
                calc = "vc-relax"
            elif any(k in lower for k in ["bands", "带"]):
                calc = "bands"
            elif any(k in lower for k in ["dos", "态密度"]):
                calc = "dos"

            material = sig.material_hint or "Si"
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
        if sig.wants_submit:
            input_file = ctx.get("last_input_file") or "input.in"
            steps.append({
                "tool": "job.submit",
                "args": {"input": input_file, "backend": "local"},
                "description": "Submit job",
            })

        # Intent: job status
        if sig.wants_status:
            job_id = ctx.get("last_job_id") or (message.split()[-1] if message.split() else "local_default")
            steps.append({
                "tool": "job.status",
                "args": {"job_id": job_id},
                "description": "Check job status",
            })

        # Intent: parse result
        if sig.wants_parse:
            output_file = ctx.get("last_output_file") or "input.in"
            steps.append({
                "tool": "result.parse",
                "args": {"input": output_file, "type": ctx.get("last_result_type", "vc-relax")},
                "description": "Parse result",
            })

        # Intent: verify
        if sig.wants_verify:
            output_file = ctx.get("last_output_file") or "input.in"
            steps.append({
                "tool": "result.verify",
                "args": {"input": output_file, "task_type": ctx.get("last_task_type", "T1")},
                "description": "Verify result",
            })

        # Intent: ledger / history
        if sig.wants_ledger:
            steps.append({
                "tool": "ledger.query",
                "args": {"limit": 20},
                "description": "Query ledger",
            })

        # Intent: doctor / environment
        if sig.wants_doctor:
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
