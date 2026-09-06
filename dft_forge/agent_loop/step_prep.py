"""Step preparation: guards + per-tool argument injection.

Everything that mutates a step's args before execution lives here, so the
executor loop stays generic: prepare → call → handle result.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from dft_forge.agent_loop.helpers import _2D_HOST_ELEMENTS, _2D_KIND_NAMES


def check_self_doping(tool_id: str, args: Dict[str, Any]) -> Optional[str]:
    """Reject physically meaningless self-doping ("C掺杂石墨烯").

    Returns a user-facing notice when the step should be skipped, else None.
    """
    if tool_id == "structure.build2d":
        host_els = _2D_HOST_ELEMENTS.get(str(args.get("kind", "")))
        kind_name = _2D_KIND_NAMES.get(str(args.get("kind", "")), str(args.get("kind", "")))
        if host_els and len(host_els) == 1:
            host_el = next(iter(host_els))
            for d in args.get("dopants") or []:
                el = str(d.get("element", "")).strip()
                el_c = (el[:1].upper() + el[1:2].lower()) if el else ""
                if el_c == host_el:
                    return (
                        f"跳过「{el}掺杂{kind_name}」: {kind_name}本身就是"
                        f"纯{el}构成的，把{el}换成{el}得到的还是纯{kind_name}，"
                        f"这个计算没有物理意义。合理做法：对比 掺杂 vs 纯{kind_name}"
                        f"（例如说\"算纯{kind_name}的能带\"），或改用真正的异质原子"
                        "（如 B/N/P/S）掺杂。"
                    )
    elif tool_id == "structure.dope":
        el = str(args.get("element", "")).strip()
        el_c = (el[:1].upper() + el[1:2].lower()) if el else ""
        if el_c:
            try:
                from dft_forge.compiler import MATERIAL_DB, build_atoms, formula_atoms

                src = str(args.get("source", ""))
                if src in MATERIAL_DB:
                    host_syms = set(build_atoms(src).get_chemical_symbols())
                else:
                    host_syms = set(formula_atoms(src)[0].get_chemical_symbols())
            except Exception:
                host_syms = None
            if host_syms is not None and len(host_syms) == 1 and el_c in host_syms:
                host = str(args.get("source", ""))
                return (
                    f"跳过「{el}掺杂{host}」: 掺杂原子与本体元素相同，"
                    "替换后结构完全没有变化。请换一个真正的掺杂元素"
                    "（如 P/B/N），或直接计算纯体系。"
                )
    return None


def prepare_step_args(
    tool_id: str,
    args: Dict[str, Any],
    *,
    session_dir: Path,
    ctx: Dict[str, Any],
    built_structure: Optional[str],
    scf_energies: List[Dict[str, Any]],
    node_log: List[Dict[str, Any]],
    emit: Callable[[Dict[str, Any]], None],
    idx: int,
) -> Optional[str]:
    """Inject session paths / chained context into step args in place.

    Returns an error string when the step must be aborted before execution.
    """
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
    elif tool_id == "structure.analyze":
        source = str(args.get("source") or "")
        # LLMs often preserve a conversational filename (e.g. nacl.cif)
        # instead of the session-generated path. Prefer the known current
        # structure only when that literal path does not exist.
        current = built_structure or ctx.get("last_structure_source")
        if source == "__generated__" or (current and not Path(source).is_file() and Path(str(current)).is_file()):
            args["source"] = str(current) if current else source
    elif tool_id == "structure.molecule":
        if not args.get("output"):
            mk = re.sub(r"[^a-z0-9]", "", str(args.get("kind", "mol")))
            args["output"] = str(session_dir / f"molecule_{mk}.cif")
    elif tool_id == "structure.reference":
        if not args.get("output"):
            el = re.sub(r"[^A-Za-z]", "", str(args.get("element", "X")))
            args["output"] = str(session_dir / f"ref_{el}.cif")
    elif tool_id == "thermo.eads" and args.get("ry_a") is None:
        # inject the last three SCF energies in plan order:
        # [adsorbed, surface, molecule]
        if len(scf_energies) >= 3:
            e_a, e_s, e_m = scf_energies[-3], scf_energies[-2], scf_energies[-1]
            args["ry_a"] = e_a["energy_ry"]
            args["ry_s"] = e_s["energy_ry"]
            args["ry_m"] = e_m["energy_ry"]
            n_a, n_s, n_m = e_a.get("natoms"), e_s.get("natoms"), e_m.get("natoms")
            if n_a and n_s and n_m and n_a > n_s:
                # molecule energy is shared between its atoms: O adsorbed
                # from an O2 reference → subtract E(O2)/2, not E(O2)
                args["nat_a"] = n_a - n_s
                args["nat_mol"] = n_m
    elif tool_id == "thermo.formation" and not args.get("refs"):
        # inject: first SCF = compound, rest = element refs in plan order
        try:
            from dft_forge.structure_builder import reference_elements as _ref_els
            els = _ref_els(str(args.get("formula", "")))
        except Exception:
            els = []
        refs_needed = len(els)
        if scf_energies and len(scf_energies) >= refs_needed + 1:
            args["compound_energy"] = scf_energies[0]["energy_ry"]
            refs = {}
            for i, el in enumerate(els):
                e = scf_energies[i + 1]
                refs[el] = {"energy_ry": e["energy_ry"], "natoms": e.get("natoms") or 1}
            args["refs"] = refs
    elif tool_id == "analysis.compare" and not args.get("entries"):
        # inject the last N ledger entries (default 2) so "对比一下"
        # finds earlier-turn results without the planner re-specifying them
        ledger = ctx.get("ledger") or []
        n = int(args.get("use_last") or 2)
        if len(ledger) >= n:
            args["entries"] = ledger[-n:]
    elif tool_id == "input.build":
        args["output"] = str(session_dir / "input.in")
        if not args.get("structure") and built_structure:
            args["structure"] = built_structure
    elif tool_id in ("graph.run", "graph.status", "graph.templates"):
        # workdir is infrastructure: never trust an LLM-provided value —
        # a hallucinated path scatters runs across the session tree.
        # status/templates must hit the SAME engine as graph.run or they
        # query an empty default workspace and report runs as missing.
        args["workdir"] = str(session_dir / "graphs")
        if tool_id == "graph.run":
            inputs = dict(args.get("inputs") or {})
            previous_structure = ctx.get("last_structure_source")
            effective_structure = built_structure or (
                previous_structure if previous_structure and Path(str(previous_structure)).is_file() else None
            )
            if effective_structure and not inputs.get("structure"):
                # a structure built earlier in this plan is authoritative —
                # it overrides any material name the planner guessed
                inputs["structure"] = effective_structure
                inputs.pop("material", None)
                # scale band-path density with system size: a supercell's
                # folded BZ needs far fewer path points than a primitive
                # cell, and 100 points on 18+ atoms will time out locally
                if not inputs.get("nkpoints_bands") and str(args.get("template_id")) == "t2_bands":
                    try:
                        from ase.io import read as _ase_read

                        _na = len(_ase_read(str(effective_structure)))
                        if _na > 4:
                            inputs["nkpoints_bands"] = 60 if _na <= 12 else 40
                    except Exception:
                        pass
                args["inputs"] = inputs
            elif not effective_structure and not inputs.get("material") and not inputs.get("structure"):
                # nothing to chain: running the template default (Si) would
                # silently produce wrong physics for E_ads/formation flows
                return "no structure to chain (previous build step failed) and no material given"

            # Product-level method card: only inject values declared by the
            # selected template. This keeps approval visible and reproducible
            # without letting UI-only metadata leak into GraphTemplate inputs.
            run_config = ctx.get("run_config") or {}
            template_id = str(args.get("template_id") or "")
            approved_keys = {
                "t0_scf": ("ecutwfc", "kpoints"),
                "t1_vc_relax": ("ecutwfc", "kpoints"),
                "t2_bands": ("ecutwfc", "kpoints", "nbnd", "nkpoints_bands"),
                "t2_dos": ("ecutwfc", "kpoints", "nbnd"),
            }.get(template_id, ())
            for key in approved_keys:
                if run_config.get(key) not in (None, ""):
                    inputs[key] = run_config[key]
            args["inputs"] = inputs

            def _node_cb(ev, _idx=idx, _log=node_log):
                _log.append({"node": ev.get("node"), "state": ev.get("state")})
                emit({"type": "node", "step": _idx, "node": ev.get("node"), "state": ev.get("state")})

            args["_on_event"] = _node_cb
            # cancellation side-channel: web puts a threading.Event in ctx;
            # the scheduler polls it and kills pw.x subprocesses when set
            cancel_event = ctx.get("cancel_event")
            if cancel_event is not None:
                args["_cancel_event"] = cancel_event
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
    return None
