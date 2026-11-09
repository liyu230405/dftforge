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


def align_formation_compounds(steps: List[Dict[str, Any]], session_dir: Path) -> List[str]:
    """Run formation-energy compounds on the same pseudopotential library as the references.

    Elemental references are generated structures, so they always use the
    GBRV ``{El}.upf`` library. A compound named by its MATERIAL_DB key would
    otherwise use that entry's own pseudopotentials/functional (MgO: PZ
    vbc/GTH), and E_form would subtract incompatible total energies.
    Rewritten before method confirmation so the method card shows the file.
    """
    if not any(str(s.get("tool")) == "thermo.formation" for s in steps):
        return []
    from ase.io import write as ase_write

    from dft_forge.compiler import DEFAULT_PSEUDO_DIR, MATERIAL_DB, build_atoms, profile_from_atoms

    notes = []
    for step in steps:
        if str(step.get("tool")) != "graph.run":
            continue
        inputs = (step.get("args") or {}).get("inputs") or {}
        material = inputs.get("material")
        if inputs.get("structure") or material not in MATERIAL_DB:
            continue
        atoms = build_atoms(material)
        try:
            library = profile_from_atoms(atoms, DEFAULT_PSEUDO_DIR)
        except FileNotFoundError:
            continue
        db = MATERIAL_DB[material]
        if db.get("pseudos") == library["pseudos"] and not db.get("input_dft"):
            continue
        safe = re.sub(r"[^A-Za-z0-9]", "_", str(material))[:24]
        path = session_dir / f"formation_{safe}.cif"
        path.parent.mkdir(parents=True, exist_ok=True)
        ase_write(str(path), atoms)
        inputs = dict(inputs)
        inputs.pop("material", None)
        inputs["structure"] = str(path)
        step["args"] = {**step["args"], "inputs": inputs}
        notes.append(f"{material} 改用与元素参考态相同的赝势库 {library['pseudos']}（原材料库条目为 {db.get('pseudos')}）")
    return notes


def formation_method_mismatch(entries: List[Dict[str, Any]]) -> Optional[str]:
    """Why the compound/reference SCFs are not comparable total energies.

    Total energies only cancel between runs sharing the pseudopotential of
    each common element, the functional and the plane-wave cutoffs.
    """
    profiles = [entry.get("method_profile") for entry in entries]
    if not all(isinstance(p, dict) for p in profiles):
        return None  # legacy results without method evidence
    from ase.data import chemical_symbols

    pseudo_of: Dict[str, str] = {}
    for profile in profiles:
        for element, filename in (profile.get("pseudos") or {}).items():
            # AFM runs split one element into FeA/FeB species labels
            base = element[:-1] if element not in chemical_symbols and element[-1:] in "AB" else element
            if pseudo_of.setdefault(base, filename) != filename:
                return f"元素 {base} 的赝势不一致（{pseudo_of[base]} vs {filename}）"
    dfts = {str(p.get("input_dft") or "from_pseudo").lower() for p in profiles}
    if len(dfts) > 1:
        return f"交换关联泛函不一致（{', '.join(sorted(dfts))}）"
    for key in ("ecutwfc", "ecutrho"):
        values = {(p.get("effective") or {}).get(key) for p in profiles}
        if len(values) > 1:
            return f"{key} 不一致（{', '.join(str(v) for v in values)}）"
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
    method: Optional[Dict[str, Any]] = None,
    structure_outputs: Optional[Dict[int, str]] = None,
) -> Optional[str]:
    """Inject session paths / chained context into step args in place.

    Returns an error string when the step must be aborted before execution.
    """
    args.pop("_method_contract", None)
    if method:
        structure_input = method["step"].get("structure_input") or {}
        producer = structure_input.get("from_step")
        if producer is not None:
            source = (structure_outputs or {}).get(producer)
            if (not source or not Path(source).is_file() or Path(source).is_symlink()
                    or not Path(source).resolve().is_relative_to(session_dir.resolve())):
                return f"确认方法要求使用第 {producer + 1} 步生成的结构，但该步骤未成功产生文件；拒绝回退到其他结构"
            if tool_id == "graph.run":
                args.setdefault("inputs", {})["structure"] = source
                args["inputs"].pop("material", None)
            elif tool_id == "input.build":
                args["structure"] = source
        bindings = method["step"]["applied"]
        target = args.setdefault("inputs", {}) if tool_id == "graph.run" else args
        for key in method["step"].get("strip_planner_parameters", []):
            target.pop(key, None)
        target.update(bindings)
        policy = method["step"].get("auto_parameter_policy")
        if policy:
            from dft_forge.compiler.qe_parameters import resolve_qe_parameters
            from dft_forge.agent_loop.method_contract import _file_identity

            try:
                before = _file_identity(target.get("structure"), session_dir)
                recommendation = resolve_qe_parameters(
                    structure=target.get("structure"), material=target.get("material"),
                    accuracy=policy["accuracy"], overrides=policy["overrides"], task=policy["task"],
                )
                after = _file_identity(target.get("structure"), session_dir)
                if before != after:
                    return "生成结构在参数解析期间发生变化，请重新预览"
            except (ValueError, OSError, KeyError, TypeError) as exc:
                return f"生成结构的参数解析失败：{exc}"
            resolved = {k: v for k, v in recommendation["parameters"].items() if k in policy["supported"]}
            target.update(resolved)
            method["step"]["recommendation"] = recommendation
            method["step"]["effective"].update(resolved)
            method["step"]["applied"].update(resolved)
            method["step"]["resolved_structure_input"] = after
            method["step"]["parameter_repair"] = "require_new_confirmation"
        if tool_id == "graph.run" or tool_id.startswith("workflow."):
            args["_method_contract"] = method
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
    elif tool_id == "structure.supercell":
        src = re.sub(r"[^A-Za-z0-9]", "_", str(args.get("source", "mat")))[:20]
        sc = re.sub(r"[^A-Za-z0-9]", "_", str(args.get("supercell", "2x2x2")))
        args["output"] = str(session_dir / f"supercell_{src}_{sc}.cif")
    elif tool_id == "structure.vacancy":
        src = re.sub(r"[^A-Za-z0-9]", "_", str(args.get("source", "mat")))[:20]
        el = re.sub(r"[^A-Za-z0-9]", "", str(args.get("element", "X")))[:3]
        sc = re.sub(r"[^A-Za-z0-9]", "_", str(args.get("supercell", "2x2x2")))
        args["output"] = str(session_dir / f"vacancy_{src}_{el}_{sc}.cif")
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
    elif tool_id in {
        "workflow.convergence",
        "workflow.neutral_vacancy",
        "workflow.charged_vacancy",
        "workflow.adsorption_scan",
        "workflow.magnetic_compare",
        "workflow.phonon_dfpt",
    }:
        # High-level workflows own a graph namespace below the session.  Do
        # not allow the planner to scatter dynamic DAG state to arbitrary
        # paths.
        name = re.sub(r"[^a-z0-9_]", "_", tool_id.replace(".", "_"))
        args["workdir"] = str(session_dir / "workflows" / name)
        if ctx.get("cancel_event") is not None:
            args["_cancel_event"] = ctx["cancel_event"]

        def _workflow_node_cb(ev, _idx=idx, _log=node_log):
            _log.append({"node": ev.get("node"), "state": ev.get("state")})
            emit({"type": "node", "step": _idx, "node": ev.get("node"), "state": ev.get("state")})

        args["_on_event"] = _workflow_node_cb
    elif tool_id == "thermo.eads" and (args.get("energy_steps") or args.get("ry_a") is None):
        # inject the last three SCF energies in plan order:
        # [adsorbed, surface, molecule]
        selected = scf_energies[-3:]
        if args.get("energy_steps"):
            import math
            refs = args["energy_steps"]
            if set(refs) != {"adsorbed", "surface", "molecule"} or len(set(refs.values())) != 3:
                return "吸附能必须分别绑定吸附体系、表面、分子的三个不同步骤"
            selected = []
            for role in ("adsorbed", "surface", "molecule"):
                matches = [entry for entry in scf_energies if entry.get("step_id") == refs[role]]
                if len(matches) != 1:
                    return f"吸附能引用 {refs[role]} 必须恰好有一个成功总能量，不能猜测节点或回退到最近结果"
                entry = matches[0]
                if (entry.get("graph_state") != "succeeded" or not isinstance(entry.get("step_index"), int)
                        or entry["step_index"] >= idx or not isinstance(entry.get("energy_ry"), (int, float))
                        or not math.isfinite(entry["energy_ry"]) or not isinstance(entry.get("natoms"), int)
                        or entry["natoms"] <= 0):
                    return f"吸附能引用 {refs[role]} 缺少成功状态、原子数或有效总能量证据"
                selected.append(entry)
            if selected[0]["natoms"] <= selected[1]["natoms"]:
                return "吸附体系的原子数必须大于纯表面；无法确定分子参考比例"
        if len(selected) == 3:
            e_a, e_s, e_m = selected
            args["ry_a"] = e_a["energy_ry"]
            args["ry_s"] = e_s["energy_ry"]
            args["ry_m"] = e_m["energy_ry"]
            args["_input_provenance"] = [item.get("provenance") for item in (e_a, e_s, e_m)]
            n_a, n_s, n_m = e_a.get("natoms"), e_s.get("natoms"), e_m.get("natoms")
            if n_a and n_s and n_m and n_a > n_s:
                # molecule energy is shared between its atoms: O adsorbed
                # from an O2 reference → subtract E(O2)/2, not E(O2)
                args["nat_a"] = n_a - n_s
                args["nat_mol"] = n_m
        else:
            return "thermo.eads 需要明确的吸附体系、纯表面和气相分子三个 SCF 能量；前置结果不足，拒绝猜测"
    elif tool_id == "thermo.formation" and not args.get("refs"):
        # inject: first SCF = compound, rest = element refs in plan order
        try:
            from dft_forge.structure_builder import reference_elements as _ref_els
            els = _ref_els(str(args.get("formula", "")))
        except Exception:
            els = []
        refs_needed = len(els)
        if scf_energies and len(scf_energies) >= refs_needed + 1:
            mismatch = formation_method_mismatch(scf_energies[:refs_needed + 1])
            if mismatch:
                return f"thermo.formation 拒绝混用不同方法的能量：{mismatch}"
            args["compound_energy"] = scf_energies[0]["energy_ry"]
            args["_method_consistent"] = all(
                isinstance(entry.get("method_profile"), dict) for entry in scf_energies[:refs_needed + 1])
            from dft_forge.structure_builder import reference_molecule_kind

            ran = set((ctx.get("structure_molecule_kind") or {}).values())
            args["_molecular_refs"] = {
                kind: True for kind in (reference_molecule_kind(el) for el in els) if kind in ran}
            refs = {}
            for i, el in enumerate(els):
                e = scf_energies[i + 1]
                refs[el] = {"energy_ry": e["energy_ry"], "natoms": e.get("natoms") or 1}
            args["refs"] = refs
            args["_input_provenance"] = [item.get("provenance") for item in scf_energies[:refs_needed + 1]]
        else:
            return (
                f"thermo.formation 需要化合物和 {refs_needed} 个元素参考态的 SCF 结果；"
                "前置结果不足，拒绝猜测"
            )
    elif tool_id == "analysis.compare" and not args.get("entries"):
        # inject the last N ledger entries (default 2) so "对比一下"
        # finds earlier-turn results without the planner re-specifying them
        ledger = ctx.get("ledger") or []
        n = int(args.get("use_last") or 2)
        if len(ledger) >= n:
            args["entries"] = ledger[-n:]
            args["_input_provenance"] = [item.get("provenance") for item in ledger[-n:]]
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

            molecule_kind = (ctx.get("structure_molecule_kind") or {}).get(str(inputs.get("structure") or ""))
            if molecule_kind:
                # An isolated molecule is Γ-only with integer occupations in
                # its ground-state spin, whatever bulk k-mesh was confirmed
                # for the crystals in the same plan.
                from dft_forge.structure_builder import molecule_scf_inputs

                molecular = molecule_scf_inputs(molecule_kind)
                inputs.update(molecular)
                if method:
                    method["step"].setdefault("notes", []).append(
                        f"{molecule_kind} 分子参考态按孤立分子计算：Γ 点、固定占据、基态自旋 {molecular}")
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
