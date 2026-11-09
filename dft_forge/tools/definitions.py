"""DFT-Forge tool definitions."""

from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from typing import Any

from dft_forge.tools.models import ToolEntry
from dft_forge.tools.registry import registry


def _balanced_json_objects(text: str) -> list:
    """Extract top-level balanced JSON objects from possibly polluted text.

    CLI payloads are pretty-printed (indent=2), so a per-line scan cannot
    recover them; brace balancing with string-awareness can.
    """
    objects = []
    depth = 0
    start = None
    in_str = False
    escape = False
    for i, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    objects.append(text[start : i + 1])
                    start = None
    return objects


def _parse_payload(text: str):
    """Pick the most likely payload: the largest parseable JSON object.

    Debug prints and one-line fragments are small; the real payload dominates.
    """
    if not text or not text.strip():
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    candidates = []
    for chunk in _balanced_json_objects(text):
        try:
            candidates.append(json.loads(chunk))
        except json.JSONDecodeError:
            continue
    if not candidates:
        return None
    return max(candidates, key=lambda o: len(json.dumps(o, default=str)))


def _call_cmd(cmd_fn, ns) -> dict:
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
        rc = cmd_fn(ns)
    stdout = stdout_buf.getvalue()
    stderr = stderr_buf.getvalue()
    payload: dict = {"returncode": rc, "stdout": stdout, "stderr": stderr}

    # With --output set, _write_output writes JSON to the file instead of
    # stdout — the file is the authoritative payload.
    out_path = getattr(ns, "output", None)
    if out_path:
        p = Path(out_path)
        if p.exists():
            try:
                payload["json"] = json.loads(p.read_text())
            except json.JSONDecodeError:
                pass

    if "json" not in payload:
        parsed = _parse_payload(stdout)
        if parsed is None:
            # error responses are printed to stderr (see _error_response)
            parsed = _parse_payload(stderr)
        if parsed is not None:
            payload["json"] = parsed

    if "json" not in payload and (stdout.strip() or stderr.strip()):
        # loud failure: silent degradation hides tool breakage from the caller
        payload["json_parse_failed"] = True
    return payload


def _structure_import(args: dict) -> dict:
    from dft_forge.cli import cmd_structure_import
    import argparse
    ns = argparse.Namespace(
        source=args.get("source", ""),
        format=args.get("format"),
        fractional=args.get("fractional", False),
        min_nn=args.get("min_nn", 0.8),
        write_cif=args.get("write_cif"),
        output=args.get("output"),
    )
    return _call_cmd(cmd_structure_import, ns)


def _input_build(args: dict) -> dict:
    from dft_forge.cli import cmd_input_build
    import argparse
    ns = argparse.Namespace(
        structure=args.get("structure"),
        structure_format=args.get("structure_format"),
        fractional=args.get("fractional", False),
        material=args.get("material"),
        type=args.get("type", "scf"),
        prefix=args.get("prefix"),
        ecutwfc=args.get("ecutwfc"),
        ecutrho=args.get("ecutrho"),
        kpoints=args.get("kpoints"),
        conv_thr=args.get("conv_thr", 1.0e-8),
        nstep=args.get("nstep", 200),
        pseudo_dir=args.get("pseudo_dir"),
        output=args.get("output"),
    )
    return _call_cmd(cmd_input_build, ns)


def _job_submit(args: dict) -> dict:
    from dft_forge.cli import cmd_job_submit
    import argparse
    ns = argparse.Namespace(
        input=args.get("input", ""),
        backend=args.get("backend", "local"),
        backend_config=args.get("backend_config"),
        output=args.get("output"),
    )
    return _call_cmd(cmd_job_submit, ns)


def _job_status(args: dict) -> dict:
    from dft_forge.cli import cmd_job_status
    import argparse
    ns = argparse.Namespace(
        job_id=args.get("job_id", ""),
        backend=args.get("backend", "local"),
        backend_config=args.get("backend_config"),
        output=args.get("output"),
    )
    return _call_cmd(cmd_job_status, ns)


def _result_parse(args: dict) -> dict:
    from dft_forge.cli import cmd_result_parse
    import argparse
    ns = argparse.Namespace(
        input=args.get("input", ""),
        type=args.get("type", "vc-relax"),
        xml_path=args.get("xml_path"),
        output=args.get("output"),
    )
    return _call_cmd(cmd_result_parse, ns)


def _result_verify(args: dict) -> dict:
    from dft_forge.cli import cmd_result_verify
    import argparse
    ns = argparse.Namespace(
        input=args.get("input", ""),
        task_type=args.get("task_type"),
        output=args.get("output"),
    )
    return _call_cmd(cmd_result_verify, ns)


def _ledger_query(args: dict) -> dict:
    from dft_forge.cli import cmd_ledger_query
    import argparse
    ns = argparse.Namespace(
        database=args.get("database", "ledger.db"),
        task_id=args.get("task_id"),
        limit=args.get("limit", 20),
        output=args.get("output"),
    )
    return _call_cmd(cmd_ledger_query, ns)


def _doctor(args: dict) -> dict:
    from dft_forge.cli import cmd_doctor
    import argparse
    ns = argparse.Namespace(output=args.get("output"))
    return _call_cmd(cmd_doctor, ns)


def _structure_generate(args: dict) -> dict:
    from dft_forge.cli import cmd_structure_generate
    import argparse
    ns = argparse.Namespace(
        source=args.get("source", ""),
        output=args.get("output"),
    )
    return _call_cmd(cmd_structure_generate, ns)


def _structure_build2d(args: dict) -> dict:
    from dft_forge.cli import cmd_structure_build2d
    import argparse
    dopants = args.get("dopants") or []
    dopant_specs = []
    for d in dopants:
        # LLM plans may omit 'index' or pass bare element strings — the
        # default substitution site (index 0) keeps the build from crashing
        if isinstance(d, str):
            dopant_specs.append(f"{d}@0")
        else:
            dopant_specs.append(f"{d.get('element', 'X')}@{d.get('index', 0)}")
    ns = argparse.Namespace(
        kind=args.get("kind", "graphene"),
        supercell=args.get("supercell", "1x1"),
        vacancy=args.get("vacancy"),
        dopant=dopant_specs or None,
        adsorb=(
            f"{args['adsorb']['element']}@{args['adsorb'].get('site', 'top')}"
            if args.get("adsorb")
            else None
        ),
        height=(args.get("adsorb") or {}).get("height", args.get("height", 1.5)),
        site_index=int((args.get("adsorb") or {}).get("site_index", 0)),
        vacuum=args.get("vacuum", 15.0),
        output=args.get("output"),
    )
    return _call_cmd(cmd_structure_build2d, ns)


def _structure_dope(args: dict) -> dict:
    from dft_forge.cli import cmd_structure_dope
    import argparse
    raw_index = args.get("index")
    ns = argparse.Namespace(
        source=args.get("source", ""),
        element=args.get("element", ""),
        supercell=args.get("supercell", "2x2x2"),
        index=int(raw_index) if raw_index is not None else None,
        output=args.get("output"),
    )
    return _call_cmd(cmd_structure_dope, ns)


def _structure_supercell(args: dict) -> dict:
    from dft_forge.cli import cmd_structure_supercell
    import argparse
    ns = argparse.Namespace(
        source=args.get("source", ""),
        supercell=args.get("supercell", "2x2x2"),
        output=args.get("output"),
    )
    return _call_cmd(cmd_structure_supercell, ns)


def _structure_vacancy(args: dict) -> dict:
    from dft_forge.cli import cmd_structure_vacancy
    import argparse
    ns = argparse.Namespace(
        source=args.get("source", ""),
        element=args.get("element", ""),
        supercell=args.get("supercell", "2x2x2"),
        index=args.get("index"),
        output=args.get("output"),
    )
    return _call_cmd(cmd_structure_vacancy, ns)


def _structure_molecule(args: dict) -> dict:
    from dft_forge.cli import cmd_structure_molecule
    import argparse
    ns = argparse.Namespace(
        kind=args.get("kind", ""),
        box=float(args.get("box", 10.0)),
        output=args.get("output"),
    )
    return _call_cmd(cmd_structure_molecule, ns)


def _structure_reference(args: dict) -> dict:
    from dft_forge.cli import cmd_structure_reference
    import argparse
    ns = argparse.Namespace(
        element=args.get("element", ""),
        output=args.get("output"),
    )
    return _call_cmd(cmd_structure_reference, ns)


def _thermo_eads(args: dict) -> dict:
    from dft_forge.cli import cmd_thermo_eads
    import argparse
    if args.get("ry_a") is None or args.get("ry_s") is None or args.get("ry_m") is None:
        return {"error": "three SCF energies missing (E(surf+ads), E(surf), E(mol)) — run t0_scf for each first"}
    ns = argparse.Namespace(
        ry_a=args.get("ry_a"),
        ry_s=args.get("ry_s"),
        ry_m=args.get("ry_m"),
        nat_a=args.get("nat_a"),
        nat_mol=args.get("nat_mol"),
    )
    return _call_cmd(cmd_thermo_eads, ns)


def _thermo_formation(args: dict) -> dict:
    from dft_forge.cli import cmd_thermo_formation
    import argparse
    if args.get("compound_energy") is None:
        return {"error": "compound_energy missing — run t0_scf for the compound first"}
    refs = args.get("refs") or {}
    ns = argparse.Namespace(
        formula=args.get("formula", ""),
        compound_energy=args.get("compound_energy"),
        ref_element=list(refs.keys()),
        ref_energy=[refs[k].get("energy_ry") for k in refs],
        ref_natoms=[refs[k].get("natoms", 1) for k in refs],
    )
    return _call_cmd(cmd_thermo_formation, ns)


def _workflow_convergence(args: dict) -> dict:
    from dft_forge.workflows import run_convergence

    return run_convergence(args)


def _workflow_neutral_vacancy(args: dict) -> dict:
    from dft_forge.workflows import run_neutral_vacancy

    return run_neutral_vacancy(args)


def _workflow_charged_vacancy(args: dict) -> dict:
    from dft_forge.workflows import run_charged_vacancy

    return run_charged_vacancy(args)


def _workflow_adsorption_scan(args: dict) -> dict:
    from dft_forge.workflows import run_adsorption_scan

    return run_adsorption_scan(args)


def _chemical_potential_scan(args: dict) -> dict:
    from dft_forge.workflows import chemical_potential_scan

    return chemical_potential_scan(
        formula=args.get("formula", ""),
        formation_energy_ev_per_formula=args.get("formation_energy_ev_per_formula"),
        competitors=args.get("competitors"),
        n_points=args.get("n_points", 21),
    )


def _workflow_magnetic_compare(args: dict) -> dict:
    from dft_forge.workflows import run_magnetic_compare

    return run_magnetic_compare(args)


def _workflow_phonon_dfpt(args: dict) -> dict:
    from dft_forge.workflows import run_phonon_dfpt

    return run_phonon_dfpt(args)


def _analysis_compare(args: dict) -> dict:
    """Deterministic comparison of calculation results across systems.

    entries: [{label, band_gap_ev?, is_metal?, fermi_ev?, energy_ry?, natoms?,
               formula?/composition?/material?, method_profile?}]
    metric: "band_gap" | "energy" | "auto" (auto = band gap if any entry has one)
    """
    entries = args.get("entries") or []
    metric = str(args.get("metric") or "auto")

    if not entries:
        return {"error": "no entries to compare — the executor injects the session ledger automatically"}
    if len(entries) < 2:
        return {"error": "comparison needs at least 2 calculations; only 1 found"}

    clean = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        clean.append({
            "label": str(e.get("label") or e.get("material") or "未命名体系"),
            "band_gap_ev": e.get("band_gap_ev"),
            "is_metal": e.get("is_metal"),
            "fermi_ev": e.get("fermi_ev"),
            "energy_ry": e.get("energy_ry"),
            "natoms": e.get("natoms"),
            "template": e.get("template"),
            "formula": e.get("formula"),
            "composition": e.get("composition"),
            "material": e.get("material"),
            "method_profile": e.get("method_profile"),
        })
    if len(clean) < 2:
        return {"error": "comparison needs at least 2 valid calculations"}

    if metric == "auto":
        metric = "band_gap" if any(c.get("band_gap_ev") is not None for c in clean) else "energy"

    items = []
    if metric == "band_gap":
        for c in clean:
            if c.get("band_gap_ev") is None:
                return {"error": f"'{c['label']}' has no band gap (wrong template? need t2_bands for both systems)"}
            gap = float(c["band_gap_ev"])
            if gap == 0:
                gap_type = "金属 (零带隙)"
            elif c.get("is_metal"):
                gap_type = f"近零带隙/金属 ({gap} eV ≤ 展宽分辨率，不可分辨)"
            else:
                gap_type = "semiconductor/insulator"
            items.append({
                "label": c["label"],
                "band_gap_ev": round(gap, 4),
                "type": gap_type,
                "fermi_ev": round(float(c["fermi_ev"]), 4) if c.get("fermi_ev") is not None else None,
            })
        diffs = []
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                d = items[j]["band_gap_ev"] - items[i]["band_gap_ev"]
                diffs.append({
                    "between": [items[i]["label"], items[j]["label"]],
                    "delta_band_gap_ev": round(d, 4),
                    "wider": items[j]["label"] if d > 0 else items[i]["label"],
                })
    else:
        import math

        def _energy_error(message: str) -> dict:
            return {
                "error": message,
                "metric": "energy",
                "comparable": False,
            }

        def _entry_composition(entry: dict, natoms: int):
            raw = entry.get("composition")
            if raw is None:
                raw = entry.get("formula")
            if raw is None:
                raw = entry.get("material")
            if raw is None or raw == "":
                return None
            try:
                from pymatgen.core import Composition

                comp = Composition(raw)
                fractions = comp.fractional_composition.as_dict()
            except Exception:
                return None
            if not fractions or any(
                not math.isfinite(float(amount)) or float(amount) <= 0
                for amount in fractions.values()
            ):
                return None
            counts = {
                str(element): float(fraction) * natoms
                for element, fraction in fractions.items()
            }
            if any(abs(count - round(count)) > 1.0e-6 for count in counts.values()):
                return None
            return {element: int(round(count)) for element, count in sorted(counts.items())}

        def _method_signature(profile: dict, elements: set[str]):
            if not isinstance(profile, dict) or not profile:
                return None, "method_profile is missing"
            effective = profile.get("effective")
            pseudos = profile.get("pseudos")
            if not isinstance(effective, dict):
                return None, "method_profile.effective is missing"
            if not isinstance(pseudos, dict) or any(not pseudos.get(el) for el in elements):
                return None, "method_profile lacks pseudopotential evidence for the compared elements"
            required = ("ecutwfc", "ecutrho", "kpoints")
            missing = [key for key in required if effective.get(key) is None]
            if missing:
                return None, f"method_profile lacks {', '.join(missing)} evidence"
            if not isinstance(effective.get("kpoints"), (list, tuple)) or not effective["kpoints"]:
                return None, "method_profile has invalid kpoints evidence"
            return {
                "pseudos": {el: pseudos[el] for el in sorted(elements)},
                "input_dft": profile.get("input_dft") or "pseudo_default",
                "ecutwfc": effective.get("ecutwfc"),
                "ecutrho": effective.get("ecutrho"),
                "kpoints": list(effective.get("kpoints") or []),
                "occupations": effective.get("occupations", profile.get("occupations")),
                "smearing": effective.get("smearing", profile.get("smearing")),
                "degauss": effective.get("degauss", profile.get("degauss")),
                "tot_charge": effective.get("tot_charge", 0.0),
            }, None

        ry_to_ev = 13.605693122994
        evidence = []
        profiles_present = any(c.get("method_profile") is not None for c in clean)
        for c in clean:
            if c.get("energy_ry") is None:
                return {"error": f"'{c['label']}' has no SCF total energy (need t0_scf for both systems)"}
            try:
                e_ry = float(c["energy_ry"])
            except (TypeError, ValueError):
                return _energy_error(f"'{c['label']}' has an invalid SCF total energy")
            if not math.isfinite(e_ry):
                return _energy_error(f"'{c['label']}' has a non-finite SCF total energy")
            raw_natoms = c.get("natoms")
            try:
                n = int(raw_natoms)
            except (TypeError, ValueError, OverflowError):
                return _energy_error(
                    f"energy comparison requires a verifiable positive integer atom count; "
                    f"'{c['label']}' has natoms={raw_natoms!r}"
                )
            if n <= 0 or isinstance(raw_natoms, bool) or float(raw_natoms) != n:
                return _energy_error(
                    f"energy comparison requires a verifiable positive integer atom count; "
                    f"'{c['label']}' has natoms={raw_natoms!r}"
                )
            composition = _entry_composition(c, n)
            if composition is None:
                return _energy_error(
                    f"energy comparison requires formula/composition evidence consistent with natoms; "
                    f"cannot verify '{c['label']}'"
                )
            evidence.append((c, n, composition))
            items.append({
                "label": c["label"],
                "energy_ry": round(e_ry, 6),
                "energy_ev": round(e_ry * ry_to_ev, 4),
                "energy_ev_per_atom": round(e_ry * ry_to_ev / n, 4),
                "natoms": n,
                "composition": composition,
            })
        expected_natoms = evidence[0][1]
        expected_composition = evidence[0][2]
        for c, n, composition in evidence[1:]:
            if n != expected_natoms:
                return _energy_error(
                    "absolute total-energy comparison requires identical atom counts; "
                    f"'{evidence[0][0]['label']}' has {expected_natoms}, "
                    f"'{c['label']}' has {n}"
                )
            if composition != expected_composition:
                return _energy_error(
                    "absolute total-energy comparison requires identical elemental composition; "
                    f"'{evidence[0][0]['label']}' has {expected_composition}, "
                    f"'{c['label']}' has {composition}"
                )
        if profiles_present:
            elements = set(expected_composition)
            signatures = []
            for c, _, _ in evidence:
                signature, error = _method_signature(c.get("method_profile"), elements)
                if error:
                    return _energy_error(f"cannot verify method compatibility for '{c['label']}': {error}")
                signatures.append((c["label"], signature))
            reference_label, reference = signatures[0]
            for label, signature in signatures[1:]:
                for key in reference:
                    if signature[key] != reference[key]:
                        return _energy_error(
                            f"method mismatch for {key}: '{reference_label}' uses "
                            f"{reference[key]!r}, '{label}' uses {signature[key]!r}"
                        )
        diffs = []
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, b = items[i], items[j]
                d = b["energy_ev"] - a["energy_ev"]
                row = {
                    "between": [a["label"], b["label"]],
                    "delta_energy_ev": round(d, 4),
                    "lower": b["label"] if d < 0 else a["label"],
                }
                if a.get("energy_ev_per_atom") is not None and b.get("energy_ev_per_atom") is not None:
                    row["delta_energy_ev_per_atom"] = round(
                        b["energy_ev_per_atom"] - a["energy_ev_per_atom"], 4
                    )
                diffs.append(row)

    return {
        "command": "analysis.compare",
        "ok": True,
        "metric": metric,
        "items": items,
        "differences": diffs,
        "note": (
            "带隙差仅描述本次计算得到的数值间隔差异，不用于判断结构稳定性。"
            if metric == "band_gap"
            else "仅对组成、原子数和可用方法证据一致的结构比较绝对总能量；负差值表示后者能量更低。"
        ),
    }


def _structure_analyze(args: dict) -> dict:
    from dft_forge.cli import cmd_structure_analyze
    import argparse
    # clamp: LLMs invent destructive values (e.g. min_nn=4 rejects every
    # real bond — C-C is 1.42 Å); this is a validity floor, not a physics
    # threshold, so cap it below the shortest real bonds (H2 0.74 Å)
    min_nn = min(max(float(args.get("min_nn", 0.8) or 0.8), 0.3), 1.2)
    fractional = str(args.get("fractional", False)).lower() in ("true", "1", "yes")
    ns = argparse.Namespace(
        source=args.get("source", ""),
        format=args.get("format"),
        fractional=fractional,
        min_nn=min_nn,
        pair_types=args.get("pair_types"),
        output=args.get("output"),
    )
    return _call_cmd(cmd_structure_analyze, ns)


def register_default_tools() -> None:
    tools = [
        ToolEntry(
            id="structure.generate",
            name="Generate Structure",
            description="Generate a standard crystal structure for known materials like NaCl, MgO, Si, Al.",
            category="structure",
            input_schema={
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "output": {"type": "string"},
                },
                "required": ["source"],
            },
            execute_fn=_structure_generate,
        ),
        ToolEntry(
            id="structure.build2d",
            name="Build 2D Material",
            description=(
                "Build a 2D monolayer with optional supercell, doping, vacancy, and "
                "single-atom adsorbate on top/bridge/hollow site (not molecules). Kinds: graphene, bn (h-BN), "
                "mos2, ws2, mose2, wse2, mote2, wte2 (TMD monolayers). "
                "Example: {kind: 'mos2', supercell: '2x2', dopants: [{index: 0, element: 'Re'}], "
                "adsorb: {element: 'H', site: 'top'}}"
            ),
            category="structure",
            input_schema={
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": ["graphene", "bn", "mos2", "ws2", "mose2", "wse2", "mote2", "wte2"],
                    },
                    "supercell": {"type": "string", "description": "e.g. '3x3' or '4x4'"},
                    "vacancy": {"type": "integer", "description": "atom index to remove"},
                    "dopants": {
                        "type": "array",
                        "items": {
                            # bare element strings are coerced to index-0
                            # substitutions inside the tool, so the schema
                            # must declare the full accepted contract
                            "type": ["object", "string"],
                            "properties": {"index": {"type": "integer"}, "element": {"type": "string"}},
                        },
                    },
                    "adsorb": {
                        "type": "object",
                        "properties": {
                            "element": {"type": "string", "pattern": "^[A-Z][a-z]?$", "description": "One atomic element symbol, not a molecule such as CO2"},
                            "site": {"type": "string", "enum": ["top", "bridge", "hollow"]},
                            "height": {"type": "number"},
                            "site_index": {
                                "type": "integer",
                                "minimum": 0,
                                "description": "Reference-atom ordinal within the topmost surface layer; use it to distinguish e.g. B-top and N-top on h-BN",
                            },
                        },
                        "required": ["element"],
                        "additionalProperties": False,
                    },
                    "height": {"type": "number"},
                    "vacuum": {"type": "number"},
                    "output": {"type": "string"},
                },
                "required": ["kind"],
            },
            execute_fn=_structure_build2d,
        ),
        ToolEntry(
            id="structure.dope",
            name="Dope Bulk Crystal",
            description=(
                "Dope a bulk crystal: build supercell of source (material name, formula "
                "like CaTiO3, or CIF path), substitute one atom with the dopant element. "
                "Example: {source: 'Si', element: 'P', supercell: '2x2x2'}"
            ),
            category="structure",
            input_schema={
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "material name, formula, or structure file path"},
                    "element": {"type": "string", "description": "dopant element, e.g. P"},
                    "supercell": {"type": "string", "description": "e.g. '2x2x2' (larger = lower concentration)"},
                    "index": {"type": "integer", "description": "absolute atom index to substitute; omit for deterministic host-site selection"},
                    "output": {"type": "string"},
                },
                "required": ["source", "element"],
            },
            execute_fn=_structure_dope,
        ),
        ToolEntry(
            id="structure.supercell",
            name="Build Pristine Bulk Supercell",
            description=(
                "Create an unchanged bulk supercell for a matched pristine reference. "
                "This is not doping and does not substitute any atom. "
                "Example: {source: 'Si', supercell: '2x2x2'}"
            ),
            category="structure",
            input_schema={
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "material name, formula, or structure file path"},
                    "supercell": {"type": "string", "description": "e.g. '2x2x2'"},
                    "output": {"type": "string"},
                },
                "required": ["source"],
            },
            execute_fn=_structure_supercell,
        ),
        ToolEntry(
            id="structure.vacancy",
            name="Build Bulk Vacancy",
            description=(
                "Remove one explicitly selected atom from a bulk supercell. "
                "This is a neutral geometric vacancy builder, not substitution and "
                "not a charged-defect correction. Example: {source: 'MgO', element: 'O', supercell: '2x2x2'}"
            ),
            category="structure",
            input_schema={
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "material name, formula, or structure file path"},
                    "element": {"type": "string", "description": "element to remove, e.g. O"},
                    "supercell": {"type": "string", "description": "e.g. '2x2x2'"},
                    "index": {"type": "integer", "description": "absolute atom index after supercell construction"},
                    "output": {"type": "string"},
                },
                "required": ["source", "element"],
            },
            execute_fn=_structure_vacancy,
        ),
        ToolEntry(
            id="structure.molecule",
            name="Build Gas Molecule",
            description=(
                "Build a gas-phase molecule in a cubic box (Γ-point reference for "
                "adsorption energies). kind: o2|n2|h2|f2|cl2|co|oh|no|h2o|co2|nh3. "
                "Example: {kind: 'o2'}"
            ),
            category="structure",
            input_schema={
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["o2", "n2", "h2", "f2", "cl2", "co", "oh", "no", "h2o", "co2", "nh3"]},
                    "box": {"type": "number"},
                    "output": {"type": "string"},
                },
                "required": ["kind"],
            },
            execute_fn=_structure_molecule,
        ),
        ToolEntry(
            id="structure.reference",
            name="Elemental Reference Phase",
            description=(
                "Elemental reference phase for formation-energy bookkeeping "
                "(Si diamond, Na bcc, Ti hcp, O→O2 gas, ...). "
                "Example: {element: 'Na'}"
            ),
            category="structure",
            input_schema={
                "type": "object",
                "properties": {
                    "element": {"type": "string", "description": "element symbol, e.g. Na, Si, Ti, O"},
                    "output": {"type": "string"},
                },
                "required": ["element"],
            },
            execute_fn=_structure_reference,
        ),
        ToolEntry(
            id="analysis.compare",
            name="Compare Calculations",
            description=(
                "Compare two (or more) calculations from this session: band gaps "
                "(from t2_bands) or total/per-atom energies (from t0_scf). Energy "
                "ranking is emitted only for entries with verified identical composition "
                "and atom count, plus compatible method profiles when present. "
                "Leave args empty (or set use_last) and the executor injects the "
                "session ledger entries. Example: {use_last: 2, metric: 'band_gap'}"
            ),
            category="analysis",
            input_schema={
                "type": "object",
                "properties": {
                    "entries": {
                        "type": "array",
                        "description": (
                            "explicit entries [{label, band_gap_ev, energy_ry, natoms, "
                            "formula/composition/material, method_profile, ...}] (usually auto-injected)"
                        ),
                        "items": {"type": "object"},
                    },
                    "use_last": {"type": "integer", "description": "compare the last N ledger entries (default 2)"},
                    "metric": {"type": "string", "enum": ["band_gap", "energy", "auto"]},
                },
            },
            execute_fn=_analysis_compare,
        ),
        ToolEntry(
            id="thermo.eads",
            name="Adsorption Energy",
            description=(
                "E_ads = E(surface+adsorbate) − E(surface) − E(molecule), in eV. "
                "Feed the three t0_scf energies. Example: {ry_a: -158.0, ry_s: -156.0, ry_m: -1.0}"
            ),
            category="analysis",
            input_schema={
                "type": "object",
                "properties": {
                    "ry_a": {"type": "number", "description": "E(surface+adsorbate) in Ry"},
                    "ry_s": {"type": "number", "description": "E(surface) in Ry"},
                    "ry_m": {"type": "number", "description": "E(molecule) in Ry"},
                    "nat_a": {"type": "integer", "description": "adsorbed atoms on the surface"},
                    "nat_mol": {"type": "integer", "description": "atoms in the gas molecule (2 for O2)"},
                    "energy_steps": {
                        "type": "object", "description": "Bind successful prior graph.run step IDs; never guess energies",
                        "properties": {role: {"type": "string"} for role in ("adsorbed", "surface", "molecule")},
                        "required": ["adsorbed", "surface", "molecule"], "additionalProperties": False,
                    },
                },
                "required": ["ry_a", "ry_s", "ry_m"],
            },
            execute_fn=_thermo_eads,
        ),
        ToolEntry(
            id="thermo.formation",
            name="Formation Energy",
            description=(
                "E_form = E(compound) − Σ n_i·e_i(element per atom), in eV/atom. "
                "refs maps element → {energy_ry, natoms} from structure.reference+t0_scf. "
                "Example: {formula: 'NaCl', compound_energy: -164.0, refs: {Na: {energy_ry: -3.0, natoms: 2}, Cl: {energy_ry: -20.0, natoms: 2}}}"
            ),
            category="analysis",
            input_schema={
                "type": "object",
                "properties": {
                    "formula": {"type": "string"},
                    "compound_energy": {"type": "number"},
                    "refs": {
                        "type": "object",
                        "description": "element → {energy_ry: number, natoms: int}",
                    },
                },
                "required": ["formula", "compound_energy", "refs"],
            },
            execute_fn=_thermo_formation,
        ),
        ToolEntry(
            id="workflow.convergence",
            name="Convergence Scan",
            description=(
                "Run a reproducible SCF convergence sweep over ecutwfc, k-points, "
                "smearing/degauss, compare against the highest-cost reference, and select "
                "the least-cost point within a meV/atom threshold."
            ),
            category="analysis",
            input_schema={
                "type": "object",
                "properties": {
                    "material": {"type": "string"},
                    "structure": {"type": "string"},
                    "ecutwfc_values": {"type": "array", "items": {"type": "number"}},
                    "kpoints_values": {"type": "array"},
                    "smearing_values": {"type": "array", "items": {"type": "string"}},
                    "degauss_values": {"type": "array", "items": {"type": "number"}},
                    "threshold_mev_per_atom": {"type": "number"},
                    "max_jobs": {"type": "integer"},
                    "conv_thr": {"type": "number"},
                    "workdir": {"type": "string"},
                },
                "required": [],
            },
            execute_fn=_workflow_convergence,
        ),
        ToolEntry(
            id="workflow.neutral_vacancy",
            name="Neutral Vacancy Formation",
            description=(
                "Build a matched pristine/vacancy bulk supercell, optionally relax the "
                "neutral defect at fixed cell, run SCF energies, and compute "
                "Edef-Ebulk+mu_removed. reference_mode=explicit requires an absolute chemical potential; "
                "same_method_bulk_per_atom derives mu=Ebulk/N for an elemental host only."
            ),
            category="analysis",
            input_schema={
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "element": {"type": "string"},
                    "supercell": {"type": "string"},
                    "index": {"type": "integer"},
                    "charge": {"type": "integer"},
                    "relax": {"type": "boolean"},
                    "chemical_potential_ry": {"type": "number"},
                    "chemical_potential_ev": {"type": "number"},
                    "reference_mode": {"type": "string", "enum": ["explicit", "same_method_bulk_per_atom"], "default": "explicit"},
                    "ecutwfc": {"type": "number"},
                    "ecutrho": {"type": "number"},
                    "kpoints": {"type": "array", "items": {"type": "integer"}},
                    "conv_thr": {"type": "number"},
                    "workdir": {"type": "string"},
                },
                "required": ["source", "element"],
            },
            execute_fn=_workflow_neutral_vacancy,
        ),
        ToolEntry(
            id="workflow.charged_vacancy",
            name="Charged Vacancy Formation",
            description=(
                "Run matched pristine and charged-vacancy fixed-cell calculations for explicit charge states, "
                "then form E_f(E_F) using a user-supplied absolute chemical potential, potential alignment, "
                "and either a cubic first-order Makov-Payne image correction or explicit correction values. "
                "The workflow never guesses dielectric screening or alignment."
            ),
            category="analysis",
            input_schema={
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "element": {"type": "string", "pattern": "^[A-Z][a-z]?$"},
                    "supercell": {"type": "string"},
                    "index": {"type": "integer", "minimum": 0},
                    "charges": {
                        "type": "array", "minItems": 1, "uniqueItems": True,
                        "items": {"type": "integer", "minimum": -4, "maximum": 4},
                    },
                    "fermi_levels_ev": {"type": "array", "minItems": 1, "items": {"type": "number", "minimum": 0}},
                    "band_gap_ev": {"type": "number", "exclusiveMinimum": 0},
                    "relax": {"type": "boolean"},
                    "force_threshold": {"type": "number", "exclusiveMinimum": 0},
                    "nstep": {"type": "integer", "minimum": 1},
                    "chemical_potential_ry": {"type": "number"},
                    "chemical_potential_ev": {"type": "number"},
                    "reference_mode": {"type": "string", "enum": ["explicit", "same_method_bulk_per_atom"], "default": "explicit"},
                    "correction_mode": {"type": "string", "enum": ["makov_payne_cubic", "explicit"], "default": "makov_payne_cubic"},
                    "dielectric_constant": {"type": "number", "exclusiveMinimum": 0},
                    "madelung_alpha": {"type": "number", "exclusiveMinimum": 0},
                    "cubic_tolerance": {"type": "number", "exclusiveMinimum": 0, "maximum": 0.1},
                    "potential_alignment_ev_by_charge": {"type": "object", "additionalProperties": {"type": "number"}},
                    "electrostatic_correction_ev_by_charge": {"type": "object", "additionalProperties": {"type": "number"}},
                    "spin_by_charge": {"type": "object", "additionalProperties": {"type": "object"}},
                    "magnetization_tolerance_muB": {"type": "number", "minimum": 0},
                    "ecutwfc": {"type": "number", "exclusiveMinimum": 0},
                    "ecutrho": {"type": "number", "exclusiveMinimum": 0},
                    "kpoints": {"type": "array", "items": {"type": "integer"}},
                    "conv_thr": {"type": "number", "exclusiveMinimum": 0},
                    "workdir": {"type": "string"},
                },
                "required": ["source", "element", "charges"],
            },
            execute_fn=_workflow_charged_vacancy,
        ),
        ToolEntry(
            id="workflow.adsorption_scan",
            name="Multi-site Atomic Adsorption",
            description=(
                "Build one clean 2D surface, one homonuclear gas reference, and every requested atomic "
                "adsorption site; optionally relax each at fixed cell, run final SCFs, validate the O2 "
                "triplet reference, and rank E_ads over the successfully verified requested sites."
            ),
            category="analysis",
            input_schema={
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["graphene", "bn", "mos2", "ws2", "mose2", "wse2", "mote2", "wte2"]},
                    "adsorbate_element": {"type": "string", "pattern": "^[A-Z][a-z]?$"},
                    "element": {"type": "string", "pattern": "^[A-Z][a-z]?$"},
                    "supercell": {"type": "string"},
                    "sites": {
                        "type": "array", "minItems": 1, "maxItems": 32,
                        "items": {
                            "anyOf": [
                                {"type": "string", "enum": ["top", "bridge", "hollow"]},
                                {
                                    "type": "object",
                                    "properties": {
                                        "site": {"type": "string", "enum": ["top", "bridge", "hollow"]},
                                        "site_index": {"type": "integer", "minimum": 0},
                                        "site_id": {"type": "string"},
                                        "height": {"type": "number", "exclusiveMinimum": 0},
                                    },
                                    "required": ["site"], "additionalProperties": False,
                                },
                            ],
                        },
                    },
                    "site": {"type": "string", "enum": ["top", "bridge", "hollow"]},
                    "site_index": {"type": "integer", "minimum": 0},
                    "height": {"type": "number", "exclusiveMinimum": 0},
                    "vacuum": {"type": "number", "exclusiveMinimum": 0},
                    "reference_molecule": {"type": "string"},
                    "molecule_box": {"type": "number", "exclusiveMinimum": 0},
                    "relax": {"type": "boolean"},
                    "force_threshold": {"type": "number", "exclusiveMinimum": 0},
                    "nstep": {"type": "integer", "minimum": 1},
                    "surface_spin": {"type": "object"},
                    "adsorbed_spin": {"type": "object"},
                    "reference_spin": {"type": "object"},
                    "reference_magnetization_tolerance_muB": {"type": "number", "minimum": 0},
                    "adsorbed_magnetization_tolerance_muB": {"type": "number", "minimum": 0},
                    "max_adsorbate_surface_distance_angstrom": {"type": "number", "exclusiveMinimum": 0},
                    "ecutwfc": {"type": "number", "exclusiveMinimum": 0},
                    "ecutrho": {"type": "number", "exclusiveMinimum": 0},
                    "kpoints": {"type": "array", "items": {"type": "integer"}},
                    "conv_thr": {"type": "number", "exclusiveMinimum": 0},
                    "max_sites": {"type": "integer", "minimum": 1, "maximum": 32},
                    "max_jobs": {"type": "integer", "minimum": 1},
                    "workdir": {"type": "string"},
                },
                "required": ["kind"],
                "anyOf": [{"required": ["adsorbate_element"]}, {"required": ["element"]}],
            },
            execute_fn=_workflow_adsorption_scan,
        ),
        ToolEntry(
            id="thermo.chemical_potential_scan",
            name="Chemical Potential Scan",
            description=(
                "Compute binary/ternary chemical-potential stability domains from the "
                "parent formation energy, elemental bounds, and supplied competing phases. "
                "Without an exhaustive competitor list the result is partial."
            ),
            category="analysis",
            input_schema={
                "type": "object",
                "properties": {
                    "formula": {"type": "string"},
                    "formation_energy_ev_per_formula": {"type": "number"},
                    "competitors": {"type": "array", "items": {"type": "object"}},
                    "n_points": {"type": "integer"},
                },
                "required": ["formula", "formation_energy_ev_per_formula"],
            },
            execute_fn=_chemical_potential_scan,
        ),
        ToolEntry(
            id="workflow.magnetic_compare",
            name="Magnetic State Comparison",
            description=(
                "Run collinear NM/FM/AFM pw.x SCF states with explicit initial moments and "
                "compare total/per-atom energies and realized atom magnetizations."
            ),
            category="analysis",
            input_schema={
                "type": "object",
                "properties": {
                    "material": {"type": "string"},
                    "structure": {"type": "string"},
                    "supercell": {"type": "string"},
                    "magnetic_element": {"type": "string"},
                    "orders": {"type": "array", "items": {"type": "string", "enum": ["NM", "FM", "AFM", "nm", "fm", "afm"]}},
                    "initial_moment_muB": {"type": "number"},
                    "ecutwfc": {"type": "number"},
                    "kpoints": {"type": "array", "items": {"type": "integer"}},
                    "conv_thr": {"type": "number"},
                    "workdir": {"type": "string"},
                },
                "required": [],
            },
            execute_fn=_workflow_magnetic_compare,
        ),
        ToolEntry(
            id="workflow.phonon_dfpt",
            name="Phonon DFPT Workflow",
            description=(
                "Run pre-relaxation, SCF, ph.x on a regular q-grid, q2r.x, and matdyn.x "
                "phonon DOS; parse frequencies and identify imaginary modes."
            ),
            category="analysis",
            input_schema={
                "type": "object",
                "properties": {
                    "material": {"type": "string"},
                    "structure": {"type": "string"},
                    "q_grid": {"type": "array", "items": {"type": "integer", "minimum": 1}, "minItems": 3, "maxItems": 3},
                    "relax": {"type": "boolean"},
                    "ecutwfc": {"type": "number"},
                    "kpoints": {"type": "array", "items": {"type": "integer"}},
                    "conv_thr": {"type": "number"},
                    "tr2_ph": {"type": "number"},
                    "imaginary_threshold_thz": {"type": "number"},
                    "workdir": {"type": "string"},
                },
                "required": [],
            },
            execute_fn=_workflow_phonon_dfpt,
        ),
        ToolEntry(
            id="structure.import",
            name="Import Structure",
            description="Import and validate a structure from CIF, POSCAR, XYZ, QE input, or explicit dict.",
            category="structure",
            input_schema={
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "format": {"type": "string", "enum": ["cif", "poscar", "xyz", "qe_input", "explicit"]},
                    "fractional": {"type": "boolean"},
                    "min_nn": {"type": "number"},
                    "write_cif": {"type": "string"},
                    "output": {"type": "string"},
                },
                "required": ["source"],
            },
            execute_fn=_structure_import,
        ),
        ToolEntry(
            id="structure.analyze",
            name="Analyze Structure",
            description="Analyze a structure for formula, cell info, bond lengths, and pair distances.",
            category="structure",
            input_schema={
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "format": {"type": "string", "enum": ["cif", "poscar", "xyz", "qe_input", "explicit"]},
                    "fractional": {"type": "boolean"},
                    "min_nn": {"type": "number"},
                    "pair_types": {"type": "string"},
                    "output": {"type": "string"},
                },
                "required": ["source"],
            },
            execute_fn=_structure_analyze,
        ),
        ToolEntry(
            id="input.build",
            name="Build QE Input",
            description="Build a Quantum ESPRESSO input file for a known material or user structure.",
            category="input",
            input_schema={
                "type": "object",
                "properties": {
                    "structure": {"type": "string"},
                    "structure_format": {"type": "string"},
                    "fractional": {"type": "boolean"},
                    "material": {"type": "string"},
                    "type": {"type": "string", "enum": ["scf", "vc-relax", "relax", "bands", "dos"]},
                    "prefix": {"type": "string"},
                    "ecutwfc": {"type": "number"},
                    "ecutrho": {"type": "number"},
                    "kpoints": {"type": "array", "items": {"type": "integer"}},
                    "conv_thr": {"type": "number"},
                    "nstep": {"type": "integer"},
                    "pseudo_dir": {"type": "string"},
                    "output": {"type": "string"},
                },
                "required": ["type"],
            },
            execute_fn=_input_build,
        ),
        ToolEntry(
            id="job.submit",
            name="Submit Job",
            description="Submit a QE job on the selected backend.",
            category="job",
            input_schema={
                "type": "object",
                "properties": {
                    "input": {"type": "string"},
                    "backend": {"type": "string", "enum": ["local", "ssh"]},
                    "backend_config": {"type": "object"},
                    "output": {"type": "string"},
                },
                "required": ["input"],
            },
            execute_fn=_job_submit,
        ),
        ToolEntry(
            id="job.status",
            name="Job Status",
            description="Check job status by job id.",
            category="job",
            input_schema={
                "type": "object",
                "properties": {
                    "job_id": {"type": "string"},
                    "backend": {"type": "string", "enum": ["local", "ssh"]},
                    "backend_config": {"type": "object"},
                    "output": {"type": "string"},
                },
                "required": ["job_id"],
            },
            execute_fn=_job_status,
        ),
        ToolEntry(
            id="result.parse",
            name="Parse Result",
            description="Parse QE output into structured JSON.",
            category="result",
            input_schema={
                "type": "object",
                "properties": {
                    "input": {"type": "string"},
                    "type": {"type": "string", "enum": ["vc-relax", "scf", "nscf", "bands", "dos"]},
                    "xml_path": {"type": "string"},
                    "output": {"type": "string"},
                },
                "required": ["input", "type"],
            },
            execute_fn=_result_parse,
        ),
        ToolEntry(
            id="result.verify",
            name="Verify Result",
            description="Run scientific verification on parsed result JSON.",
            category="result",
            input_schema={
                "type": "object",
                "properties": {
                    "input": {"type": "string"},
                    "task_type": {"type": "string", "enum": ["T1", "T2"]},
                    "output": {"type": "string"},
                },
                "required": ["input"],
            },
            execute_fn=_result_verify,
        ),
        ToolEntry(
            id="ledger.query",
            name="Query Ledger",
            description="Query evidence ledger runs.",
            category="ledger",
            input_schema={
                "type": "object",
                "properties": {
                    "database": {"type": "string"},
                    "task_id": {"type": "string"},
                    "limit": {"type": "integer"},
                    "output": {"type": "string"},
                },
            },
            execute_fn=_ledger_query,
        ),
        ToolEntry(
            id="doctor",
            name="Environment Doctor",
            description="Probe QE, pseudopotentials, packages, and GPU environment.",
            category="general",
            input_schema={
                "type": "object",
                "properties": {
                    "output": {"type": "string"},
                },
            },
            execute_fn=_doctor,
        ),
    ]

    # Keep the effect/semantic contract next to the built-in registry.  The
    # planner sees the same metadata and the registry can enforce it for
    # callers that bypass AgentLoop.
    metadata = {
        "structure.generate": ("workspace_write", "build a standard crystal structure", []),
        "structure.build2d": ("workspace_write", "build a 2D cell with optional substitution/vacancy/adsorption", []),
        "structure.dope": ("workspace_write", "bulk elemental substitution; not vacancy or pristine copying", []),
        "structure.supercell": ("workspace_write", "copy a bulk structure into an unchanged matched supercell", []),
        "structure.vacancy": ("workspace_write", "remove one selected bulk atom; neutral geometry only", []),
        "structure.molecule": ("workspace_write", "build a boxed gas-phase molecule", []),
        "structure.reference": ("workspace_write", "build a conventional elemental reference phase", []),
        "structure.import": ("workspace_write", "read and normalize a user-provided structure", []),
        "structure.analyze": ("workspace_write", "analyze an existing structure; may write a report", []),
        "input.build": ("workspace_write", "compile a deterministic QE input file", []),
        "job.submit": ("external_execution", "submit or run a QE calculation", []),
        "job.status": ("read_only", "inspect a job status", []),
        "result.parse": ("workspace_write", "parse an existing QE result", []),
        "result.verify": ("workspace_write", "verify an existing parsed result; may write a report", []),
        "ledger.query": ("read_only", "query the evidence ledger", []),
        "doctor": ("read_only", "inspect the local compute environment", []),
        "analysis.compare": ("read_only", "compare already-computed results", []),
        "thermo.eads": ("read_only", "reduce three named SCF energies into adsorption energy", ["ry_a", "ry_s", "ry_m"]),
        "thermo.formation": ("read_only", "reduce a compound energy and elemental references into formation energy", ["compound_energy", "refs"]),
        "workflow.convergence": ("external_execution", "run a bounded SCF parameter sweep and select a converged point", []),
        "workflow.neutral_vacancy": ("external_execution", "run matched pristine/neutral-vacancy energies and reduce formation energy", []),
        "workflow.charged_vacancy": ("external_execution", "run matched charged-vacancy states with explicit correction evidence", []),
        "workflow.adsorption_scan": ("external_execution", "run shared-reference fixed-cell atomic adsorption site scans", []),
        "thermo.chemical_potential_scan": ("read_only", "solve elemental and supplied competing-phase chemical-potential constraints", []),
        "workflow.magnetic_compare": ("external_execution", "run collinear NM/FM/AFM pw.x states and compare realized magnetism", []),
        "workflow.phonon_dfpt": ("external_execution", "run ph.x/q2r.x/matdyn.x DFPT and verify frequencies", []),
    }
    for tool in tools:
        side_effect, contract, deferred = metadata.get(
            tool.id, (tool.side_effect, tool.semantic_contract, tool.deferred_inputs)
        )
        tool.side_effect = side_effect
        tool.semantic_contract = contract
        tool.deferred_inputs = list(deferred)

    for tool in tools:
        registry.register(tool)

    from dft_forge.tools.graph_tools import register_graph_tools

    register_graph_tools(registry)
