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
        height=args.get("height", 1.5),
        vacuum=args.get("vacuum", 15.0),
        output=args.get("output"),
    )
    return _call_cmd(cmd_structure_build2d, ns)


def _structure_dope(args: dict) -> dict:
    from dft_forge.cli import cmd_structure_dope
    import argparse
    ns = argparse.Namespace(
        source=args.get("source", ""),
        element=args.get("element", ""),
        supercell=args.get("supercell", "2x2x2"),
        index=int(args.get("index", 0) or 0),
        output=args.get("output"),
    )
    return _call_cmd(cmd_structure_dope, ns)


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


def _analysis_compare(args: dict) -> dict:
    """Deterministic comparison of calculation results across systems.

    entries: [{label, band_gap_ev?, is_metal?, fermi_ev?, energy_ry?, natoms?}]
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
        ry_to_ev = 13.605693122994
        for c in clean:
            if c.get("energy_ry") is None:
                return {"error": f"'{c['label']}' has no SCF total energy (need t0_scf for both systems)"}
            e_ry = float(c["energy_ry"])
            n = int(c.get("natoms") or 0)
            items.append({
                "label": c["label"],
                "energy_ry": round(e_ry, 6),
                "energy_ev": round(e_ry * ry_to_ev, 4),
                "energy_ev_per_atom": round(e_ry * ry_to_ev / n, 4) if n else None,
                "natoms": n or None,
            })
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
            "带隙差 = 两者带隙之差；能量差为负表示后者更稳定（能量更低）。"
            if metric == "band_gap"
            else "总能量受原子数影响，每原子能量更适合不同体系间的比较。"
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
                "adsorbate on top/bridge/hollow site. Kinds: graphene, bn (h-BN), "
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
                            "element": {"type": "string"},
                            "site": {"type": "string", "enum": ["top", "bridge", "hollow"]},
                            "height": {"type": "number"},
                        },
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
                    "index": {"type": "integer", "description": "atom index to substitute"},
                    "output": {"type": "string"},
                },
                "required": ["source", "element"],
            },
            execute_fn=_structure_dope,
        ),
        ToolEntry(
            id="structure.molecule",
            name="Build Gas Molecule",
            description=(
                "Build a gas-phase molecule in a cubic box (Γ-point reference for "
                "adsorption energies). kind: o2|n2|h2|cl2|co|oh|no|h2o|co2|nh3. "
                "Example: {kind: 'o2'}"
            ),
            category="structure",
            input_schema={
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["o2", "n2", "h2", "cl2", "co", "oh", "no", "h2o", "co2", "nh3"]},
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
                "(from t2_bands) or total/per-atom energies (from t0_scf). "
                "Leave args empty (or set use_last) and the executor injects the "
                "session ledger entries. Example: {use_last: 2, metric: 'band_gap'}"
            ),
            category="analysis",
            input_schema={
                "type": "object",
                "properties": {
                    "entries": {
                        "type": "array",
                        "description": "explicit entries [{label, band_gap_ev, energy_ry, natoms, ...}] (usually auto-injected)",
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

    for tool in tools:
        registry.register(tool)

    from dft_forge.tools.graph_tools import register_graph_tools

    register_graph_tools(registry)
