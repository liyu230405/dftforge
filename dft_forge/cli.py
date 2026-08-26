"""DFT-Forge CLI: agent-friendly atomic commands for materials computing.

Design goals:
- Each command does ONE thing and outputs structured JSON by default.
- Commands are composable: an agent can chain them.
- Human-readable output is still supported via --text / --quiet.
- Backends are selectable per command: local or ssh/slurm|pbs.

Examples:
    dft-forge structure import structure.cif --output json
    dft-forge input build --structure norm.cif --type vc-relax --ecut 50 --output input.in
    dft-forge job submit input.in --backend local --output json
    dft-forge job status <job-id> --backend local --output json
    dft-forge result parse output.out --type vc-relax --output json
    dft-forge result verify result.json --output json
    dft-forge ledger query --task-id T1_Si_vcrelax --limit 20 --output json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


# ── Helpers ──────────────────────────────────────────────────────────────────

def _json_dumps(data: Any) -> str:
    return json.dumps(data, indent=2, default=str)


def _write_output(data: Any, output_path: Optional[Path], output_format: str = "json") -> Optional[Path]:
    """Write data to stdout or file depending on flags."""
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_format == "json":
            output_path.write_text(_json_dumps(data))
        else:
            output_path.write_text(str(data))
        return output_path
    else:
        if output_format == "json":
            print(_json_dumps(data))
        else:
            print(data)
        return None


def _success_response(data: Dict[str, Any], output_path: Optional[Path]) -> int:
    data.setdefault("ok", True)
    _write_output(data, output_path)
    return 0


def _error_response(message: str, code: int = 1, details: Optional[Dict[str, Any]] = None) -> int:
    payload: Dict[str, Any] = {
        "ok": False,
        "error": message,
        "code": code,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if details:
        payload["details"] = details
    print(_json_dumps(payload), file=sys.stderr)
    return code


# ── Backend resolution ───────────────────────────────────────────────────────

def _resolve_executor(args: argparse.Namespace, default_bin_dir: Path = Path("/opt/homebrew/bin")):
    """Return (executor, backend_name, backend_config)."""
    backend = getattr(args, "backend", "local") or "local"
    config_path = getattr(args, "backend_config", None)
    config: Dict[str, Any] = {}
    if config_path:
        config_path = Path(config_path)
        if config_path.exists():
            config = json.loads(config_path.read_text())

    if backend == "local":
        from dft_forge.executor import LocalExecutor
        executor = LocalExecutor(
            qe_bin_dir=Path(config.get("qe_bin_dir", str(default_bin_dir))),
            max_walltime_sec=int(config.get("max_walltime_sec", 600)),
            max_retries=int(config.get("max_retries", 1)),
        )
        return executor, "local", config

    if backend == "ssh":
        from dft_forge.executor import SSHExecutor
        required = ["host", "username", "remote_workdir"]
        missing = [k for k in required if k not in config]
        if missing:
            raise ValueError(f"SSH backend config missing keys: {missing}")
        executor = SSHExecutor(
            host=config["host"],
            username=config["username"],
            remote_workdir=Path(config["remote_workdir"]),
            ssh_key=config.get("ssh_key"),
            qe_bin_dir=Path(config.get("qe_bin_dir", "/opt/software/q-e/bin")),
            scheduler=config.get("scheduler", "slurm"),
            max_walltime_sec=int(config.get("max_walltime_sec", 3600)),
            max_retries=int(config.get("max_retries", 1)),
            port=int(config.get("port", 22)),
        )
        return executor, "ssh", config

    raise ValueError(f"Unsupported backend: {backend}")


# ── Subcommands ──────────────────────────────────────────────────────────────

def cmd_structure_import(args: argparse.Namespace) -> int:
    from dft_forge.structure import StructureImporter

    source = args.source
    fmt = args.format
    if not fmt:
        path = Path(source)
        if path.exists():
            suffix = path.suffix.lower()
            if suffix == ".cif":
                fmt = "cif"
            elif suffix == ".xyz":
                fmt = "xyz"
            elif suffix in {".in", ".pwi", ".qe"}:
                fmt = "qe_input"
            else:
                fmt = "poscar"
        else:
            fmt = "explicit"

    importer = StructureImporter(min_nearest_neighbor_angstrom=args.min_nn)
    report = importer.import_structure(source, format=fmt, fractional=args.fractional)

    data = report.to_dict()
    data.setdefault("command", "structure.import")
    data.setdefault("source", source)
    data.setdefault("format", fmt)

    if args.write_cif and report.success and report.atoms is not None:
        try:
            from ase.io import write as ase_write
            cif_path = Path(args.write_cif)
            cif_path.parent.mkdir(parents=True, exist_ok=True)
            ase_write(str(cif_path), report.atoms)
            data["cif_path"] = str(cif_path)
        except Exception as exc:
            return _error_response("Failed to write CIF", details={"exception": str(exc)})

    if not report.success:
        return _error_response("Import failed", details=data)

    return _success_response(data, args.output)


def cmd_structure_validate(args: argparse.Namespace) -> int:
    from dft_forge.structure import StructureImporter

    source = args.source
    fmt = args.format
    if not fmt:
        path = Path(source)
        if path.exists():
            suffix = path.suffix.lower()
            if suffix == ".cif":
                fmt = "cif"
            elif suffix == ".xyz":
                fmt = "xyz"
            elif suffix in {".in", ".pwi", ".qe"}:
                fmt = "qe_input"
            else:
                fmt = "poscar"
        else:
            fmt = "explicit"

    importer = StructureImporter(min_nearest_neighbor_angstrom=args.min_nn)
    report = importer.import_structure(source, format=fmt, fractional=args.fractional)

    data = report.to_dict()
    data.setdefault("command", "structure.validate")
    data.setdefault("valid", report.success)
    if not report.success:
        return _error_response("Validation failed", code=2, details=data)
    return _success_response(data, args.output)


def cmd_structure_generate(args: argparse.Namespace) -> int:
    from ase.build import bulk as ase_bulk
    from ase.io import write as ase_write

    source = args.source.strip()
    workdir = Path.cwd()
    workdir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else workdir / "generated.cif"

    formula = None
    atoms = None
    if source.lower() in {"nacl", "clna"}:
        atoms = ase_bulk("NaCl", "rocksalt", a=5.64)
        formula = atoms.get_chemical_formula()
    elif source.lower() in {"mgo", "gomg"}:
        atoms = ase_bulk("MgO", "rocksalt", a=4.21)
        formula = atoms.get_chemical_formula()
    elif source.lower() in {"si"}:
        atoms = ase_bulk("Si", "diamond", a=5.43)
        formula = atoms.get_chemical_formula()
    elif source.lower() in {"al"}:
        atoms = ase_bulk("Al", "fcc", a=4.05)
        formula = atoms.get_chemical_formula()
    else:
        try:
            from dft_forge.compiler import formula_atoms

            atoms, _proto = formula_atoms(source)
            formula = atoms.get_chemical_formula()
        except Exception as exc:  # noqa: BLE001
            return _error_response(f"Unsupported generated structure: {source} ({exc})")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ase_write(str(output_path), atoms)

    data = {
        "command": "structure.generate",
        "source": source,
        "formula": formula,
        "output": str(output_path),
        "natoms": len(atoms),
        "species": sorted({sym for sym in atoms.get_chemical_symbols()}),
    }
    # JSON always goes to stdout — _success_response would clobber the CIF
    # written above when args.output is set
    data["ok"] = True
    print(_json_dumps(data))
    return 0


def cmd_structure_build2d(args: argparse.Namespace) -> int:
    from ase.io import write as ase_write

    from dft_forge.structure_builder import BuildResult, StructureBuildError, build_2d

    dopants = []
    for spec in args.dopant or []:
        try:
            element, index = spec.split("@", 1)
            dopants.append({"element": element.strip(), "index": int(index)})
        except ValueError:
            return _error_response(f"bad dopant spec {spec!r} (expected ELEMENT@INDEX, e.g. N@0)")

    adsorbate = None
    if args.adsorb:
        try:
            element, site = args.adsorb.split("@", 1)
            adsorbate = {"element": element.strip(), "site": site.strip().lower()}
        except ValueError:
            return _error_response(f"bad adsorb spec {args.adsorb!r} (expected ELEMENT@SITE, e.g. O@hollow)")

    try:
        result: BuildResult = build_2d(
            args.kind,
            supercell=args.supercell,
            vacancy_index=args.vacancy,
            dopants=dopants,
            adsorbate=adsorbate,
            height=args.height,
            vacuum=args.vacuum,
        )
    except StructureBuildError as exc:
        return _error_response(str(exc))

    output_path = Path(args.output) if args.output else Path.cwd() / f"{args.kind}_{args.supercell}.cif"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ase_write(str(output_path), result.atoms)

    data = result.to_dict()
    data["command"] = "structure.build2d"
    data["ok"] = True
    data["output"] = str(output_path)
    print(_json_dumps(data))
    return 0


def cmd_structure_dope(args: argparse.Namespace) -> int:
    """structure dope: supercell + single-atom substitution → CIF."""
    from pathlib import Path as _P

    from ase.io import write as ase_write

    from dft_forge.structure_builder import BuildResult, StructureBuildError, dope_3d

    try:
        result: BuildResult = dope_3d(
            args.source,
            args.element,
            supercell=args.supercell,
            index=args.index,
        )
    except (StructureBuildError, ValueError, FileNotFoundError) as exc:
        return _error_response(str(exc))

    output_path = _P(args.output) if args.output else _P.cwd() / f"doped_{args.source}_{args.element}.cif"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ase_write(str(output_path), result.atoms)

    data = result.to_dict()
    data["command"] = "structure.dope"
    data["ok"] = True
    data["output"] = str(output_path)
    print(_json_dumps(data))
    return 0


def cmd_structure_molecule(args: argparse.Namespace) -> int:
    """structure molecule: gas-phase molecule in a box → CIF."""
    from ase.io import write as ase_write

    from dft_forge.structure_builder import BuildResult, StructureBuildError, build_molecule

    try:
        result: BuildResult = build_molecule(args.kind, box=args.box)
    except StructureBuildError as exc:
        return _error_response(str(exc))

    output_path = Path(args.output) if args.output else Path.cwd() / f"molecule_{args.kind}.cif"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ase_write(str(output_path), result.atoms)

    data = result.to_dict()
    data["command"] = "structure.molecule"
    data["ok"] = True
    data["output"] = str(output_path)
    print(_json_dumps(data))
    return 0


def cmd_structure_reference(args: argparse.Namespace) -> int:
    """structure reference: elemental reference phase for formation energies."""
    from ase.io import write as ase_write

    from dft_forge.structure_builder import BuildResult, StructureBuildError, build_reference

    try:
        result: BuildResult = build_reference(args.element)
    except StructureBuildError as exc:
        return _error_response(str(exc))

    output_path = Path(args.output) if args.output else Path.cwd() / f"ref_{args.element}.cif"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ase_write(str(output_path), result.atoms)

    data = result.to_dict()
    data["command"] = "structure.reference"
    data["ok"] = True
    data["element"] = args.element
    data["output"] = str(output_path)
    print(_json_dumps(data))
    return 0


RY_TO_EV = 13.605693122994


def cmd_thermo_eads(args: argparse.Namespace) -> int:
    """thermo eads: E_ads = E(adsorbed) − E(surface) − (n_ads/n_mol)·E(molecule)."""
    try:
        e_a, e_s, e_m = args.ry_a, args.ry_s, args.ry_m
    except TypeError:
        return _error_response("eads requires --ry-a, --ry-s, --ry-m energies in Ry")
    n_ads = int(getattr(args, "nat_a", 0) or 0)   # adsorbed atoms on the surface
    n_mol = int(getattr(args, "nat_mol", 0) or 0)  # atoms in the gas molecule
    if n_ads > 0 and n_mol > 0:
        # one O atom adsorbed but the reference gas is O2 → subtract E(O2)/2
        ratio = n_ads / n_mol
    else:
        ratio = 1.0
    e_ads = e_a - e_s - ratio * e_m
    data = {
        "command": "thermo.eads",
        "ok": True,
        "e_ads_ev": round(e_ads * RY_TO_EV, 4),
        "e_ads_ry": round(e_ads, 6),
        "e_ads_ev_per_adsorbate": round(e_ads * RY_TO_EV / n_ads, 4) if n_ads else None,
        "molecule_ratio": round(ratio, 4),
        "formula_note": (
            f"E_ads = E(surf+ads) − E(surf) − {ratio:g}·E(mol) "
            "(molecule energy scaled by adsorbed-atom count); negative = exothermic"
        ),
    }
    print(_json_dumps(data))
    return 0


def cmd_thermo_formation(args: argparse.Namespace) -> int:
    """thermo formation: E_form from compound + elemental reference energies."""
    from dft_forge.compiler import formula_atoms

    try:
        atoms, _proto = formula_atoms(args.formula)
    except Exception as exc:
        return _error_response(f"cannot parse formula {args.formula!r}: {exc}")

    counts: Dict[str, int] = {}
    for s in atoms.get_chemical_symbols():
        counts[s] = counts.get(s, 0) + 1

    refs = dict(zip(args.ref_element or [], args.ref_energy or []))
    refs = {el: float(e) for el, e in refs.items()}
    ref_n = dict(zip(args.ref_element or [], args.ref_natoms or []))
    ref_n = {el: int(n) for el, n in ref_n.items()}

    missing = [el for el in counts if el not in refs]
    if missing:
        return _error_response(
            f"missing reference energy for: {', '.join(missing)} — run structure.reference + t0_scf for each"
        )

    e_comp = float(args.compound_energy)
    e_form = e_comp - sum(n * (refs[el] / ref_n.get(el, 1)) for el, n in counts.items())
    n_atoms = sum(counts.values())
    data = {
        "command": "thermo.formation",
        "ok": True,
        "formula": args.formula,
        "formation_energy_ev_per_atom": round(e_form * RY_TO_EV / n_atoms, 4),
        "formation_energy_ev_per_formula": round(e_form * RY_TO_EV, 4),
        "composition": counts,
        "n_atoms": n_atoms,
        "formula_note": "E_form = E(compound) − Σ n_i·e_i(element per atom); negative = thermodynamically stable vs elements",
    }
    print(_json_dumps(data))
    return 0


def cmd_structure_analyze(args: argparse.Namespace) -> int:
    from dft_forge.structure import StructureImporter

    source = args.source
    fmt = args.format
    if not fmt:
        path = Path(source)
        if path.exists():
            suffix = path.suffix.lower()
            if suffix == ".cif":
                fmt = "cif"
            elif suffix == ".xyz":
                fmt = "xyz"
            elif suffix in {".in", ".pwi", ".qe"}:
                fmt = "qe_input"
            else:
                fmt = "poscar"
        else:
            fmt = "explicit"

    importer = StructureImporter(min_nearest_neighbor_angstrom=args.min_nn)
    report = importer.import_structure(source, format=fmt, fractional=args.fractional)

    data = report.to_dict()
    data.setdefault("command", "structure.analyze")

    if report.success and report.atoms is not None:
        analysis = _analyze_structure(report.atoms, args.pair_types)
        data.setdefault("analysis", analysis)
    else:
        data.setdefault("analysis", None)
        if not report.success:
            return _error_response("Structure analyze failed", code=2, details=data)

    return _success_response(data, args.output)


def _analyze_structure(atoms, pair_types=None):
    from dft_forge.structure_analysis import analyze

    return analyze(atoms, pair_types=pair_types)


def cmd_materials_list(args: argparse.Namespace) -> int:
    from dft_forge.catalog import list_materials, get_material_profile

    names = list_materials()
    items = []
    for name in names:
        try:
            profile = get_material_profile(name)
            items.append({
                "material": name,
                "species": profile.get("species", []),
                "pseudos": profile.get("pseudos", {}),
                "ibrav": profile.get("ibrav"),
                "lattice_constant_angstrom": profile.get("lattice_constant_angstrom"),
                "input_dft": profile.get("input_dft"),
            })
        except Exception:
            items.append({"material": name})

    data = {
        "command": "materials.list",
        "count": len(items),
        "materials": items,
    }
    return _success_response(data, args.output)


def cmd_input_build(args: argparse.Namespace) -> int:
    from dft_forge.catalog import build_dynamic_material_profile, build_dynamic_task_spec, register_dynamic_material
    from dft_forge.compiler import QECompiler, get_kpoints
    from dft_forge.structure import StructureImporter

    calc_type = args.type
    workdir = Path.cwd()
    workdir.mkdir(parents=True, exist_ok=True)

    compiler = QECompiler(pseudo_dir=Path(args.pseudo_dir) if args.pseudo_dir else Path(__file__).resolve().parent.parent / "assets" / "pseudos")

    output_path = Path(args.output) if args.output else workdir / "dft-forge.in"

    if args.structure:
        importer = StructureImporter()
        report = importer.import_structure(args.structure, format=args.structure_format or "auto", fractional=args.fractional)
        if not report.success:
            return _error_response("Structure import failed", details=report.to_dict())

        material = f"user_{report.formula}"
        try:
            profile = build_dynamic_material_profile(material, report.atoms, report.formula)
            register_dynamic_material(material, profile)
            spec = build_dynamic_task_spec(material, calc_type=calc_type)
            task_id = f"user_{report.formula}_{calc_type.replace('-', '_')}"
        except Exception as exc:
            return _error_response("Failed to register dynamic material", details={"exception": str(exc)})

        prefix = args.prefix or task_id
        kpoints_tuple = tuple(args.kpoints) if args.kpoints else (4, 4, 4, 1, 1, 1)
        if calc_type in {"vc-relax", "relax"}:
            content = compiler.compile_t1(
                material,
                output_path,
                prefix=prefix,
                ecutwfc=args.ecutwfc,
                ecutrho=args.ecutrho,
                kpoints=get_kpoints(material, kpoints_tuple),
                conv_thr=args.conv_thr,
                nstep=args.nstep,
                cell_optimization=True,
            )
        elif calc_type == "scf":
            content = compiler.compile_scf(
                material,
                output_path,
                prefix=prefix,
                calculation="scf",
                ecutwfc=args.ecutwfc,
                ecutrho=args.ecutrho,
                kpoints=get_kpoints(material, kpoints_tuple),
                conv_thr=args.conv_thr,
            )
        else:
            return _error_response(f"Unsupported calc type for structure-based input: {calc_type}")

        data = {
            "command": "input.build",
            "task_id": task_id,
            "material": material,
            "calc_type": calc_type,
            "input_file": str(output_path),
            "prefix": prefix,
            "kpoints": list(get_kpoints(material, kpoints_tuple)),
        }
        print(_json_dumps(data))
        return 0

    if not args.material:
        return _error_response("Either --structure or --material is required")

    material = args.material
    prefix = args.prefix or material.lower()
    kpoints_tuple = tuple(args.kpoints) if args.kpoints else (4, 4, 4, 1, 1, 1)

    if calc_type in {"vc-relax", "relax"}:
        content = compiler.compile_t1(
            material,
            output_path,
            prefix=prefix,
            ecutwfc=args.ecutwfc,
            ecutrho=args.ecutrho,
            kpoints=get_kpoints(material, kpoints_tuple),
            conv_thr=args.conv_thr,
            nstep=args.nstep,
            cell_optimization=True,
        )
    elif calc_type == "scf":
        content = compiler.compile_scf(
            material,
            output_path,
            prefix=prefix,
            calculation="scf",
            ecutwfc=args.ecutwfc,
            ecutrho=args.ecutrho,
            kpoints=get_kpoints(material, kpoints_tuple),
            conv_thr=args.conv_thr,
        )
    else:
        return _error_response(f"Unsupported calc type: {calc_type}")

    data = {
        "command": "input.build",
        "task_id": f"{material}_{calc_type.replace('-', '_')}",
        "material": material,
        "calc_type": calc_type,
        "input_file": str(output_path),
        "prefix": prefix,
        "kpoints": list(get_kpoints(material, kpoints_tuple)),
    }
    print(_json_dumps(data))
    return 0


def _local_job_id(workdir: Path) -> str:
    return f"local_{workdir.name}_{int(time.time())}"


def cmd_job_submit(args: argparse.Namespace) -> int:
    input_file = Path(args.input)
    if not input_file.exists():
        return _error_response(f"Input file not found: {input_file}")

    executor, backend_name, backend_config = _resolve_executor(args)
    workdir = input_file.parent.resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    data: Dict[str, Any] = {
        "command": "job.submit",
        "backend": backend_name,
        "input_file": str(input_file),
        "workdir": str(workdir),
    }

    if backend_name == "local":
        try:
            result = executor.run_pw(input_file, workdir)
        except TypeError:
            # Fallback for executors without run_pw: stage + submit + poll
            handle = executor.submit(input_file, workdir)
            job_id = getattr(handle, "job_id", _local_job_id(workdir))
            status = executor.status(handle)
            while status and getattr(status, "state", None) not in {None, "done", "failed", "cancelled"}:
                time.sleep(1)
                status = executor.status(handle)
            result = executor.fetch(handle)

        job_id = getattr(result, "remote_path", None) or _local_job_id(workdir)
        data.update({
            "job_id": job_id,
            "success": result.success,
            "exit_code": result.exit_code,
            "walltime_sec": result.walltime_sec,
            "job_done": result.job_done,
            "output_files": result.output_files,
        })
        if not result.success:
            data["verifier_summary"] = "Local job reported failure"
            return _error_response("Local job failed", code=3, details=data)
        return _success_response(data, args.output)

    # SSH backend
    try:
        handle = executor.submit(input_file, workdir)
    except Exception as exc:
        return _error_response("Submit failed", details={"exception": str(exc), "backend": backend_name})

    job_id = getattr(handle, "job_id", str(uuid.uuid4()))
    data["job_id"] = job_id
    return _success_response(data, args.output)


def cmd_job_status(args: argparse.Namespace) -> int:
    executor, backend_name, backend_config = _resolve_executor(args)
    job_id = args.job_id

    data: Dict[str, Any] = {
        "command": "job.status",
        "backend": backend_name,
        "job_id": job_id,
    }

    try:
        if backend_name == "local":
            # Local jobs are synchronous; infer from workdir if possible.
            workdir = Path(job_id) if Path(job_id).exists() else None
            if workdir and (workdir / "result.json").exists():
                data.update({
                    "state": "done",
                    "workdir": str(workdir),
                    "result_path": str(workdir / "result.json"),
                })
            else:
                data.update({"state": "unknown"})
            return _success_response(data, args.output)

        status = executor.status(None)
    except NotImplementedError:
        data["state"] = "unknown"
        return _success_response(data, args.output)
    except Exception as exc:
        return _error_response("Status check failed", details={"exception": str(exc), "backend": backend_name})

    state = getattr(status, "state", None) or getattr(status, "status", "unknown")
    data["state"] = str(state)
    return _success_response(data, args.output)


def cmd_job_cancel(args: argparse.Namespace) -> int:
    executor, backend_name, backend_config = _resolve_executor(args)
    job_id = args.job_id
    data: Dict[str, Any] = {
        "command": "job.cancel",
        "backend": backend_name,
        "job_id": job_id,
    }
    try:
        if hasattr(executor, "cancel"):
            executor.cancel(None)
            data["cancelled"] = True
        else:
            data["cancelled"] = False
            data["note"] = f"{backend_name} backend does not implement cancel"
    except Exception as exc:
        return _error_response("Cancel failed", details={"exception": str(exc), "backend": backend_name})
    return _success_response(data, args.output)


def cmd_job_fetch(args: argparse.Namespace) -> int:
    executor, backend_name, backend_config = _resolve_executor(args)
    job_id = args.job_id
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data: Dict[str, Any] = {
        "command": "job.fetch",
        "backend": backend_name,
        "job_id": job_id,
        "output_dir": str(output_dir),
    }

    try:
        if backend_name == "local":
            src = Path(job_id)
            if src.exists() and src.is_dir():
                data["fetched"] = str(src)
                return _success_response(data, args.output)
            data["fetched"] = str(output_dir)
            return _success_response(data, args.output)

        result = executor.fetch(None)
        data["fetched"] = getattr(result, "remote_path", str(output_dir))
    except NotImplementedError:
        data["fetched"] = str(output_dir)
    except Exception as exc:
        return _error_response("Fetch failed", details={"exception": str(exc), "backend": backend_name})

    return _success_response(data, args.output)


def cmd_result_parse(args: argparse.Namespace) -> int:
    from dft_forge.parser import QEParser, ParsedVCResult, ParsedBands, ParsedDOS

    output_file = Path(args.input)
    if not output_file.exists():
        return _error_response(f"Output file not found: {output_file}")

    result_type = args.type
    xml_path = Path(args.xml_path) if args.xml_path else None
    if xml_path and not xml_path.exists():
        xml_path = None

    text = output_file.read_text(errors="replace")
    parsed: Any = None
    if result_type == "vc-relax":
        parsed = QEParser.parse_vc_relax(text, xml_path=xml_path)
    elif result_type == "scf":
        parsed = QEParser.parse_scf(text, xml_path=xml_path)
    elif result_type == "nscf":
        parsed = QEParser.parse_nscf(text, xml_path=xml_path)
    elif result_type == "bands":
        parsed = QEParser.parse_bands(text, xml_path=xml_path)
    elif result_type == "dos":
        parsed = QEParser.parse_dos(text, xml_path=xml_path)
    else:
        return _error_response(f"Unsupported result type: {result_type}")

    if hasattr(parsed, "to_dict"):
        data = parsed.to_dict()
    elif hasattr(parsed, "__dict__"):
        data = {k: v for k, v in parsed.__dict__.items() if not k.startswith("_")}
    else:
        data = {"value": str(parsed)}

    data.setdefault("command", "result.parse")
    data.setdefault("input_file", str(output_file))
    data.setdefault("result_type", result_type)
    return _success_response(data, args.output)


def cmd_result_verify(args: argparse.Namespace) -> int:
    from dft_forge.verifier import ScientificVerifier, ConvergenceReport

    result_path = Path(args.input)
    if not result_path.exists():
        return _error_response(f"Result file not found: {result_path}")

    result = json.loads(result_path.read_text())
    task_type = getattr(args, "task_type", None) or result.get("task_type", "T1")

    verifier = ScientificVerifier()
    # Reconstruct minimal job_result-like object
    class FakeJobResult:
        def __init__(self, result: Dict[str, Any]):
            self.success = result.get("status") == "pass"
            self.exit_code = 0 if self.success else 1
            self.stdout = ""
            self.output_files = result.get("output_files", [])
            self.walltime_sec = float(result.get("walltime_sec", 0.0))
            self.job_done = True

    job_result = FakeJobResult(result)

    # Try to parse parsed data if present
    parsed = result.get("parsed") or result.get("physical_results") or {}
    if hasattr(parsed, "to_dict"):
        parsed_obj = parsed
    else:
        from types import SimpleNamespace
        parsed_obj = SimpleNamespace(**{k: v for k, v in parsed.items() if isinstance(v, (int, float, str, list, dict))})

    if task_type == "T1":
        report = verifier.verify_t1(job_result, parsed_obj)
    elif task_type == "T2":
        report = verifier.verify_t2_bands(job_result, parsed_obj)
    else:
        report = verifier.verify_t1(job_result, parsed_obj)

    if isinstance(report, ConvergenceReport):
        data = {
            "command": "result.verify",
            "task_type": task_type,
            "passed": report.passed,
            "checks": {name: {"pass": c.pass_, "detail": c.detail, "value": c.value, "threshold": c.threshold} for name, c in report.checks.items()},
            "warnings": report.warnings,
        }
    else:
        data = {"command": "result.verify", "task_type": task_type, "passed": bool(report)}

    if not data["passed"]:
        data.setdefault("error", "Verification failed")
        return _error_response("Verification failed", code=4, details=data)
    return _success_response(data, args.output)


def cmd_ledger_record(args: argparse.Namespace) -> int:
    from dft_forge.ledger import EvidenceLedger, LedgerRunRecord

    db_path = Path(args.database) if args.database else Path("ledger.db")
    db_path = db_path.resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    ledger = EvidenceLedger(db_path=db_path)

    record = LedgerRunRecord(
        task_id=args.task_id,
        task_type=args.task_type,
        attempt=int(args.attempt),
        success=args.success.lower() in {"true", "1", "yes"},
        input_hash=args.input_hash or "",
        walltime_sec=float(args.walltime_sec or 0.0),
        verifier_passed=args.verifier_passed.lower() in {"true", "1", "yes"},
        failure_reasons=[s.strip() for s in (args.failure_reasons or "").split(",") if s.strip()],
        recovery_actions=[s.strip() for s in (args.recovery_actions or "").split(",") if s.strip()],
        failure_kind=args.failure_kind or "",
        evidence_path=args.evidence_path or "",
        output_files=[s.strip() for s in (args.output_files or "").split(",") if s.strip()],
    )
    row_id = ledger.record_run(record)
    data = {
        "command": "ledger.record",
        "database": str(db_path),
        "row_id": row_id,
        "task_id": record.task_id,
        "attempt": record.attempt,
    }
    ledger.close()
    return _success_response(data, args.output)


def cmd_ledger_query(args: argparse.Namespace) -> int:
    from dft_forge.ledger import EvidenceLedger

    db_path = Path(args.database).resolve() if args.database else Path("ledger.db").resolve()
    if not db_path.exists():
        return _error_response(f"Ledger database not found: {db_path}")

    ledger = EvidenceLedger(db_path=db_path)
    task_id = args.task_id if args.task_id else None
    runs = ledger.get_runs(task_id=task_id, limit=int(args.limit))
    data = {
        "command": "ledger.query",
        "database": str(db_path),
        "task_id": task_id,
        "count": len(runs),
        "runs": runs,
    }
    ledger.close()
    return _success_response(data, args.output)


def cmd_ledger_patterns(args: argparse.Namespace) -> int:
    from dft_forge.ledger import EvidenceLedger

    db_path = Path(args.database).resolve() if args.database else Path("ledger.db").resolve()
    if not db_path.exists():
        return _error_response(f"Ledger database not found: {db_path}")

    ledger = EvidenceLedger(db_path=db_path)
    task_id = args.task_id if args.task_id else None
    patterns = ledger.get_failure_patterns(task_id=task_id)
    data = {
        "command": "ledger.patterns",
        "database": str(db_path),
        "task_id": task_id,
        "patterns": patterns,
    }
    ledger.close()
    return _success_response(data, args.output)


def cmd_doctor(args: argparse.Namespace) -> int:
    report: Dict[str, Any] = {
        "command": "doctor",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "architecture": platform.machine(),
        "python_version": sys.version,
        "cpu_count": os.cpu_count(),
    }

    qe_bin = shutil.which("pw.x")
    report["qe_binary"] = qe_bin
    if qe_bin:
        try:
            proc = subprocess.run(["pw.x", "--version"], capture_output=True, text=True, timeout=10)
            report["qe_version"] = proc.stdout.strip()[:200]
        except Exception:
            report["qe_version"] = "unknown"
    else:
        report["qe_version"] = None

    report["bands_x_binary"] = shutil.which("bands.x")
    report["dos_x_binary"] = shutil.which("dos.x")

    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        report["gpu"] = gpu.stdout.strip() if gpu.returncode == 0 else "none"
    except Exception:
        report["gpu"] = "none"

    report["cuda_available"] = False
    try:
        import torch
        report["cuda_available"] = torch.cuda.is_available()
        report["cuda_device_count"] = torch.cuda.device_count()
    except ImportError:
        pass

    packages = {}
    for pkg in ["numpy", "scipy", "ase", "pymatgen", "spglib", "seekpath"]:
        try:
            mod = __import__(pkg)
            packages[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            packages[pkg] = "not installed"
    report["python_packages"] = packages

    try:
        import psutil
        report["ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 1)
    except ImportError:
        report["ram_gb"] = "unknown (install psutil)"

    pseudo_dir = Path(__file__).resolve().parent.parent / "assets" / "pseudos"
    report["pseudo_dir"] = str(pseudo_dir)
    report["available_pseudos"] = [
        f.name for f in pseudo_dir.iterdir() if f.suffix.lower() in {".upf", ".gth"}
    ] if pseudo_dir.exists() else []

    return _success_response(report, args.output)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        prog="dft-forge",
        description="DFT-Forge: agent-friendly Quantum ESPRESSO CLI",
    )
    parser.add_argument("--backend", default="local", choices=["local", "ssh"], help="Execution backend")
    parser.add_argument("--backend-config", default=None, help="Path to backend JSON config")
    parser.add_argument("--output", default=None, help="Write command output JSON to this file instead of stdout")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # structure import
    p = subparsers.add_parser("structure", help="Structure commands")
    struct_sub = p.add_subparsers(dest="structure_command")
    imp = struct_sub.add_parser("import", help="Import and validate a structure")
    imp.add_argument("source", help="Path to structure file or inline content")
    imp.add_argument("--format", default=None, choices=["cif", "poscar", "xyz", "qe_input", "explicit"])
    imp.add_argument("--fractional", action="store_true", default=False)
    imp.add_argument("--min-nn", type=float, default=0.8)
    imp.add_argument("--write-cif", default=None)
    imp.add_argument("--output", default=None)

    val = struct_sub.add_parser("validate", help="Validate a structure file")
    val.add_argument("source", help="Path to structure file or inline content")
    val.add_argument("--format", default=None, choices=["cif", "poscar", "xyz", "qe_input", "explicit"])
    val.add_argument("--fractional", action="store_true", default=False)
    val.add_argument("--min-nn", type=float, default=0.8)
    val.add_argument("--output", default=None)

    gen = struct_sub.add_parser("generate", help="Generate a standard crystal structure for known materials")
    gen.add_argument("source", help="Material name, e.g. NaCl, MgO, Si, Al")
    gen.add_argument("--output", default=None)

    b2d = struct_sub.add_parser("build2d", help="Build a 2D material (graphene/h-BN/TMD monolayer) with supercell/doping/adsorbate")
    b2d.add_argument("kind", help="graphene | bn | mos2 | ws2 | mose2 | wse2 | mote2 | wte2")
    b2d.add_argument("--supercell", default="1x1", help="Supercell spec, e.g. 3x3")
    b2d.add_argument("--vacancy", type=int, default=None, help="Atom index to remove")
    b2d.add_argument("--dopant", action="append", default=None, metavar="ELEMENT@INDEX", help="e.g. N@0 (repeatable)")
    b2d.add_argument("--adsorb", default=None, metavar="ELEMENT@SITE", help="e.g. O@hollow (site: top|bridge|hollow)")
    b2d.add_argument("--height", type=float, default=1.5, help="Adsorbate height above site (Å)")
    b2d.add_argument("--vacuum", type=float, default=15.0)
    b2d.add_argument("--output", default=None, help="Output CIF path")

    dop = struct_sub.add_parser("dope", help="Dope a bulk crystal: supercell + substitution (e.g. P in Si)")
    dop.add_argument("source", help="Material name, formula (CaTiO3), or CIF/POSCAR path")
    dop.add_argument("element", help="Dopant element, e.g. P")
    dop.add_argument("--supercell", default="2x2x2", help="Supercell spec, e.g. 2x2x2 (larger = lower concentration)")
    dop.add_argument("--index", type=int, default=0, help="Atom index to substitute in the supercell")
    dop.add_argument("--output", default=None, help="Output CIF path")

    ana = struct_sub.add_parser("analyze", help="Analyze structure: formula, cell, bonds, angles, coordination, space group")
    ana.add_argument("source", help="Path to structure file or inline content")
    ana.add_argument("--format", default=None, choices=["cif", "poscar", "xyz", "qe_input", "explicit"])
    ana.add_argument("--fractional", action="store_true", default=False)
    ana.add_argument("--min-nn", type=float, default=0.8)
    ana.add_argument("--pair-types", default=None, help="Optional comma-separated pairs like Na-Cl,Si-Si")
    ana.add_argument("--output", default=None)

    mol = struct_sub.add_parser("molecule", help="Build a gas molecule in a box (O2/H2/CO/...)")
    mol.add_argument("kind", help="o2 | n2 | h2 | cl2 | co | oh | no | h2o | co2 | nh3")
    mol.add_argument("--box", type=float, default=10.0, help="Cubic box edge in Å")
    mol.add_argument("--output", default=None)

    ref = struct_sub.add_parser("reference", help="Elemental reference phase for formation energies")
    ref.add_argument("element", help="Element symbol, e.g. Na, Si, Ti, O (O→O2 gas)")
    ref.add_argument("--output", default=None)

    # thermo
    th = subparsers.add_parser("thermo", help="Thermochemistry from computed energies")
    th_sub = th.add_subparsers(dest="thermo_command")
    ea = th_sub.add_parser("eads", help="Adsorption energy from three SCF energies")
    ea.add_argument("--ry-a", type=float, required=True, help="E(surface+adsorbate) in Ry")
    ea.add_argument("--ry-s", type=float, required=True, help="E(surface) in Ry")
    ea.add_argument("--ry-m", type=float, required=True, help="E(molecule) in Ry")
    ea.add_argument("--nat-a", type=int, default=None, help="Adsorbed atoms on the surface (e.g. 1)")
    ea.add_argument("--nat-mol", type=int, default=None, help="Atoms in the gas molecule (e.g. 2 for O2)")
    fm = th_sub.add_parser("formation", help="Formation energy from compound + element references")
    fm.add_argument("--formula", required=True, help="Compound formula, e.g. NaCl, CaTiO3")
    fm.add_argument("--compound-energy", type=float, required=True, help="E(compound) in Ry")
    fm.add_argument("--ref-element", action="append", default=None, help="Element (repeatable)")
    fm.add_argument("--ref-energy", action="append", default=None, help="Reference cell energy in Ry (repeatable)")
    fm.add_argument("--ref-natoms", action="append", default=None, help="Atoms in reference cell (repeatable)")

    # materials
    mat = subparsers.add_parser("materials", help="Material catalog commands")
    mat_sub = mat.add_subparsers(dest="materials_command")
    ml = mat_sub.add_parser("list", help="List known materials")
    ml.add_argument("--output", default=None)

    # input
    inp = subparsers.add_parser("input", help="QE input generation commands")
    inp_sub = inp.add_subparsers(dest="input_command")
    ib = inp_sub.add_parser("build", help="Build a QE input file")
    ib.add_argument("--structure", default=None, help="Path to structure file")
    ib.add_argument("--structure-format", default=None, help="Structure format override")
    ib.add_argument("--fractional", action="store_true", default=False)
    ib.add_argument("--material", default=None, help="Known material shortcut, e.g. Si, Al, MgO")
    ib.add_argument("--type", required=True, choices=["scf", "vc-relax", "relax", "bands", "dos"], help="Calculation type")
    ib.add_argument("--prefix", default=None)
    ib.add_argument("--ecutwfc", type=float, default=None)
    ib.add_argument("--ecutrho", type=float, default=None)
    ib.add_argument("--kpoints", type=int, nargs=6, default=None)
    ib.add_argument("--conv-thr", type=float, default=1.0e-8)
    ib.add_argument("--nstep", type=int, default=200)
    ib.add_argument("--pseudo-dir", default=None)
    ib.add_argument("--output", default=None)

    # job
    job = subparsers.add_parser("job", help="Job lifecycle commands")
    job_sub = job.add_subparsers(dest="job_command")
    js = job_sub.add_parser("submit", help="Submit a QE job")
    js.add_argument("input", help="Path to QE input file")
    js.add_argument("--output", default=None)

    jst = job_sub.add_parser("status", help="Check job status")
    jst.add_argument("job_id", help="Job identifier")
    jst.add_argument("--output", default=None)

    jc = job_sub.add_parser("cancel", help="Cancel a job")
    jc.add_argument("job_id", help="Job identifier")
    jc.add_argument("--output", default=None)

    jf = job_sub.add_parser("fetch", help="Fetch job outputs")
    jf.add_argument("job_id", help="Job identifier")
    jf.add_argument("--output-dir", default=".")
    jf.add_argument("--output", default=None)

    # result
    res = subparsers.add_parser("result", help="Result commands")
    res_sub = res.add_subparsers(dest="result_command")
    rp = res_sub.add_parser("parse", help="Parse QE output into JSON")
    rp.add_argument("input", help="Path to QE output file")
    rp.add_argument("--type", required=True, choices=["vc-relax", "scf", "nscf", "bands", "dos"])
    rp.add_argument("--xml-path", default=None, help="Optional path to data-file-schema.xml")
    rp.add_argument("--output", default=None)

    rv = res_sub.add_parser("verify", help="Verify parsed result")
    rv.add_argument("input", help="Path to parsed result JSON")
    rv.add_argument("--task-type", default=None, choices=["T1", "T2"])
    rv.add_argument("--output", default=None)

    # ledger
    led = subparsers.add_parser("ledger", help="Evidence ledger commands")
    led_sub = led.add_subparsers(dest="ledger_command")
    lr = led_sub.add_parser("record", help="Record a run into the ledger")
    lr.add_argument("--database", default="ledger.db")
    lr.add_argument("--task-id", required=True)
    lr.add_argument("--task-type", required=True, choices=["T1", "T2"])
    lr.add_argument("--attempt", default=1, type=int)
    lr.add_argument("--success", required=True, choices=["true", "false", "1", "0", "yes", "no"])
    lr.add_argument("--input-hash", default="")
    lr.add_argument("--walltime-sec", default=0.0, type=float)
    lr.add_argument("--verifier-passed", required=True, choices=["true", "false", "1", "0", "yes", "no"])
    lr.add_argument("--failure-reasons", default="")
    lr.add_argument("--recovery-actions", default="")
    lr.add_argument("--failure-kind", default="")
    lr.add_argument("--evidence-path", default="")
    lr.add_argument("--output-files", default="")
    lr.add_argument("--output", default=None)

    lq = led_sub.add_parser("query", help="Query ledger runs")
    lq.add_argument("--database", default="ledger.db")
    lq.add_argument("--task-id", default=None)
    lq.add_argument("--limit", default=20, type=int)
    lq.add_argument("--output", default=None)

    lp = led_sub.add_parser("patterns", help="Show failure patterns")
    lp.add_argument("--database", default="ledger.db")
    lp.add_argument("--task-id", default=None)
    lp.add_argument("--output", default=None)

    # doctor / probe-env
    subparsers.add_parser("doctor", help="Probe environment and dependencies")
    subparsers.add_parser("probe-env", help="Alias for doctor")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 0

    try:
        if args.command == "structure":
            if args.structure_command == "import":
                return cmd_structure_import(args)
            if args.structure_command == "validate":
                return cmd_structure_validate(args)
            if args.structure_command == "generate":
                return cmd_structure_generate(args)
            if args.structure_command == "build2d":
                return cmd_structure_build2d(args)
            if args.structure_command == "dope":
                return cmd_structure_dope(args)
            if args.structure_command == "analyze":
                return cmd_structure_analyze(args)
            if args.structure_command == "molecule":
                return cmd_structure_molecule(args)
            if args.structure_command == "reference":
                return cmd_structure_reference(args)
            return _error_response("Unknown structure command")

        if args.command == "thermo":
            if args.thermo_command == "eads":
                return cmd_thermo_eads(args)
            if args.thermo_command == "formation":
                return cmd_thermo_formation(args)
            return _error_response("Unknown thermo command")

        if args.command == "materials":
            if args.materials_command == "list":
                return cmd_materials_list(args)
            return _error_response("Unknown materials command")

        if args.command == "input":
            if args.input_command == "build":
                return cmd_input_build(args)
            return _error_response("Unknown input command")

        if args.command == "job":
            if args.job_command == "submit":
                return cmd_job_submit(args)
            if args.job_command == "status":
                return cmd_job_status(args)
            if args.job_command == "cancel":
                return cmd_job_cancel(args)
            if args.job_command == "fetch":
                return cmd_job_fetch(args)
            return _error_response("Unknown job command")

        if args.command == "result":
            if args.result_command == "parse":
                return cmd_result_parse(args)
            if args.result_command == "verify":
                return cmd_result_verify(args)
            return _error_response("Unknown result command")

        if args.command == "ledger":
            if args.ledger_command == "record":
                return cmd_ledger_record(args)
            if args.ledger_command == "query":
                return cmd_ledger_query(args)
            if args.ledger_command == "patterns":
                return cmd_ledger_patterns(args)
            return _error_response("Unknown ledger command")

        if args.command in {"doctor", "probe-env"}:
            return cmd_doctor(args)

        return _error_response(f"Unknown command: {args.command}")
    except ValueError as exc:
        return _error_response(str(exc), code=2)
    except Exception as exc:  # noqa: BLE001
        return _error_response("Unexpected error", details={"exception": str(exc), "type": type(exc).__name__, "traceback": traceback.format_exc()})


if __name__ == "__main__":
    sys.exit(main())
