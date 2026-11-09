from __future__ import annotations

from pathlib import Path

from ._helpers import _error_response, _json_dumps, _success_response


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
            adsorbate = {
                "element": element.strip(),
                "site": site.strip().lower(),
                "height": args.height,
                "site_index": getattr(args, "site_index", 0),
            }
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



def cmd_structure_supercell(args: argparse.Namespace) -> int:
    """structure supercell: unchanged bulk structure → CIF."""
    from ase.io import write as ase_write

    from dft_forge.structure_builder import StructureBuildError, supercell_3d

    try:
        result = supercell_3d(args.source, supercell=args.supercell)
    except (StructureBuildError, ValueError, FileNotFoundError) as exc:
        return _error_response(str(exc))
    output_path = Path(args.output) if args.output else Path.cwd() / f"supercell_{args.supercell}.cif"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ase_write(str(output_path), result.atoms)
    data = result.to_dict()
    data.update({"command": "structure.supercell", "ok": True, "source": args.source, "output": str(output_path)})
    print(_json_dumps(data))
    return 0



def cmd_structure_vacancy(args: argparse.Namespace) -> int:
    """structure vacancy: remove one selected atom from a bulk supercell."""
    from ase.io import write as ase_write

    from dft_forge.structure_builder import StructureBuildError, vacancy_3d

    try:
        result = vacancy_3d(
            args.source, args.element, supercell=args.supercell, index=args.index,
        )
    except (StructureBuildError, ValueError, FileNotFoundError) as exc:
        return _error_response(str(exc))
    output_path = Path(args.output) if args.output else Path.cwd() / f"vacancy_{args.element}_{args.supercell}.cif"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ase_write(str(output_path), result.atoms)
    data = result.to_dict()
    data.update({
        "command": "structure.vacancy", "ok": True, "source": args.source,
        "element": args.element, "index": data.get("selected_index", args.index),
        "output": str(output_path),
    })
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
    data["molecule_kind"] = str(args.kind).lower().replace("-", "").replace("_", "").replace(" ", "")
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
    from dft_forge.structure_builder import reference_molecule_kind

    kind = reference_molecule_kind(args.element)
    if kind:
        data["molecule_kind"] = kind
    data["element"] = args.element
    data["output"] = str(output_path)
    print(_json_dumps(data))
    return 0
