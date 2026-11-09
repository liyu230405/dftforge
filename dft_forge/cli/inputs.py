from __future__ import annotations

from pathlib import Path

from ._helpers import _error_response, _json_dumps


def cmd_input_build(args: argparse.Namespace) -> int:
    from dft_forge.catalog import build_dynamic_material_profile, build_dynamic_task_spec, register_dynamic_material
    from dft_forge.compiler import DEFAULT_PSEUDO_DIR, QECompiler, get_kpoints
    from dft_forge.structure import StructureImporter

    calc_type = args.type
    workdir = Path.cwd()
    workdir.mkdir(parents=True, exist_ok=True)

    compiler = QECompiler(pseudo_dir=Path(args.pseudo_dir) if args.pseudo_dir else DEFAULT_PSEUDO_DIR)

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
