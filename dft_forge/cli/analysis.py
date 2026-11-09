from __future__ import annotations

from pathlib import Path

from ._helpers import _error_response, _success_response


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
