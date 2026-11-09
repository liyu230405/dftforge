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

Command implementations live in focused submodules; this package re-exports
the public command surface so ``from dft_forge.cli import cmd_...`` keeps
working.
"""

from __future__ import annotations

import argparse
import sys
import traceback

from ._backend import _resolve_executor
from ._helpers import _error_response
from .analysis import _analyze_structure, cmd_structure_analyze
from .doctor import cmd_doctor
from .inputs import cmd_input_build
from .jobs import (
    _local_job_id,
    cmd_job_cancel,
    cmd_job_fetch,
    cmd_job_status,
    cmd_job_submit,
)
from .ledger import cmd_ledger_patterns, cmd_ledger_query, cmd_ledger_record
from .materials import cmd_materials_list
from .results import cmd_result_parse, cmd_result_verify
from .structure import (
    cmd_structure_build2d,
    cmd_structure_dope,
    cmd_structure_generate,
    cmd_structure_import,
    cmd_structure_molecule,
    cmd_structure_reference,
    cmd_structure_supercell,
    cmd_structure_vacancy,
    cmd_structure_validate,
)
from .thermo import RY_TO_EV, cmd_thermo_eads, cmd_thermo_formation

def _add_backend_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--backend", default="local", choices=["local", "ssh"], help="Execution backend")
    p.add_argument("--backend-config", default=None, help="Path to backend JSON config")



def main() -> int:
    parser = argparse.ArgumentParser(
        prog="dft-forge",
        description="DFT-Forge: agent-friendly Quantum ESPRESSO CLI",
    )
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
    b2d.add_argument("--site-index", type=int, default=0, help="Reference atom ordinal in the topmost surface layer")
    b2d.add_argument("--vacuum", type=float, default=15.0)
    b2d.add_argument("--output", default=None, help="Output CIF path")

    dop = struct_sub.add_parser("dope", help="Dope a bulk crystal: supercell + substitution (e.g. P in Si)")
    dop.add_argument("source", help="Material name, formula (CaTiO3), or CIF/POSCAR path")
    dop.add_argument("element", help="Dopant element, e.g. P")
    dop.add_argument("--supercell", default="2x2x2", help="Supercell spec, e.g. 2x2x2 (larger = lower concentration)")
    dop.add_argument("--index", type=int, default=0, help="Atom index to substitute in the supercell")
    dop.add_argument("--output", default=None, help="Output CIF path")

    sup = struct_sub.add_parser("supercell", help="Build an unchanged bulk supercell")
    sup.add_argument("source", help="Material name, formula, or CIF/POSCAR path")
    sup.add_argument("--supercell", default="2x2x2", help="Supercell spec, e.g. 2x2x2")
    sup.add_argument("--output", default=None, help="Output CIF path")

    vac = struct_sub.add_parser("vacancy", help="Remove one atom from a bulk supercell")
    vac.add_argument("source", help="Material name, formula, or CIF/POSCAR path")
    vac.add_argument("element", help="Element to remove, e.g. O")
    vac.add_argument("--supercell", default="2x2x2", help="Supercell spec, e.g. 2x2x2")
    vac.add_argument("--index", type=int, default=None, help="Absolute atom index after supercell construction")
    vac.add_argument("--output", default=None, help="Output CIF path")

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

    # job — --backend lives on each job subcommand (argparse rejects it after
    # the subcommand when declared on the main parser, which is where the
    # docs tell users to put it)
    job = subparsers.add_parser("job", help="Job lifecycle commands")
    job_sub = job.add_subparsers(dest="job_command")
    js = job_sub.add_parser("submit", help="Submit a QE job")
    js.add_argument("input", help="Path to QE input file")
    _add_backend_args(js)
    js.add_argument("--output", default=None)

    jst = job_sub.add_parser("status", help="Check job status")
    jst.add_argument("job_id", help="Job identifier (SSH: scheduler job id)")
    _add_backend_args(jst)
    jst.add_argument("--output", default=None)

    jc = job_sub.add_parser("cancel", help="Cancel a job")
    jc.add_argument("job_id", help="Job identifier (SSH: scheduler job id)")
    _add_backend_args(jc)
    jc.add_argument("--output", default=None)

    jf = job_sub.add_parser("fetch", help="Fetch job outputs")
    jf.add_argument("job_id", help="Job identifier (SSH: remote job directory)")
    _add_backend_args(jf)
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
            if args.structure_command == "supercell":
                return cmd_structure_supercell(args)
            if args.structure_command == "vacancy":
                return cmd_structure_vacancy(args)
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
