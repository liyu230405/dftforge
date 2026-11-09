"""Deterministic ABINIT input-file writer.

Produces a single ``.abi`` input (modern ABINIT 9.x syntax, with the
``pseudos`` / ``pp_dirpath`` variables) from a material key or structure file
plus a small typed set of method parameters. No LLM is involved. Missing
pseudopotentials fail closed; nothing is fabricated.

ABINIT works in atomic (Hartree/Bohr) units. Lengths written to ``.abi`` are in
Bohr (``acell``/``rprim`` scaled); energies parsed back are Hartree and
converted to eV in the parser. The structure helpers are reused verbatim from
the VASP writer so both engines resolve materials identically.
"""

from __future__ import annotations

import os
import math
import itertools
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ase import Atoms
from ase.data import atomic_numbers

from dft_forge.compiler import MATERIAL_DB, profile_from_atoms
from dft_forge.engines.vasp.writer import apply_supercell, resolve_atoms

# Å -> Bohr (ABINIT native length unit). CODATA 2018, matches ASE/QE.
ANG_TO_BOHR = 1.8897261246257702


class ABINITInputError(ValueError):
    """Invalid ABINIT input request or unusable pseudopotential set."""


# Conservative, reproducible defaults — initial choices, not a convergence
# certificate (the same contract QE/VASP parameters carry).
_DEFAULT_ECUT_HA = 15.0          # plane-wave cutoff, Hartree
_DEFAULT_TOLDFE_HA = 1.0e-10     # SCF energy tolerance, Hartree
_DEFAULT_TOLMXF_HABOHR = 5.0e-5  # force tolerance for relaxation, Ha/Bohr
_DEFAULT_TSMEAR_HA = 0.01        # metal smearing width, Hartree
_DEFAULT_NGKPT = 8               # default Monkhorst-Pack subdivisions
_DEFAULT_NSTEP = 100             # max SCF iterations

# Per-calc control blocks mirror the QE/VASP calc set:
#   scf      : self-consistent field (iscf 7, no ionic motion)
#   relax    : ionic relaxation, fixed cell        (ionmov 2, optcell 0)
#   vc-relax : ionic + cell relaxation             (ionmov 2, optcell 2)
#   nscf     : non-self-consistent from a WFK       (iscf -2)
#   bands    : nscf along a k-path                  (iscf -2, kptbounds)
#   dos      : nscf dense mesh, total DOS           (prtdos 1)
#   pdos     : nscf dense mesh, projected DOS       (prtdos 3)
#   dfpt     : DFPT response function -> DDB        (rfphon 1)  [phonon chain]
#   anaddb   : post-process DDB -> dispersion/DOS   (separate anaddb binary)
_SCF_LIKE = {"scf", "relax", "vc-relax"}
_NSCF_LIKE = {"nscf", "bands", "dos", "pdos"}  # consume upstream WFK
# Calc types that produce a WFK for downstream reuse.
_WRITE_WFK = {"scf", "relax", "vc-relax"}


def default_pp_dir() -> Optional[Path]:
    """User-supplied pseudopotential library root (PseudoDojo-style)."""
    raw = os.environ.get("DFT_FORGE_ABINIT_PP_DIR", "").strip()
    return Path(raw).expanduser() if raw else None


def _as_float(params: Dict[str, Any], key: str, default: float) -> float:
    val = params.get(key)
    if val is None:
        return default
    try:
        result = float(val)
        if not math.isfinite(result):
            raise ValueError()
        return result
    except (TypeError, ValueError) as exc:
        raise ABINITInputError(f"{key} must be a finite number") from exc


def _as_int(params: Dict[str, Any], key: str, default: int) -> int:
    val = params.get(key)
    if val is None:
        return default
    try:
        result = int(val)
        if isinstance(val, bool) or result != float(val) or result <= 0:
            raise ValueError()
        return result
    except (TypeError, ValueError, OverflowError) as exc:
        raise ABINITInputError(f"{key} must be a positive integer") from exc


def grid_parameter(params, key, default):
    raw = params.get(key)
    raw = default if raw is None else raw
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ABINITInputError(f"{key} must contain three positive integers")
    return [_as_int({key: x}, key, 1) for x in raw]


def phonon_qpoints(params):
    grid = grid_parameter(params, "ngqpt", [2, 2, 2])
    if math.prod(grid) > 64:
        raise ABINITInputError("DFPT q-mesh is limited to 64 points per node")
    return grid, [[i / grid[0], j / grid[1], k / grid[2]]
                  for i, j, k in itertools.product(*(range(n) for n in grid))]


def species_order(atoms: Atoms) -> List[str]:
    """Distinct species in first-appearance order.

    ABINIT groups atoms by type via ``typat`` (1-based indices into ``znucl``);
    this order defines the znucl/typat/pseudos correspondence, so it must be
    stable and shared across znucl, typat and the pseudo list.
    """
    order: List[str] = []
    for sym in atoms.get_chemical_symbols():
        if sym not in order:
            order.append(sym)
    return order


def pseudo_candidates(element: str) -> List[str]:
    """Pseudopotential filenames to try for an element, most-common first.

    Covers PseudoDojo NC (``Si.psp8``), PAW-XML (``Si.xml``, ``Si.GGA_PBE.xml``)
    and legacy (``Si.pspnc``) naming. Case of the extension is not assumed.
    """
    return [
        f"{element}.psp8",
        f"{element}.xml",
        f"{element}.GGA_PBE.xml",
        f"{element}.LDA.xml",
        f"{element}.pspnc",
        f"{element}.fhi",
        f"{element}.upf",
    ]


def resolve_pseudos(species: List[str], pp_dir: Optional[Path]) -> List[Path]:
    """Locate one pseudopotential per species, in znucl order. Fail closed."""
    if pp_dir is None:
        raise ABINITInputError(
            "ABINIT pseudopotentials are not bundled. Set DFT_FORGE_ABINIT_PP_DIR "
            "to a directory of pseudopotentials (e.g. a PseudoDojo NC set of "
            "<Element>.psp8 files). None are fabricated."
        )
    if not pp_dir.is_dir():
        raise ABINITInputError(f"DFT_FORGE_ABINIT_PP_DIR is not a directory: {pp_dir}")
    resolved: List[Path] = []
    missing: List[str] = []
    for element in species:
        found: Optional[Path] = None
        for name in pseudo_candidates(element):
            candidate = pp_dir / name
            if candidate.is_file():
                found = candidate
                break
        if found is None:
            missing.append(element)
        else:
            resolved.append(found.resolve())
    if missing:
        raise ABINITInputError(
            "missing pseudopotential for: " + ", ".join(missing)
            + f" (looked under {pp_dir}); provide real pseudos, none are fabricated"
        )
    return resolved


def _geometry_block(atoms: Atoms, species: List[str]) -> List[str]:
    """acell/rprim/znucl/typat/xred block in Bohr, atomic units."""
    cell = atoms.get_cell()
    cell_arr = cell.array if hasattr(cell, "array") else cell
    znucl = [atomic_numbers[el] for el in species]
    # typat: 1-based index into species order, per atom, in structure order.
    sym_to_type = {el: i + 1 for i, el in enumerate(species)}
    typat = [sym_to_type[s] for s in atoms.get_chemical_symbols()]
    xred = atoms.get_scaled_positions(wrap=False)

    lines = [
        "# --- structure (atomic units, lengths in Bohr) ---",
        "acell 1.0 1.0 1.0",  # rprim already carries the absolute Bohr lengths
        "rprim",
    ]
    for row in cell_arr:
        lines.append("  " + "  ".join(f"{float(x) * ANG_TO_BOHR:.10f}" for x in row))
    lines.append(f"natom {len(atoms)}")
    lines.append(f"ntypat {len(species)}")
    lines.append("znucl " + " ".join(str(z) for z in znucl))
    lines.append("typat " + " ".join(str(t) for t in typat))
    lines.append("xred")
    for pos in xred:
        lines.append("  " + "  ".join(f"{float(x):.10f}" for x in pos))
    return lines


def _kpoint_block(params: Dict[str, Any], calc: str, atoms: Optional[Atoms]) -> List[str]:
    """Monkhorst-Pack mesh, or a line-mode band path for calc='bands'."""
    if calc == "bands":
        return _band_kpoints(params, atoms)
    ngkpt = params.get("ngkpt") if params.get("ngkpt") is not None else params.get("kpoints")
    if ngkpt is not None:
        grid = grid_parameter({"ngkpt": ngkpt}, "ngkpt", [8, 8, 8])
    else:
        n = _as_int(params, "ngkpt_density", _DEFAULT_NGKPT)
        # nscf/dos want a denser mesh than the scf that produced the density.
        if calc in ("dos", "pdos", "nscf"):
            n = max(n, int(round(n * 1.5)))
        grid = [n, n, n]
    return [
        "# --- k-points ---",
        "kptopt 1",
        f"ngkpt {grid[0]} {grid[1]} {grid[2]}",
        "nshiftk 1",
        "shiftk 0.0 0.0 0.0",
    ]


def _band_kpoints(params: Dict[str, Any], atoms: Optional[Atoms]) -> List[str]:
    """Line-mode band path via ASE's high-symmetry bandpath -> kptbounds."""
    if atoms is None:
        raise ABINITInputError("bands calc needs atoms to derive a k-path")
    pbc = tuple(bool(b) for b in atoms.pbc) if len(atoms.pbc) == 3 else None
    bandpath = atoms.cell.bandpath(pbc=pbc)
    special = bandpath.special_points
    path_str = bandpath.path.replace(",", "")
    pts = [special[ch] for ch in path_str if ch in special]
    if len(pts) < 2:
        raise ABINITInputError("could not derive a band k-path from the cell")
    ndivsm = _as_int(params, "ndivsm", 10)
    lines = [
        "# --- band path (line mode) ---",
        "iscf -2",
        "kptopt " + str(-(len(pts) - 1)),  # negative: number of path segments
        f"ndivsm {ndivsm}",
        f"nband {_as_int(params, 'nband', 8)}",
        "kptbounds",
    ]
    for p in pts:
        lines.append("  " + "  ".join(f"{float(x):.8f}" for x in p))
    return lines


def _magnetism_block(params: Dict[str, Any], atoms: Atoms, species: List[str]) -> List[str]:
    """nsppol/spinat for FM/AFM initial moments (ABINIT collinear spin)."""
    order = str(params.get("magnetic_order") or "").lower().replace("-", "")
    spin_polarized = bool(params.get("spin_polarized")) or order in {"fm", "afm"}
    if not spin_polarized:
        return []
    moment = _as_float(params, "initial_moment_muB", 2.0)
    symbols = atoms.get_chemical_symbols()
    spinat: List[str] = []
    sign = 1.0
    for idx, _sym in enumerate(symbols):
        if order == "afm":
            sign = 1.0 if idx % 2 == 0 else -1.0
        spinat.append(f"  0.0 0.0 {sign * moment:.3f}")
    return ["# --- collinear spin ---", "nsppol 2", "spinat"] + spinat


def build_abi(
    calc: str,
    params: Dict[str, Any],
    atoms: Atoms,
    is_metal: bool,
    species: List[str],
) -> str:
    """Assemble a full ``.abi`` input for one calculation."""
    if calc not in (_SCF_LIKE | _NSCF_LIKE | {"dfpt"}):
        raise ABINITInputError(f"unknown ABINIT calc type '{calc}'")

    ecut = _as_float(params, "ecut", _DEFAULT_ECUT_HA)
    if ecut <= 0:
        raise ABINITInputError("ecut must be positive (Hartree)")
    nstep = _as_int(params, "nstep", _DEFAULT_NSTEP)
    lines: List[str] = [f"# ABINIT input: calc={calc} (generated, no LLM)", ""]
    lines += _geometry_block(atoms, species)
    lines.append("")
    lines += _kpoint_block(params, calc, atoms)
    lines.append("")
    lines.append("# --- plane-wave basis & electronic structure ---")
    lines.append(f"ecut {ecut:g}")
    lines.append(f"nstep {nstep}")
    if params.get("nband") is not None and calc != "bands":
        lines.append(f"nband {_as_int(params, 'nband', 8)}")

    # Metals need smearing (occopt 3-7 + tsmear); insulators use fixed
    # occupations (occopt 1). Mirrors QE's smearing/fixed choice.
    if is_metal:
        lines.append("occopt 7")
        lines.append(f"tsmear {_as_float(params, 'tsmear', _DEFAULT_TSMEAR_HA):g}")
    else:
        lines.append("occopt 1")

    lines += _magnetism_block(params, atoms, species)

    # Per-calc control blocks.
    if calc in _SCF_LIKE:
        lines.append(f"toldfe {_as_float(params, 'toldfe', _DEFAULT_TOLDFE_HA):g}")
        lines += ["prtden 1", "prtwf 1"]
    if calc == "relax":
        lines += ["ionmov 2", "optcell 0",
                  f"ntime {_as_int(params, 'ntime', 50)}",
                  f"tolmxf {_as_float(params, 'tolmxf', _DEFAULT_TOLMXF_HABOHR):g}"]
    elif calc == "vc-relax":
        lines += ["ionmov 2", "optcell 2",
                  f"ntime {_as_int(params, 'ntime', 50)}",
                  f"tolmxf {_as_float(params, 'tolmxf', _DEFAULT_TOLMXF_HABOHR):g}",
                  f"ecutsm {_as_float(params, 'ecutsm', 0.5):g}",
                  f"dilatmx {_as_float(params, 'dilatmx', 1.1):g}"]
    elif calc == "nscf":
        lines += ["iscf -2", f"tolwfr {_as_float(params, 'tolwfr', 1.0e-18):g}",
                  'getden_filepath "upstream_DEN"']
    elif calc == "bands":
        # kptopt/iscf already emitted in the band k-point block.
        lines += [f"tolwfr {_as_float(params, 'tolwfr', 1.0e-18):g}", 'getden_filepath "upstream_DEN"']
    elif calc == "dos":
        lines += ["iscf -2", "prtdos 2", 'getden_filepath "upstream_DEN"',
                  f"tolwfr {_as_float(params, 'tolwfr', 1.0e-18):g}"]
    elif calc == "pdos":
        lines += ["iscf -2", "prtdos 3", 'getden_filepath "upstream_DEN"',
                  f"tolwfr {_as_float(params, 'tolwfr', 1.0e-18):g}",
                  f"natsph {len(atoms)}",
                  "iatsph " + " ".join(str(i + 1) for i in range(len(atoms))),
                  "ratsph " + " ".join(f"{_as_float(params, 'ratsph', 2.0):g}" for _ in species)]
    elif calc == "dfpt":
        # Response-function (phonon) datasets on a q-mesh; ndtset drives the
        # GS + perturbation sequence. Kept in one .abi so a single abinit run
        # emits the DDB the anaddb node consumes.
        lines += _dfpt_block(params, len(atoms))
    return "\n".join(lines) + "\n"


def _dfpt_block(params: Dict[str, Any], natoms: int) -> List[str]:
    """Ground state followed by every commensurate q point, inside one node."""
    grid, qpoints = phonon_qpoints(params)
    kgrid = grid_parameter(params, "ngkpt", params.get("kpoints") or [8, 8, 8])
    if any(k % q for k, q in zip(kgrid, grid)):
        raise ABINITInputError("DFPT ngkpt must be divisible by ngqpt in each direction")
    tol = _as_float(params, "tolvrs", 1.0e-12)
    lines = ["# --- complete uniform DFPT q-mesh ---", f"ndtset {len(qpoints) + 1}",
             "iscf1 7", f"tolvrs1 {tol:g}", "prtwf1 1", "prtden1 1"]
    for index, q in enumerate(qpoints, 2):
        lines += [f"getwfk{index} 1", f"rfphon{index} 1", f"rfatpol{index} 1 {natoms}",
                  f"rfdir{index} 1 1 1", f"kptopt{index} {2 if index == 2 else 3}",
                  f"nqpt{index} 1", f"qpt{index} " + " ".join(f"{v:.12g}" for v in q),
                  f"tolvrs{index} {tol:g}", f"prtwf{index} 0", f"prtden{index} 0"]
    return lines


def _pseudo_block(species: List[str], pseudos: List[Path]) -> List[str]:
    """pp_dirpath + pseudos variables (ABINIT 9.x). pseudos order == znucl."""
    # A common parent directory keeps pp_dirpath meaningful; fall back to '.'.
    parents = {p.parent for p in pseudos}
    if len(parents) == 1:
        dirpath = str(next(iter(parents)))
        names = [p.name for p in pseudos]
    else:
        dirpath = ""
        names = [str(p) for p in pseudos]
    quoted = ", ".join(names)
    lines = ["# --- pseudopotentials (real files, never fabricated) ---"]
    if dirpath:
        lines.append(f'pp_dirpath "{dirpath}"')
    lines.append(f'pseudos "{quoted}"')
    return lines


def write_inputs(
    material: str,
    structure: Optional[str],
    calc: str,
    params: Dict[str, Any],
    workdir: Path,
    pp_dir: Optional[Path] = None,
    atoms: Optional[Atoms] = None,
) -> Dict[str, Any]:
    """Write ``<label>.abi`` into workdir. Returns a metadata dict.

    ``atoms`` may be supplied directly (e.g. a relaxed structure recovered from
    an upstream output); otherwise it is resolved from material/structure.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    if atoms is None:
        atoms, label = resolve_atoms(material, structure)
    else:
        label = material or (Path(str(structure)).stem if structure else "calc")
    atoms = apply_supercell(atoms, params.get("supercell"))
    profile = (MATERIAL_DB[material] if material in MATERIAL_DB and not structure
               else profile_from_atoms(atoms))
    is_metal = bool(params.get("is_metal", profile.get("is_metal", True)))

    species = species_order(atoms)
    pseudos = resolve_pseudos(species, pp_dir if pp_dir is not None else default_pp_dir())
    pseudo_sources = {el: str(p) for el, p in zip(species, pseudos)}

    body = build_abi(calc, params, atoms, is_metal, species)
    abi = body + "\n" + "\n".join(_pseudo_block(species, pseudos)) + "\n"
    abi_path = workdir / f"{label}.abi"
    abi_path.write_text(abi)
    from ase.io import write as ase_write
    ase_write(str(workdir / "input_structure.cif"), atoms, format="cif")
    effective = {"ecut": _as_float(params, "ecut", _DEFAULT_ECUT_HA),
                 "ngkpt": None if calc == "bands" else [int(x) for x in next(
                     line for line in body.splitlines() if line.startswith("ngkpt ")).split()[1:]],
                 "toldfe": _as_float(params, "toldfe", _DEFAULT_TOLDFE_HA) if calc in _SCF_LIKE else None,
                 "tolmxf": _as_float(params, "tolmxf", _DEFAULT_TOLMXF_HABOHR) if calc in {"relax", "vc-relax"} else None,
                 "tsmear": _as_float(params, "tsmear", _DEFAULT_TSMEAR_HA) if is_metal else None,
                 "ecut_unit": "Ha", "occupation_mode": "smearing" if is_metal else "fixed"}

    order = str(params.get("magnetic_order") or "").lower().replace("-", "")
    return {
        "label": label,
        "calc": calc,
        "species": species,
        "is_metal": is_metal,
        "natoms": len(atoms),
        "pseudo_sources": pseudo_sources,
        "abi_file": str(abi_path),
        "spin_polarized": bool(params.get("spin_polarized")) or order in {"fm", "afm"},
        "magnetic_order": params.get("magnetic_order"),
        "supercell": params.get("supercell"),
        "effective_parameters": effective,
        "input_structure_cif": str(workdir / "input_structure.cif"),
        "atom_symbols": atoms.get_chemical_symbols(),
    }


def write_anaddb_input(params: Dict[str, Any], workdir: Path, label: str,
                       atoms: Optional[Atoms]) -> Path:
    """Write an anaddb input that interpolates the DDB onto a band q-path plus
    a q-mesh for the phonon DOS. Returns the input-file path."""
    workdir = Path(workdir)
    grid, _ = phonon_qpoints(params)
    dgrid = grid_parameter(params, "dos_ngqpt", [8, 8, 8])
    lines = [
        "# anaddb input (generated): phonon band structure + DOS from DDB",
        'ddb_filepath "in.ddb"',
        f'output_file "{label}_anaddb.out"',
        f'outdata_prefix "{label}_anaddb"',
        "ifcflag 1",
        f"ngqpt {grid[0]} {grid[1]} {grid[2]}",
        "nqshft 1", "q1shft 0 0 0", "asr 2", "chneut 0", "dipdip 0",
        "# phonon DOS",
        "prtdos 1",
        f"ng2qpt {dgrid[0]} {dgrid[1]} {dgrid[2]}",
        "dossmear 4.5d-6",
    ]
    # A band q-path from the cell, when available.
    if atoms is not None:
        try:
            pbc = tuple(bool(b) for b in atoms.pbc) if len(atoms.pbc) == 3 else None
            bandpath = atoms.cell.bandpath(pbc=pbc, npoints=_as_int(params, "path_points", 81))
            pts = bandpath.kpts
            if len(pts) >= 2:
                lines += ["nph1l " + str(len(pts)), "qph1l"]
                for p in pts:
                    lines.append("  " + "  ".join(f"{float(x):.6f}" for x in p) + " 1.0")
        except Exception:
            pass
    path = workdir / f"{label}_anaddb.in"
    path.write_text("\n".join(lines) + "\n")
    return path
