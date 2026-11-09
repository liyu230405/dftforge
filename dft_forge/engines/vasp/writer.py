"""Deterministic VASP input-file writer.

Produces INCAR, POSCAR, KPOINTS and (by concatenation) POTCAR from a material
key or structure file plus a small typed set of method parameters. No LLM is
involved. Missing POTCARs fail closed; nothing is fabricated.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ase import Atoms

from dft_forge.compiler import (
    MATERIAL_DB,
    build_atoms,
    formula_atoms,
    profile_from_atoms,
    read_structure,
)


class VASPInputError(ValueError):
    """Invalid VASP input request or unusable pseudopotential set."""


# Conservative, reproducible defaults. These are *initial* choices, not a
# convergence certificate — the same contract QE parameters carry.
_DEFAULT_ENCUT_EV = 520.0
_DEFAULT_KSPACING = 0.30  # 1/Angstrom; VASP-native automatic k-mesh density
_DEFAULT_EDIFF = 1.0e-6
_DEFAULT_EDIFFG = -0.02  # eV/Angstrom force threshold for relaxation

# IBRION/ISIF per calc type mirror QE's calc set:
#   scf      : static self-consistent field (no ionic motion)
#   relax    : ionic relaxation, fixed cell           (QE relax)
#   vc-relax : ionic + cell shape + volume relaxation (QE vc-relax)
#   nscf     : non-self-consistent from a fixed CHGCAR (QE nscf)
#   bands    : nscf along a k-path                     (QE bands)
#   dos      : nscf on a dense mesh, total DOS         (QE dos)
#   pdos     : nscf on a dense mesh, projected DOS     (QE pdos, LORBIT=11)
#   phonon   : Gamma-point DFPT force constants        (QE ph, IBRION=8)
# icharg=11 means "read a frozen CHGCAR and do not update it" — the VASP
# analogue of QE reusing an upstream {prefix}.save for nscf/bands/dos.
_CALC_MODES = {
    "scf": {"ibrion": -1, "nsw": 0, "isif": 2},
    "relax": {"ibrion": 2, "nsw": 100, "isif": 2},
    "vc-relax": {"ibrion": 2, "nsw": 100, "isif": 3},
    "nscf": {"ibrion": -1, "nsw": 0, "isif": 2, "icharg": 11},
    "bands": {"ibrion": -1, "nsw": 0, "isif": 2, "icharg": 11},
    "dos": {"ibrion": -1, "nsw": 0, "isif": 2, "icharg": 11, "lorbit": 11},
    "pdos": {"ibrion": -1, "nsw": 0, "isif": 2, "icharg": 11, "lorbit": 11},
    "phonon": {"ibrion": 8, "nsw": 1, "isif": 2},
}

# Calc types that must not persist charge/wavefunctions vs. those that must.
_WRITE_CHGCAR = {"scf", "relax", "vc-relax"}  # produce CHGCAR for downstream
_READ_CHGCAR = {"nscf", "bands", "dos", "pdos"}  # consume upstream CHGCAR


def default_pp_dir() -> Optional[Path]:
    """User-supplied POTCAR library root (VASP_PP_PATH-style)."""
    raw = os.environ.get("DFT_FORGE_VASP_PP_DIR", "").strip()
    return Path(raw).expanduser() if raw else None


def resolve_atoms(material: str, structure: Optional[str]) -> Tuple[Atoms, str]:
    """Resolve a material key or structure path into ASE Atoms + a label."""
    if structure:
        atoms = read_structure(structure)
        label = Path(str(structure)).stem.replace(" ", "_") or material or "calc"
        return atoms, label
    if material and material in MATERIAL_DB:
        return build_atoms(material), material
    if material:
        suffix = Path(material).suffix.lower()
        if suffix in (".cif", ".poscar", ".xyz", ".vasp"):
            atoms = read_structure(material)
            return atoms, Path(material).stem.replace(" ", "_")
        atoms, _proto = formula_atoms(material)
        return atoms, material
    raise VASPInputError("VASP node needs a 'material' key or a 'structure' file")


def apply_supercell(atoms: Atoms, spec: Optional[str]) -> Atoms:
    """Expand to an m×n×p supercell when a non-trivial spec is given."""
    if not spec or str(spec) in {"1x1x1", "1x1x1.0"}:
        return atoms
    from ase.build import make_supercell
    from dft_forge.structure_builder import parse_supercell_3d

    m, n, p = parse_supercell_3d(str(spec))
    return make_supercell(atoms, [[m, 0, 0], [0, n, 0], [0, 0, p]])


def poscar_species_order(atoms: Atoms) -> List[str]:
    """POSCAR groups atoms by species in first-appearance order; POTCAR must
    concatenate in exactly this order or VASP silently mislabels elements."""
    order: List[str] = []
    for sym in atoms.get_chemical_symbols():
        if sym not in order:
            order.append(sym)
    return order


def build_incar(
    calc: str,
    params: Dict[str, Any],
    is_metal: bool,
    atoms: Optional[Atoms] = None,
    species: Optional[List[str]] = None,
    nelect: Optional[float] = None,
) -> str:
    """Render a deterministic INCAR for the requested calc type.

    ``atoms``/``species`` are needed for spin (per-atom MAGMOM); ``nelect`` (a
    non-neutral electron count) is passed through for charged-cell doping.
    """
    if calc not in _CALC_MODES:
        raise VASPInputError(
            f"unsupported VASP calc '{calc}'; expected one of {sorted(_CALC_MODES)}"
        )
    mode = _CALC_MODES[calc]
    encut = _as_float(params, "encut", _DEFAULT_ENCUT_EV)
    ediff = _as_float(params, "ediff", _DEFAULT_EDIFF)
    lines = [
        "# Generated by DFT-Forge (deterministic; no LLM in input path)",
        f"SYSTEM = {params.get('prefix', calc)}",
        "PREC = Accurate",
        f"ENCUT = {encut:g}",
        f"EDIFF = {ediff:g}",
        f"IBRION = {mode['ibrion']}",
        f"NSW = {mode['nsw']}",
        f"ISIF = {mode['isif']}",
        "LREAL = .FALSE.",
    ]
    # Charge/wavefunction persistence: producers write CHGCAR so consumers
    # (nscf/bands/dos/pdos) can restart from it; consumers freeze it (ICHARG=11).
    lines.append("LCHARG = .TRUE." if calc in _WRITE_CHGCAR else "LCHARG = .FALSE.")
    lines.append("LWAVE = .FALSE.")
    if "icharg" in mode:
        lines.append(f"ICHARG = {mode['icharg']}")
    if mode["nsw"] > 0 and calc != "phonon":
        ediffg = _as_float(params, "ediffg", _DEFAULT_EDIFFG)
        lines.append(f"EDIFFG = {ediffg:g}")

    # Smearing: DOS-type runs want the tetrahedron method (ISMEAR=-5) for an
    # accurate integrated DOS; SCF/relax use MP (metal) or Gaussian (insulator).
    if calc in ("dos", "pdos", "bands"):
        lines += ["ISMEAR = -5"]
    elif is_metal:
        lines += ["ISMEAR = 1", f"SIGMA = {_as_float(params, 'sigma', 0.2):g}"]
    else:
        lines += ["ISMEAR = 0", f"SIGMA = {_as_float(params, 'sigma', 0.05):g}"]

    if "lorbit" in mode:
        lines.append(f"LORBIT = {mode['lorbit']}")
        lines.append(f"NEDOS = {_as_int(params, 'nedos', 3000)}")

    # Spin polarization / magnetic order (ISPIN=2 + per-atom MAGMOM).
    magmom = _magmom_line(params, atoms, species)
    if magmom is not None:
        lines += ["ISPIN = 2", magmom]

    # Charged-cell doping: NELECT != neutral count. The neutral count itself is
    # computed by the caller from POTCAR ZVAL (deterministic, no fabrication).
    if nelect is not None:
        lines.append(f"NELECT = {nelect:g}")

    return "\n".join(lines) + "\n"


def _as_int(params: Dict[str, Any], key: str, default: int) -> int:
    value = params.get(key)
    return int(value) if value is not None else default


def _magmom_line(
    params: Dict[str, Any], atoms: Optional[Atoms], species: Optional[List[str]]
) -> Optional[str]:
    """Build a per-atom MAGMOM line for FM/AFM order, in POSCAR (grouped) order.

    Returns None when the run is non-magnetic. AFM alternates the sign on the
    magnetic element; FM gives every magnetic atom the same moment.
    """
    order = str(params.get("magnetic_order") or "").strip().lower().replace("-", "")
    spin = bool(params.get("spin_polarized")) or order in {"fm", "afm"}
    if not spin:
        return None
    if atoms is None:
        # spin requested but no geometry to place moments on: a bare ISPIN=2
        # with VASP's default MAGMOM (1 per atom) still polarizes.
        return "# MAGMOM defaulted (no per-atom placement available)"
    moment = _as_float(params, "initial_moment_muB", 5.0)
    mag_el = str(params.get("magnetic_element") or "").strip()
    symbols = atoms.get_chemical_symbols()
    grouped = species or poscar_species_order(atoms)
    counts = {el: symbols.count(el) for el in grouped}
    tokens: List[str] = []
    for el in grouped:
        n = counts[el]
        magnetic = (not mag_el) or el == mag_el
        if not magnetic:
            tokens.append(f"{n}*0")
            continue
        if order == "afm" and n >= 2:
            half = n // 2
            tokens.append(f"{half}*{moment:g} {n - half}*{-moment:g}")
        else:
            tokens.append(f"{n}*{moment:g}")
    return "MAGMOM = " + " ".join(tokens)


def build_kpoints(params: Dict[str, Any], calc: str = "scf", atoms: Optional[Atoms] = None) -> str:
    """Render KPOINTS for the calc type.

    bands            -> line mode along the ASE-derived high-symmetry path
    dos/pdos/nscf    -> denser explicit/automatic mesh than SCF
    scf/relax/vc     -> explicit Monkhorst-Pack, else KSPACING length-mode
    """
    if calc == "bands":
        return _band_kpoints(params, atoms)

    kpts = params.get("kpoints")
    # DOS-type runs want a denser mesh; when the user gave none, scale the
    # length-mode density up so the integrated DOS is smooth.
    if kpts:
        grid = [int(x) for x in list(kpts)[:3]]
        if len(grid) != 3 or any(g <= 0 for g in grid):
            raise VASPInputError(f"invalid kpoints grid: {kpts!r}")
        if calc in ("dos", "pdos", "nscf"):
            factor = 2 if calc in ("dos", "pdos") else 1
            grid = [max(1, g * factor) for g in grid]
        return (
            "Automatic mesh\n0\nGamma\n"
            f"{grid[0]} {grid[1]} {grid[2]}\n0 0 0\n"
        )
    kspacing = _as_float(params, "kspacing", _DEFAULT_KSPACING)
    if calc in ("dos", "pdos"):
        kspacing = _as_float(params, "kspacing", _DEFAULT_KSPACING) / 1.5
    length = max(1, round(2.0 * 3.14159265 / kspacing))
    return f"Automatic mesh\n0\nAuto\n{length}\n"


def _band_kpoints(params: Dict[str, Any], atoms: Optional[Atoms]) -> str:
    """Line-mode KPOINTS along the standard high-symmetry path (ASE bandpath)."""
    n = _as_int(params, "nkpoints_bands", 40)
    if atoms is None:
        raise VASPInputError("bands KPOINTS require a structure to derive the k-path")
    pbc = tuple(bool(b) for b in atoms.pbc) if len(atoms.pbc) == 3 else None
    bandpath = atoms.cell.bandpath(npoints=max(20, n), pbc=pbc)
    kpts = bandpath.kpts
    special = bandpath.special_points
    # map each special point coord -> label for readable segment comments
    lines = ["Line-mode band path (DFT-Forge)", str(max(1, n)), "Line-mode", "Reciprocal"]
    # Build contiguous segments from the path string (e.g. 'GXWKGLUWLK,UX')
    path_str = bandpath.path
    label_coords = {k: v for k, v in special.items()}
    seg_lines: List[str] = []
    prev: Optional[str] = None
    for ch in path_str:
        if ch == ",":
            prev = None
            continue
        if ch not in label_coords:
            continue
        coord = label_coords[ch]
        tag = "Γ" if ch.upper() in ("G", "GAMMA") else ch
        if prev is not None:
            pc = label_coords[prev]
            ptag = "Γ" if prev.upper() in ("G", "GAMMA") else prev
            seg_lines.append(f"  {pc[0]:.6f} {pc[1]:.6f} {pc[2]:.6f}  ! {ptag}")
            seg_lines.append(f"  {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}  ! {tag}")
            seg_lines.append("")
        prev = ch
    if not seg_lines:
        raise VASPInputError(f"could not build a band path from {path_str!r}")
    return "\n".join(lines + seg_lines).rstrip() + "\n"


def zval_from_potcar(potcar_path: Path) -> float:
    """Sum ZVAL (valence electrons) across all concatenated POTCAR blocks.

    VASP prints ``ZVAL   =   4.000`` once per species block; the neutral
    electron count is the per-species ZVAL times the atom count. This reads
    only the licensed POTCAR the user supplied; nothing is fabricated.
    """
    text = potcar_path.read_text(errors="ignore")
    return sum(float(m) for m in re.findall(r"ZVAL\s*=\s*(-?\d+\.?\d*)", text))


def _as_float(params: Dict[str, Any], key: str, default: float) -> float:
    value = params.get(key)
    return float(value) if value is not None else default


def write_poscar(atoms: Atoms, path: Path) -> List[str]:
    """Write POSCAR via ASE (vasp5 format) and return the species order used."""
    from ase.io import write as ase_write

    order = poscar_species_order(atoms)
    # ASE sorts by species internally for vasp5; we pass sort=True and then
    # re-read the species order from the written file to stay authoritative.
    ase_write(str(path), atoms, format="vasp", direct=True, sort=True, vasp5=True)
    return _poscar_species_from_file(path)


def _poscar_species_from_file(path: Path) -> List[str]:
    """Line 6 of a vasp5 POSCAR is the whitespace-separated species order."""
    lines = path.read_text().splitlines()
    if len(lines) < 7:
        raise VASPInputError(f"malformed POSCAR written at {path}")
    species = lines[5].split()
    if not species or not all(tok.isalpha() for tok in species):
        raise VASPInputError(f"POSCAR line 6 is not a species list: {lines[5]!r}")
    return species


def potcar_candidates(element: str) -> List[str]:
    """Subdirectory names to try for an element, most-specific first. Mirrors
    common VASP PP layouts (PBE PAW folders like ``Fe_pv``, ``Li_sv``)."""
    return [element, f"{element}_pv", f"{element}_sv", f"{element}_h", f"{element}_d"]


def resolve_potcars(species: List[str], pp_dir: Optional[Path]) -> List[Path]:
    """Locate one POTCAR per species, in POSCAR order. Fail closed."""
    if pp_dir is None:
        raise VASPInputError(
            "VASP pseudopotentials are license-restricted and not bundled. "
            "Set DFT_FORGE_VASP_PP_DIR to a directory of PAW POTCARs "
            "(one subfolder per species, each containing a POTCAR file)."
        )
    if not pp_dir.is_dir():
        raise VASPInputError(f"DFT_FORGE_VASP_PP_DIR is not a directory: {pp_dir}")
    resolved: List[Path] = []
    missing: List[str] = []
    for element in species:
        found: Optional[Path] = None
        for name in potcar_candidates(element):
            candidate = pp_dir / name / "POTCAR"
            if candidate.is_file():
                found = candidate
                break
        if found is None:
            missing.append(element)
        else:
            resolved.append(found)
    if missing:
        raise VASPInputError(
            "missing POTCAR for: " + ", ".join(missing)
            + f" (looked under {pp_dir}); provide licensed POTCARs, none are fabricated"
        )
    return resolved


def write_potcar(species: List[str], pp_dir: Optional[Path], path: Path) -> Dict[str, str]:
    """Concatenate per-species POTCARs in POSCAR order; return source map."""
    sources = resolve_potcars(species, pp_dir)
    parts, provenance = [], {}
    for element, src in zip(species, sources):
        parts.append(src.read_text(errors="ignore"))
        provenance[element] = str(src)
    path.write_text("".join(parts))
    return provenance


def write_inputs(
    material: str,
    structure: Optional[str],
    calc: str,
    params: Dict[str, Any],
    workdir: Path,
    pp_dir: Optional[Path] = None,
    atoms: Optional[Atoms] = None,
) -> Dict[str, Any]:
    """Write the four VASP input files into workdir. Returns a metadata dict.

    ``atoms`` may be supplied directly (e.g. a relaxed structure recovered from
    an upstream CONTCAR); otherwise it is resolved from material/structure.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    if atoms is None:
        atoms, label = resolve_atoms(material, structure)
    else:
        label = material or (Path(str(structure)).stem if structure else "calc")
    atoms = apply_supercell(atoms, params.get("supercell"))
    profile = profile_from_atoms(atoms)
    is_metal = bool(profile.get("is_metal", True))

    # POTCAR first: its ZVAL sum gives the neutral electron count needed to
    # translate a requested tot_charge into an absolute NELECT.
    species = write_poscar(atoms, workdir / "POSCAR")
    potcar_sources = write_potcar(
        species, pp_dir if pp_dir is not None else default_pp_dir(), workdir / "POTCAR"
    )

    nelect: Optional[float] = None
    tot_charge = float(params.get("tot_charge", 0.0) or 0.0)
    if tot_charge != 0.0:
        neutral = _neutral_electrons(atoms, species, workdir / "POTCAR")
        # VASP NELECT convention: adding electrons (negative charge) raises it.
        nelect = neutral - tot_charge

    incar = build_incar(
        calc,
        {**params, "prefix": params.get("prefix", label)},
        is_metal,
        atoms=atoms,
        species=species,
        nelect=nelect,
    )
    (workdir / "INCAR").write_text(incar)
    (workdir / "KPOINTS").write_text(build_kpoints(params, calc=calc, atoms=atoms))

    return {
        "label": label,
        "calc": calc,
        "species": species,
        "is_metal": is_metal,
        "natoms": len(atoms),
        "potcar_sources": potcar_sources,
        "tot_charge": tot_charge,
        "nelect": nelect,
        "spin_polarized": bool(params.get("spin_polarized"))
        or str(params.get("magnetic_order") or "").lower().replace("-", "") in {"fm", "afm"},
        "magnetic_order": params.get("magnetic_order"),
        "supercell": params.get("supercell"),
    }


def _neutral_electrons(atoms: Atoms, species: List[str], potcar_path: Path) -> float:
    """Neutral valence-electron count = Σ per-species ZVAL × atom count.

    Reads ZVAL from each POTCAR block in POSCAR-species order; the count of
    atoms per species comes from the structure. Deterministic, no fabrication.
    """
    text = potcar_path.read_text(errors="ignore")
    zvals = [float(m) for m in re.findall(r"ZVAL\s*=\s*(-?\d+\.?\d*)", text)]
    if len(zvals) != len(species):
        raise VASPInputError(
            f"POTCAR ZVAL count ({len(zvals)}) does not match species "
            f"count ({len(species)}); cannot compute NELECT for a charged cell"
        )
    symbols = atoms.get_chemical_symbols()
    total = 0.0
    for element, zval in zip(species, zvals):
        total += zval * symbols.count(element)
    return total
