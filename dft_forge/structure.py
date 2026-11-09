"""User-provided structure import and validation module.

Supports:
  - CIF
  - POSCAR / CONTCAR / VASP
  - XYZ
  - QE input fragments (ATOMIC_SPECIES / ATOMIC_POSITIONS / CELL_PARAMETERS / K_POINTS)
  - Explicit lattice vectors + fractional/cartesian coordinates

All inputs are parsed with ASE / pymatgen / spglib, normalized to a primitive
ASE ``Atoms`` object, and validated for:
  - Element symbols and occupancies
  - Nearest-neighbor distances
  - Periodicity / dimensionality
  - Vacuum / unrealistic cell shapes

Unsafe or invalid inputs are rejected with a detailed ``ValidationReport``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    from ase import Atoms
    from ase.io import cif as ase_cif
    from ase.io import read as ase_read
    from ase.io import vasp as ase_vasp
    from ase.io import xyz as ase_xyz
    from ase.neighborlist import NeighborList
except ImportError:  # pragma: no cover
    raise ImportError("ASE is required for dft_forge.structure. Install it first.")

from dft_forge.constants import ANG_TO_BOHR, BOHR_TO_ANG, qe_label_to_element

try:
    from pymatgen.core import Lattice, Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
except ImportError:  # pragma: no cover
    raise ImportError("pymatgen is required for dft_forge.structure. Install it first.")

try:
    import spglib
except ImportError:  # pragma: no cover
    raise ImportError("spglib is required for dft_forge.structure. Install it first.")


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class ValidationReport:
    """Result of structure import and validation."""

    success: bool
    atoms: Optional[Atoms] = None
    format: Optional[str] = None
    formula: Optional[str] = None
    space_group: Optional[str] = None
    natoms: int = 0
    nspecies: int = 0
    species: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "success": self.success,
            "format": self.format,
            "formula": self.formula,
            "space_group": self.space_group,
            "natoms": self.natoms,
            "nspecies": self.nspecies,
            "species": self.species,
            "warnings": self.warnings,
            "errors": self.errors,
            "metadata": self.metadata,
        }
        if self.atoms is not None:
            d["lattice_constant_bohr"] = float(np.linalg.norm(self.atoms.cell[0]) * ANG_TO_BOHR)
            d["cell_volume_bohr3"] = float(abs(np.linalg.det(self.atoms.cell)) * (ANG_TO_BOHR ** 3))
        return d


# ── Helpers ────────────────────────────────────────────────────────────────────

_BOHR = ANG_TO_BOHR  # historical (misnamed) alias: Å → bohr factor
_ANGSTROM = BOHR_TO_ANG
_VALID_ELEMENT_RE = re.compile(r"^[A-Z][a-z]?(?:\d+)?$")
_ALLOWED_ELEMENTS = {
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
    "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
}


def _normalize_element(symbol: str) -> str:
    symbol = symbol.strip()
    if not _VALID_ELEMENT_RE.match(symbol):
        raise ValueError(f"Invalid element symbol: {symbol!r}")
    return symbol[0] + symbol[1:].lower()


def _is_periodic(atoms: Atoms) -> bool:
    return all(atoms.pbc)


def _min_nearest_neighbor_distance(atoms: Atoms) -> float:
    """Return the minimum nearest-neighbor distance in Angstrom."""
    if len(atoms) < 2:
        return float("inf")
    positions = atoms.get_positions()
    cutoff = max(5.0, float(np.max(np.linalg.norm(atoms.cell, axis=1))))
    nl = NeighborList(cutoffs=[cutoff / 2.0] * len(atoms), self_interaction=False, bothways=True)
    nl.update(atoms)
    min_dist = float("inf")
    for i in range(len(atoms)):
        indices, offsets = nl.get_neighbors(i)
        if len(indices) == 0:
            continue
        for j, offset in zip(indices.tolist(), offsets.tolist()):
            dist = np.linalg.norm(positions[i] - (positions[j] + offset @ atoms.cell))
            if dist > 1e-12:
                min_dist = min(min_dist, dist)
    return float(min_dist)


def _infer_format(source: Union[str, Path, Dict[str, Any]]) -> str:
    if isinstance(source, dict):
        return "explicit"
    path = Path(source) if not isinstance(source, Path) else source
    suffix = path.suffix.lower()
    name = path.name.lower()
    if suffix in {".cif"} or name.startswith("poscar") or name.startswith("contcar") or suffix in {".vasp"}:
        return "cif" if suffix == ".cif" else "poscar"
    if suffix == ".xyz" or name == "xyz":
        return "xyz"
    if suffix in {".in", ".pwi", ".qe"}:
        return "qe_input"
    return "unknown"


def _detect_format_from_text(text: str) -> str:
    upper = text.upper()
    if "_ATOMIC_POSITIONS" in upper or "ATOMIC_SPECIES" in upper:
        return "qe_input"
    if upper.startswith("POSCAR") or upper.startswith("ATOM") or "DIRECT" in upper or "CARTESIAN" in upper:
        return "poscar"
    if len(text.splitlines()) >= 2:
        first = text.splitlines()[0]
        toks = first.split()
        if len(toks) == 1:
            return "xyz"
    return "unknown"


# ── Parsers ────────────────────────────────────────────────────────────────────

def _parse_cif(source: Union[str, Path]) -> Atoms:
    result = ase_cif.read_cif(str(source))
    if isinstance(result, list):
        return result[0]
    return result


def _parse_poscar(source: Union[str, Path, str]) -> Atoms:
    if isinstance(source, str) and not Path(source).exists():
        # String is POSCAR content
        return ase_vasp.read_vasp(StringIO(source))
    return ase_vasp.read_vasp(str(source))


def _parse_xyz(source: Union[str, Path, str]) -> Atoms:
    if isinstance(source, str) and not Path(source).exists():
        text = source
    else:
        text = Path(source).read_text()
    gen = ase_xyz.read_xyz(StringIO(text), index=slice(None))
    frames = list(gen)
    if not frames:
        raise ValueError("XYZ source contains no frames")
    return frames[-1]


def _qe_element_from_label(label: str) -> str:
    """Backwards-compatible wrapper for the shared element-label helper."""
    return qe_label_to_element(label)


def _parse_qe_input(text: str) -> Atoms:
    """Parse a QE pw.x input fragment and return an ASE Atoms object.

    Honours QE unit conventions for both cards:
    - CELL_PARAMETERS (bohr|angstrom|alat[=X]); default is bohr
    - ATOMIC_POSITIONS (alat|bohr|angstrom|crystal|crystal_sg|cartesian);
      default is alat
    alat is taken from celldm(1) (bohr) or A (Å) in &SYSTEM.
    """
    cell_params_lines: List[str] = []
    atomic_positions_lines: List[str] = []
    atomic_species: List[Tuple[str, float, str]] = []
    cell_vectors_ang: Optional[np.ndarray] = None
    symbols: List[str] = []
    cart_ang: Optional[np.ndarray] = None
    frac: Optional[np.ndarray] = None

    current_section: Optional[str] = None
    cell_unit = "bohr"
    pos_unit = "alat"

    # alat (in bohr) from &SYSTEM: celldm(1) wins over A (given in Å).
    alat_bohr: Optional[float] = None

    def _option(header: str) -> str:
        if "(" not in header:
            return ""
        token = header.split("(", 1)[1].split(")", 1)[0].strip().lower()
        return token

    for raw_line in text.splitlines():
        line = raw_line.split("!", 1)[0].split("#", 1)[0]
        stripped = line.strip()
        upper = stripped.upper()
        if stripped.startswith("&"):
            current_section = "namelist"
            continue
        if current_section == "namelist":
            if stripped == "/":
                current_section = None
                continue
            m_celldm = re.search(r"celldm\s*\(\s*1\s*\)\s*=\s*([-+0-9.eEdD]+)", stripped, re.I)
            if m_celldm:
                alat_bohr = float(m_celldm.group(1).replace("d", "e").replace("D", "E"))
            m_a = re.search(r"(?<![A-Za-z0-9_])A\s*=\s*([-+0-9.eEdD]+)", stripped)
            if m_a and alat_bohr is None:
                alat_bohr = float(m_a.group(1).replace("d", "e").replace("D", "E")) / BOHR_TO_ANG
            continue
        if upper.startswith("CELL_PARAMETERS"):
            current_section = "cell"
            cell_unit = _option(upper) or "bohr"
            continue
        if upper.startswith("ATOMIC_POSITIONS"):
            current_section = "positions"
            pos_unit = _option(upper) or "alat"
            continue
        if upper.startswith("ATOMIC_SPECIES"):
            current_section = "species"
            continue
        if stripped == "":
            continue

        if current_section == "cell":
            cell_params_lines.append(stripped)
        elif current_section == "positions":
            atomic_positions_lines.append(stripped)
        elif current_section == "species":
            parts = stripped.split()
            if len(parts) >= 3:
                atomic_species.append((parts[0], float(parts[1]), parts[2]))

    def _alat_bohr_required(where: str) -> float:
        if alat_bohr is None:
            raise ValueError(
                f"{where} uses 'alat' units but neither celldm(1) nor A is given in &SYSTEM"
            )
        return alat_bohr

    if cell_params_lines:
        arr = np.array([list(map(float, ln.split()[:3])) for ln in cell_params_lines[:3]])
        unit = cell_unit.replace(" ", "")
        if unit in ("bohr", ""):
            cell_vectors_ang = arr * BOHR_TO_ANG
        elif unit == "angstrom":
            cell_vectors_ang = arr
        elif unit == "alat":
            cell_vectors_ang = arr * _alat_bohr_required("CELL_PARAMETERS") * BOHR_TO_ANG
        elif unit.startswith("alat="):
            # QE 7.5 treats CELL_PARAMETERS {alat=X} exactly like {alat}:
            # vectors are in units of celldm(1) from &SYSTEM (the =X value
            # is ignored); verified with real pw.x (alat=1.0 and alat=2.0
            # produce the identical unit-cell volume).
            cell_vectors_ang = arr * _alat_bohr_required("CELL_PARAMETERS") * BOHR_TO_ANG
        else:
            raise ValueError(f"unsupported CELL_PARAMETERS unit: {cell_unit!r}")

    if atomic_positions_lines:
        coords: List[List[float]] = []
        for line in atomic_positions_lines:
            parts = line.split()
            if len(parts) >= 4:
                symbols.append(parts[0])
                coords.append([float(x) for x in parts[1:4]])
        raw = np.array(coords)
        unit = pos_unit.replace(" ", "")
        if unit in ("crystal", "crystal_sg"):
            frac = raw
        elif unit in ("bohr",):
            cart_ang = raw * BOHR_TO_ANG
        elif unit == "angstrom":
            cart_ang = raw
        elif unit in ("alat", "cartesian", ""):
            cart_ang = raw * _alat_bohr_required("ATOMIC_POSITIONS") * BOHR_TO_ANG
        else:
            raise ValueError(f"unsupported ATOMIC_POSITIONS unit: {pos_unit!r}")

    if not symbols:
        raise ValueError("QE input fragment has no ATOMIC_POSITIONS")
    if cell_vectors_ang is None:
        raise ValueError("QE input fragment missing CELL_PARAMETERS")

    # QE species labels may carry suffixes (FeA/Fe1 for AFM species); the
    # Atoms object needs real element symbols.
    normalized = [_qe_element_from_label(sym) for sym in symbols]

    if cart_ang is not None:
        atoms = Atoms(normalized, positions=cart_ang, cell=cell_vectors_ang, pbc=True)
    elif frac is not None:
        atoms = Atoms(
            normalized,
            scaled_positions=frac,
            cell=cell_vectors_ang,
            pbc=True,
        )
    else:
        atoms = Atoms(normalized, cell=cell_vectors_ang, pbc=True)
    return atoms


def _build_from_explicit(
    lattice: Sequence[Sequence[float]],
    positions: Sequence[Sequence[float]],
    symbols: Sequence[str],
    *,
    fractional: bool = False,
    pbc: bool = True,
) -> Atoms:
    lattice_arr = np.array(lattice, dtype=float)
    if lattice_arr.shape != (3, 3):
        raise ValueError("lattice must be a 3x3 array")
    positions_arr = np.array(positions, dtype=float)
    if positions_arr.ndim == 1:
        positions_arr = positions_arr.reshape(1, -1)
    if positions_arr.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    if len(symbols) != len(positions_arr):
        raise ValueError("symbols and positions length mismatch")
    if fractional:
        positions_arr = positions_arr @ lattice_arr
    return Atoms(
        [_normalize_element(sym) for sym in symbols],
        positions=positions_arr,
        cell=lattice_arr,
        pbc=[pbc] * 3 if isinstance(pbc, bool) else list(pbc),
    )


# ── Normalization ──────────────────────────────────────────────────────────────

def _normalize_to_primitive(atoms: Atoms) -> Atoms:
    """Try to map to primitive cell via pymatgen SpacegroupAnalyzer + spglib."""
    try:
        structure = Structure(
            Lattice(atoms.cell),
            atoms.get_chemical_symbols(),
            atoms.get_positions(),
            coords_are_cartesian=True,
        )
        sga = SpacegroupAnalyzer(structure, symprec=1e-3)
        primitive = sga.get_primitive_standard_structure()
        return Atoms(
            [site.species_string for site in primitive],
            positions=primitive.cart_coords,
            cell=primitive.lattice.matrix,
            pbc=True,
        )
    except Exception:
        # Fallback: keep original and just scale to standard orientation.
        return atoms.copy()


def _sort_symbols(atoms: Atoms) -> Atoms:
    """Sort atoms by chemical symbol to give deterministic ordering."""
    order = sorted(range(len(atoms)), key=lambda i: atoms.symbols[i])
    return atoms[order]


# ── Validation ─────────────────────────────────────────────────────────────────

def _validate_elements(atoms: Atoms) -> List[str]:
    errors: List[str] = []
    for sym in atoms.get_chemical_symbols():
        if sym not in _ALLOWED_ELEMENTS:
            errors.append(f"Unsupported element symbol: {sym}")
    return errors


def _validate_occupancies(atoms: Atoms) -> List[str]:
    errors: List[str] = []
    try:
        structure = Structure(
            Lattice(atoms.cell),
            atoms.get_chemical_symbols(),
            atoms.get_positions(),
            coords_are_cartesian=True,
        )
        for site in structure:
            occ = getattr(site.species, "num_atoms", 1.0)
            if occ < 0.99 or occ > 1.01:
                errors.append(f"Site {site.species_string} has occupancy {occ}")
    except Exception as exc:
        errors.append(f"Occupancy check failed: {exc}")
    return errors


def _validate_nearest_neighbors(atoms: Atoms, min_dist_angstrom: float = 0.8) -> List[str]:
    errors: List[str] = []
    if len(atoms) < 2:
        return errors
    min_dist = _min_nearest_neighbor_distance(atoms)
    if min_dist < min_dist_angstrom:
        errors.append(
            f"Nearest-neighbor distance {min_dist:.3f} Å is below threshold {min_dist_angstrom} Å"
        )
    return errors


def _validate_periodicity(atoms: Atoms) -> List[str]:
    errors: List[str] = []
    if not _is_periodic(atoms):
        errors.append("Structure is not periodic in all three dimensions (pbc must be True for x, y, z)")
    return errors


def _validate_vacuum_and_cell(atoms: Atoms) -> List[str]:
    warnings: List[str] = []
    if not _is_periodic(atoms):
        return warnings
    a = np.linalg.norm(atoms.cell[0])
    b = np.linalg.norm(atoms.cell[1])
    c = np.linalg.norm(atoms.cell[2])
    max_axis = max(a, b, c)
    min_axis = min(a, b, c)
    if max_axis > 100.0:
        warnings.append(f"Very large cell axis ({max_axis:.2f} Å) may indicate excessive vacuum")
    if min_axis < 1.5:
        warnings.append(f"Very small cell axis ({min_axis:.2f} Å) may indicate an incomplete cell")
    return warnings


# ── Main importer ─────────────────────────────────────────────────────────────

class StructureImporter:
    """Parse and validate user-provided structures into a normalized ASE Atoms object.

    Accepted ``source`` types:
      - ``str`` / ``Path`` pointing to CIF, POSCAR, CONTCAR, XYZ, or QE input files
      - ``str`` containing POSCAR/QE/XYZ content directly
      - ``dict`` with explicit structure data::

            {
                "lattice": [[...], [...], [...]],
                "positions": [[x1,y1,z1], ...],
                "symbols": ["Si", "Si"],
                "fractional": false,
            }
    """

    def __init__(
        self,
        min_nearest_neighbor_angstrom: float = 0.8,
        standardize: bool = True,
    ) -> None:
        self.min_nearest_neighbor_angstrom = min_nearest_neighbor_angstrom
        self.standardize = standardize

    def import_structure(
        self,
        source: Union[str, Path, Dict[str, Any]],
        *,
        format: Optional[str] = None,
        fractional: bool = False,
    ) -> ValidationReport:
        """Import a structure from *source* and validate it.

        Returns a ``ValidationReport``.  ``report.success`` indicates whether
        the structure passed validation; ``report.atoms`` is ``None`` on failure.
        """
        report = ValidationReport(success=False)

        # ── Dispatch parsing ────────────────────────────────────────────────────
        fmt = format or _infer_format(source)
        report.format = fmt

        try:
            if fmt == "cif":
                atoms = _parse_cif(Path(source) if not isinstance(source, Path) else source)
            elif fmt == "poscar":
                atoms = _parse_poscar(source)
            elif fmt == "xyz":
                atoms = _parse_xyz(source)
            elif fmt == "qe_input":
                if isinstance(source, (str, Path)) and Path(source).exists():
                    text = Path(source).read_text()
                else:
                    text = str(source)
                atoms = _parse_qe_input(text)
            elif fmt == "explicit":
                if not isinstance(source, dict):
                    raise TypeError("explicit format requires a dict with lattice/positions/symbols")
                atoms = _build_from_explicit(
                    source["lattice"],
                    source["positions"],
                    source["symbols"],
                    fractional=source.get("fractional", fractional),
                    pbc=source.get("pbc", True),
                )
            else:
                # Fallback: try text-based auto-detection.
                if isinstance(source, str) and not Path(source).exists():
                    text = source
                    detected = _detect_format_from_text(text)
                    if detected == "xyz":
                        atoms = _parse_xyz(text)
                    elif detected == "poscar":
                        atoms = _parse_poscar(text)
                    elif detected == "qe_input":
                        atoms = _parse_qe_input(text)
                    else:
                        report.errors.append(f"Unsupported or unknown structure format: {fmt}")
                        return report
                else:
                    report.errors.append(f"Unsupported or unknown structure format: {fmt}")
                    return report
        except Exception as exc:
            report.errors.append(f"Parsing failed: {exc}")
            return report

        # Basic sanity check.
        if not isinstance(atoms, Atoms) or len(atoms) == 0:
            report.errors.append("Parser returned an empty or invalid structure")
            return report

        # QE is always 3D-periodic: a 2D slab is modelled as a 3D cell with
        # vacuum (and k=1 along it). When a full-rank 3D cell is present but
        # the reader flagged pbc=False on some axes (molecules/slabs from
        # XYZ/explicit dicts), adopt the QE convention with a warning instead
        # of rejecting; genuinely cell-less inputs still fail below.
        if atoms.cell is not None and np.all(np.asarray(atoms.cell).shape == (3, 3)) \
                and abs(float(np.linalg.det(atoms.cell))) > 1e-8 and not all(atoms.pbc):
            atoms.pbc = True
            report.warnings.append(
                "输入结构存在非周期方向；按 QE 约定以 3D 周期胞处理（2D 体系请保留真空层，"
                "非周期方向 k 点取 1）"
            )

        # Physical sanity BEFORE normalization: two atoms on top of each
        # other make spglib/pymatgen fail with a misleading "Invalid cell
        # vectors" — report the real problem instead.
        nn_errors = _validate_nearest_neighbors(atoms, self.min_nearest_neighbor_angstrom)
        if nn_errors:
            report.errors.extend(nn_errors)
            return report

        # ── Normalize ───────────────────────────────────────────────────────────
        try:
            if self.standardize:
                atoms = _normalize_to_primitive(atoms)
            atoms = _sort_symbols(atoms)
            # Ensure Angstrom positions and cell.
            if atoms.cell is None or np.linalg.det(atoms.cell) <= 0:
                raise ValueError("Invalid cell vectors")
        except Exception as exc:
            report.errors.append(f"Normalization failed: {exc}")
            return report

        report.atoms = atoms
        report.natoms = len(atoms)
        report.species = sorted({sym for sym in atoms.get_chemical_symbols()})
        report.nspecies = len(report.species)
        report.formula = atoms.get_chemical_formula()

        # Space group (best effort).
        try:
            structure = Structure(
                Lattice(atoms.cell),
                atoms.get_chemical_symbols(),
                atoms.get_positions(),
                coords_are_cartesian=True,
            )
            sga = SpacegroupAnalyzer(structure, symprec=1e-3)
            report.space_group = sga.get_space_group_symbol()
            report.metadata["crystal_system"] = sga.get_crystal_system()
            report.metadata["spglib_dataset"] = {
                "number": int(getattr(sga, "_sg_number", 1)),
            }
        except Exception as exc:
            report.warnings.append(f"Space group analysis failed: {exc}")

        # ── Validations ─────────────────────────────────────────────────────────
        report.errors.extend(_validate_periodicity(atoms))
        report.errors.extend(_validate_elements(atoms))
        report.errors.extend(_validate_occupancies(atoms))
        report.errors.extend(_validate_nearest_neighbors(atoms, self.min_nearest_neighbor_angstrom))
        report.warnings.extend(_validate_vacuum_and_cell(atoms))

        report.success = len(report.errors) == 0
        return report
