"""QE input compiler: deterministic generation of QE input files.

Uses ASE/pymatgen/spglib/seekpath for structure handling, k-points, and paths.
The LLM never writes shell commands or full QE input.
"""

from __future__ import annotations

import hashlib
import math
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from ase import Atoms
from ase.calculators.espresso import Espresso
from ase.io import write as ase_write
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.kpath import KPathSeek


# ── Material database ─────────────────────────────────────────────────────────

MATERIAL_DB: Dict[str, Dict[str, Any]] = {
    "Si": {
        "formula": "Si",
        "structure_type": "diamond",
        "space_group": "Fd-3m",
        "ibrav": 2,
        "lattice_constant_angstrom": 5.43,
        "natoms_primitive": 2,
        "nspecies": 1,
        "species": ["Si"],
        "masses": [28.086],
        "pseudos": {"Si": "Si_r.upf"},
        "occupations": "smearing",
        "smearing": "mp",
        "degauss": 0.005,
        "ecutwfc_default": 25.0,
        "ecutrho_default": 200.0,
        "kpoints_default": (4, 4, 4, 1, 1, 1),
        "is_metal": False,
    },
    "Al": {
        "formula": "Al",
        "structure_type": "fcc",
        "space_group": "Fm-3m",
        "ibrav": 2,
        "lattice_constant_angstrom": 4.05,
        "natoms_primitive": 1,
        "nspecies": 1,
        "species": ["Al"],
        "masses": [26.982],
        "pseudos": {"Al": "Al.pbe-n-rrkjus_psl.1.0.2.UPF"},
        "occupations": "smearing",
        "smearing": "mp",
        "degauss": 0.02,
        "ecutwfc_default": 20.0,
        "ecutrho_default": 160.0,
        "kpoints_default": (8, 8, 8, 1, 1, 1),
        "is_metal": True,
    },
    "MgO": {
        "formula": "MgO",
        "structure_type": "nacl",
        "space_group": "Fm-3m",
        "ibrav": 2,
        "lattice_constant_angstrom": 4.21,
        "natoms_primitive": 2,
        "nspecies": 2,
        "species": ["Mg", "O"],
        "masses": [24.305, 15.999],
        "pseudos": {"Mg": "Mg.pz-n-vbc.UPF", "O": "O-q6.gth"},
        "occupations": "smearing",
        "smearing": "mp",
        "degauss": 0.01,
        "ecutwfc_default": 40.0,
        "ecutrho_default": 320.0,
        "kpoints_default": (4, 4, 4, 1, 1, 1),
        "is_metal": False,
        "input_dft": "pz",
    },
}

from dft_forge.catalog.library import LIBRARY_MATERIALS  # noqa: E402

MATERIAL_DB.update(LIBRARY_MATERIALS)


# ── Structure builder ─────────────────────────────────────────────────────────

def get_lattice_constant_angstrom(material: str) -> float:
    """Scale length for a material (built-in or library), in Angstrom."""
    db = MATERIAL_DB[material]
    return db["lattice_constant_angstrom"] if "lattice_constant_angstrom" in db else db["cellpar"][0]


def build_atoms(material: str) -> Atoms:
    """Build ASE Atoms object for a known material.

    Returns primitive cell structures suitable for QE ibrav=0 input.
    For FCC-based materials (Si, Al, MgO), the primitive cell vectors are:
        a1 = [0, a/2, a/2], a2 = [a/2, 0, a/2], a3 = [a/2, a/2, 0]
    where a is the conventional cubic lattice constant.
    """
    if material not in MATERIAL_DB:
        raise ValueError(f"Unknown material: {material}. Known: {list(MATERIAL_DB.keys())}")

    db = MATERIAL_DB[material]
    a_angstrom = get_lattice_constant_angstrom(material)

    if "cellpar" in db and "sites" in db:
        # Library material: exact MP cell + fractional sites (Å, ASE convention)
        from ase.cell import Cell

        a, b, c, alpha, beta, gamma = db["cellpar"]
        cell = Cell.fromcellpar([a, b, c, alpha, beta, gamma])
        symbols = [s[0] for s in db["sites"]]
        positions = [s[1] for s in db["sites"]]
        atoms = Atoms(symbols=symbols, scaled_positions=positions, cell=cell, pbc=True)
        return atoms

    if material == "Si":
        # Diamond structure: primitive 2-atom cell
        # Lattice vectors for FCC primitive cell
        cell = [
            [0, a_angstrom/2, a_angstrom/2],
            [a_angstrom/2, 0, a_angstrom/2],
            [a_angstrom/2, a_angstrom/2, 0]
        ]
        # Positions in crystal coordinates of primitive cell
        pos = [
            [0, 0, 0],
            [0.25, 0.25, 0.25]
        ]
        atoms = Atoms("Si2", scaled_positions=pos, cell=cell, pbc=True)
        return atoms

    elif material == "Al":
        # FCC primitive cell: 1 atom
        cell = [
            [0, a_angstrom/2, a_angstrom/2],
            [a_angstrom/2, 0, a_angstrom/2],
            [a_angstrom/2, a_angstrom/2, 0]
        ]
        pos = [[0, 0, 0]]
        atoms = Atoms("Al", scaled_positions=pos, cell=cell, pbc=True)
        return atoms

    elif material == "MgO":
        # NaCl structure: primitive 2-atom cell (FCC lattice with 2-atom basis)
        cell = [
            [0, a_angstrom/2, a_angstrom/2],
            [a_angstrom/2, 0, a_angstrom/2],
            [a_angstrom/2, a_angstrom/2, 0]
        ]
        # Mg at (0,0,0), O at (0.5,0.5,0.5) in primitive coordinates
        pos = [
            [0, 0, 0],
            [0.5, 0.5, 0.5]
        ]
        atoms = Atoms("MgO", scaled_positions=pos, cell=cell, pbc=True)
        return atoms

    else:
        raise ValueError(f"Material '{material}' structure not implemented")


# ── Generic structures (any CIF/POSCAR via ASE) ───────────────────────────────

BOHR_PER_ANG = 1.8897261246
DEFAULT_PSEUDO_DIR = Path(__file__).resolve().parents[2] / "assets" / "pseudos"


def read_structure(path) -> Atoms:
    """Read a structure file (CIF/POSCAR/...) into ASE Atoms.

    A vacuum axis is detected from the atomic coordinate span: if the
    largest circular gap between fractional coordinates along an axis
    exceeds half the cell (and the axis is ≥ 8 Å), the slab occupies less
    than half the cell there — periodicity is turned off so k-meshes
    collapse and band paths stay in-plane. A length-ratio test would miss
    large in-plane supercells (e.g. 4×4 graphene with 15 Å vacuum).
    """
    from ase.io import read as ase_read

    p = Path(str(path))
    if not p.exists():
        raise FileNotFoundError(f"structure file not found: {path}")
    atoms = ase_read(str(p))
    lengths = atoms.cell.cellpar()[:3]
    frac = atoms.get_scaled_positions(wrap=True)
    pbc = [True, True, True]
    for i in range(3):
        if lengths[i] < 8.0 or len(atoms) < 2:
            continue
        f = np.sort(frac[:, i])
        gaps = np.diff(np.concatenate([f, [f[0] + 1.0]]))
        if float(gaps.max()) > 0.5:
            pbc[i] = False
    atoms.pbc = pbc
    return atoms


def profile_from_atoms(atoms: Atoms, pseudo_dir: Optional[Path] = None) -> Dict[str, Any]:
    """MATERIAL_DB-shaped profile for an arbitrary Atoms object.

    Species map onto the GBRV ``{Element}.upf`` library; ibrav=0 with the
    cell emitted explicitly. The k-mesh adapts to cell size; non-periodic
    axes (slab vacuum, atoms.pbc=False) always get k=1, while long but
    genuinely periodic axes (e.g. 2H TMD c) are still sampled.
    """
    from ase.data import atomic_masses, chemical_symbols

    pseudo_dir = Path(pseudo_dir) if pseudo_dir else DEFAULT_PSEUDO_DIR
    species = sorted(set(atoms.get_chemical_symbols()))
    pseudos: Dict[str, str] = {}
    for el in species:
        found = next((c for c in (f"{el}.upf", f"{el}.UPF") if (pseudo_dir / c).exists()), None)
        if found is None:
            raise FileNotFoundError(f"no GBRV pseudopotential for element '{el}' in {pseudo_dir}")
        pseudos[el] = found
    masses = [float(atomic_masses[chemical_symbols.index(el)]) for el in species]
    lengths = atoms.cell.cellpar()[:3]
    pbc = list(atoms.pbc) if len(atoms.pbc) == 3 else [True, True, True]
    k = [
        1 if not periodic else max(1, min(8, round(24.0 / L)))
        for L, periodic in zip(lengths, pbc)
    ]
    return {
        "ibrav": 0,
        "nspecies": len(species),
        "species": species,
        "masses": masses,
        "pseudos": pseudos,
        "occupations": "smearing",
        "smearing": "mv",
        "degauss": 0.01,
        "ecutwfc_default": 40.0,
        "ecutrho_default": 320.0,
        "kpoints_default": (k[0], k[1], k[2], 1, 1, 1),
        "is_metal": True,
        "_alat_bohr": float(max(lengths)) * BOHR_PER_ANG,
    }


# ── Formula → prototype structure ─────────────────────────────────────────────

# experimental cubic lattice constants (Å) for common ABO3 perovskites
PEROVSKITE_A: Dict[str, float] = {
    "CaTiO3": 3.827, "SrTiO3": 3.905, "BaTiO3": 4.006, "KTaO3": 3.983,
    "KNbO3": 4.021, "PbTiO3": 3.969, "SrZrO3": 4.101, "BaZrO3": 4.193,
    "LaAlO3": 3.791, "LaFeO3": 3.930, "SrCoO3": 3.829, "BaSnO3": 4.116,
    "SrSnO3": 4.034, "CaZrO3": 4.020, "PbZrO3": 4.155, "LaCrO3": 3.887,
    "YAlO3": 3.792, "NaTaO3": 3.929, "AgNbO3": 3.953, "BiFeO3": 3.965,
}
_III_V = {"B", "Al", "Ga", "In", "Tl"}
_V_VI = {"N", "P", "As", "Sb", "Bi"}
_DIAMOND_ELS = {"C", "Si", "Ge", "Sn"}
_BCC_ELS = {"Fe", "Cr", "W", "Mo", "V", "Nb", "Ta", "K", "Na", "Li"}
_HCP_ELS = {"Mg", "Zn", "Ti", "Zr", "Co", "Be", "Ru", "Os", "Sc", "Y", "Hf", "Re"}

# 2H-layered TMD bulks: formula -> (metal, chalcogen, a [Å], M-X bond [Å],
# interlayer gap [Å]). Built instead of the fluorite prototype — fluorite
# MX2 is not a real TMD structure.
TMD_BULKS: Dict[str, Tuple[str, str, float, float, float]] = {
    "MoS2": ("Mo", "S", 3.16, 2.41, 3.0),
    "WS2": ("W", "S", 3.15, 2.41, 3.1),
    "MoSe2": ("Mo", "Se", 3.29, 2.38, 3.3),
    "WSe2": ("W", "Se", 3.28, 2.41, 3.3),
    "MoTe2": ("Mo", "Te", 3.52, 2.73, 3.5),
    "WTe2": ("W", "Te", 3.51, 2.72, 3.5),
}


def _tmd_bulk(formula_key: str) -> Tuple[Atoms, str]:
    """2H-phase TMD bulk: two AB-stacked X-M-X layers per hexagonal cell."""
    metal, chalc, a, bond, gap = TMD_BULKS[formula_key]
    inplane = a / math.sqrt(3.0)
    dz = math.sqrt(bond**2 - inplane**2)
    c = 4.0 * dz + 2.0 * gap
    cell = [[a, 0, 0], [-a / 2, a * math.sqrt(3) / 2, 0], [0, 0, c]]
    z = dz / c
    pos = [
        (1 / 3, 2 / 3, 0.25), (1 / 3, 2 / 3, 0.25 + z), (1 / 3, 2 / 3, 0.25 - z),
        (2 / 3, 1 / 3, 0.75), (2 / 3, 1 / 3, 0.75 + z), (2 / 3, 1 / 3, 0.75 - z),
    ]
    syms = [metal, chalc, chalc] * 2
    return Atoms(syms, scaled_positions=pos, cell=cell, pbc=True), "tmd-2h"


def _cov_radius(el: str) -> float:
    from ase.data import chemical_symbols, covalent_radii

    return float(covalent_radii[chemical_symbols.index(el)])


def formula_atoms(formula: str) -> Tuple[Atoms, str]:
    """Build a starting structure for any chemical formula via prototype matching.

    Prototypes: ABO3 perovskite, AB zincblende/rocksalt, AB2/A2B fluorite,
    elemental diamond/fcc/bcc/hcp. Lattice constants come from experiment
    where tabulated, else from covalent-radius estimates — vc-relax then
    refines the cell, so these only need to be physically sensible.

    Returns (Atoms, prototype_name).
    """
    from pymatgen.core import Composition

    comp = Composition(str(formula)).as_dict()
    els = sorted(comp, key=lambda e: -comp[e])
    nums = [int(round(comp[e])) for e in els]
    if any(n <= 0 for n in nums):
        raise ValueError(f"cannot parse formula: {formula}")

    # elemental
    if len(els) == 1:
        el = els[0]
        if el in _DIAMOND_ELS:
            a = 8.0 / math.sqrt(3.0) * _cov_radius(el)
            cell = np.eye(3) * a
            pos = [[0, 0, 0], [0.25, 0.25, 0.25], [0.5, 0.5, 0], [0.75, 0.75, 0],
                   [0.5, 0, 0.5], [0.75, 0.25, 0.75], [0, 0.5, 0.5], [0.25, 0.75, 0.75]]
            return Atoms(el * 8, scaled_positions=pos, cell=cell, pbc=True), "diamond"
        if el in _BCC_ELS:
            a = 4.0 / math.sqrt(3.0) * _cov_radius(el)
            cell = np.eye(3) * a
            return Atoms(el * 2, scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]], cell=cell, pbc=True), "bcc"
        if el in _HCP_ELS:
            r = _cov_radius(el)
            a = 2.0 * r
            c = 1.633 * a
            cell = [[a, 0, 0], [-a / 2, a * math.sqrt(3) / 2, 0], [0, 0, c]]
            pos = [[1/3, 2/3, 0.25], [2/3, 1/3, 0.75]]
            return Atoms(el * 2, scaled_positions=pos, cell=cell, pbc=True), "hcp"
        a = 2.0 * math.sqrt(2.0) * _cov_radius(el)  # fcc default for metals
        cell = np.eye(3) * a
        pos = [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
        return Atoms(el * 4, scaled_positions=pos, cell=cell, pbc=True), "fcc"

    # ABO3 perovskite: one A-site cation, one B-site, three O
    if len(els) == 3 and "O" in els and sorted(nums) == [1, 1, 3] and comp.get("O") == 3:
        a_site = _perovskite_a_site(els, comp)
        b_site = next(e for e in els if e != "O" and e != a_site)
        a = PEROVSKITE_A.get(str(formula).replace(" ", ""), 1.414 * (_cov_radius(a_site) + _cov_radius("O")) * 1.02)
        cell = np.eye(3) * a
        pos = [[0.5, 0.5, 0.5], [0, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]]
        syms = [a_site, b_site, "O", "O", "O"]
        return Atoms(syms, scaled_positions=pos, cell=cell, pbc=True), "perovskite"

    # 2H-layered TMD bulk (MoS2 family) — before the generic AB2 fluorite
    # branch, which would otherwise emit a physically wrong structure
    tmd_key = next((k for k in TMD_BULKS if k.lower() == str(formula).replace(" ", "").lower()), None)
    if tmd_key is not None:
        return _tmd_bulk(tmd_key)

    # AB binary
    if len(els) == 2 and nums[0] == 1 and nums[1] == 1:
        e1, e2 = els
        if (e1 in _III_V and e2 in _V_VI) or (e2 in _III_V and e1 in _V_VI):
            a = 4.0 / math.sqrt(3.0) * (_cov_radius(e1) + _cov_radius(e2))
            cell = np.eye(3) * a
            pos = [[0, 0, 0], [0.25, 0.25, 0.25]]
            return Atoms(e1 + e2, scaled_positions=pos, cell=cell, pbc=True), "zincblende"
        a = 2.0 * (_cov_radius(e1) + _cov_radius(e2)) * 1.15  # rocksalt
        cell = np.eye(3) * a
        pos = [[0, 0, 0], [0.5, 0.5, 0.5]]
        return Atoms(e1 + e2, scaled_positions=pos, cell=cell, pbc=True), "rocksalt"

    # AB2 / A2B fluorite-type
    if len(els) == 2 and sorted(nums) == [1, 2]:
        a_el = els[nums.index(1)]
        b_el = els[nums.index(2)]
        a = 4.0 / math.sqrt(3.0) * (_cov_radius(a_el) + _cov_radius(b_el)) * 1.02
        cell = np.eye(3) * a
        pos = [[0, 0, 0], [0.75, 0.75, 0.75], [0.25, 0.25, 0.25],
               [0.75, 0.25, 0.25], [0.25, 0.75, 0.25], [0.25, 0.25, 0.75],
               [0.25, 0.75, 0.75], [0.75, 0.25, 0.75], [0.75, 0.75, 0.25]]
        syms = [a_el] + [b_el] * 8
        return Atoms(syms, scaled_positions=pos, cell=cell, pbc=True), "fluorite"

    raise ValueError(
        f"no structure prototype for '{formula}' (supports: elemental, AB, ABO3, "
        "AB2, TMD MoS2-family). Provide a CIF file or use structure.build2d "
        "for 2D monolayers."
    )


def _perovskite_a_site(els, comp) -> str:
    """A-site = the larger-cation (non-O) element of an ABO3 formula."""
    cations = [e for e in els if e != "O"]
    return max(cations, key=_cov_radius)


# ── K-point generation ────────────────────────────────────────────────────────

def get_kpoints(material: str, kpoints_override: Optional[Tuple] = None) -> Tuple:
    """Get k-point mesh for a material.
    
    Returns (nx, ny, nz, sx, sy, sz) tuple for QE K_POINTS automatic.
    """
    if kpoints_override:
        return kpoints_override
    
    db = MATERIAL_DB.get(material, {})
    return db.get("kpoints_default", (4, 4, 4, 1, 1, 1))


def get_high_symmetry_path(material: str) -> Tuple[np.ndarray, List[str]]:
    """Get high-symmetry k-path for band structure calculations (T2).

    Returns (kpoints_array, labels) where kpoints_array is (N, 3) in
    fractional coordinates of the primitive cell.
    """
    db = MATERIAL_DB.get(material)
    if not db:
        raise ValueError(f"Unknown material: {material}")

    a = get_lattice_constant_angstrom(material)
    sg = db["space_group"]

    # Build primitive structure
    struct = build_atoms(material)

    # Use seekpath for high-symmetry path
    kpath = KPathSeek(struct).kpath

    kpoints = kpath["kpoints"]
    labels = kpath["path"][0]  # First path segment

    return kpoints, labels


def get_bandpath_segments(material: str) -> List[List[Tuple[str, tuple]]]:
    """High-symmetry k-path segments for a MATERIAL_DB material."""
    return bandpath_segments_from_atoms(build_atoms(material))


def bandpath_segments_from_atoms(atoms: Atoms) -> List[List[Tuple[str, tuple]]]:
    """High-symmetry k-path as a list of continuous segments for any Atoms.

    Each segment is a list of (label, (kx, ky, kz)) pairs in fractional
    coordinates of the cell. Uses ASE's Bravais-lattice aware bandpath
    (version-robust, no seekpath dependency). ``atoms.pbc`` is forwarded so
    slab/2D cells (vacuum axis non-periodic) get an in-plane path (Γ-M-K-Γ)
    instead of a 3D one with A/H/L.
    """
    pbc = tuple(bool(b) for b in atoms.pbc) if len(atoms.pbc) == 3 else None
    bp = atoms.cell.bandpath(pbc=pbc)
    special = bp.special_points
    segments: List[List[Tuple[str, tuple]]] = []
    for seg in str(bp.path).split(","):
        if not seg:
            continue
        chain: List[Tuple[str, tuple]] = []
        for label in seg:
            if label in ("|", "$"):
                continue
            if label not in special:
                raise ValueError(f"unknown special point '{label}' for cell")
            k = special[label]
            chain.append((label, (float(k[0]), float(k[1]), float(k[2]))))
        if len(chain) >= 2:
            segments.append(chain)
    if not segments:
        raise ValueError("could not build k-path for cell")
    return segments


_LABEL_MAP = {"G": "GAMMA"}


def get_bandpath_kpoints(material: str, npoints: int = 100, atoms: Optional[Atoms] = None) -> np.ndarray:
    """Explicit k-point list along the high-symmetry path, (N, 3) fractional.

    Handles path discontinuities correctly (jumps are preserved, not
    interpolated across), unlike a naive crystal_b segment rendering.
    """
    a = atoms if atoms is not None else build_atoms(material)
    pbc = tuple(bool(b) for b in a.pbc) if len(a.pbc) == 3 else None
    return np.asarray(a.cell.bandpath(npoints=max(10, npoints), pbc=pbc).kpts)


# ── QE input generator ────────────────────────────────────────────────────────

class QECompiler:
    """Deterministic QE input file compiler.
    
    Takes a TaskSpec and material profile, produces a valid .in file.
    No LLM involved in input generation.
    """

    def __init__(self, pseudo_dir: Path):
        self.pseudo_dir = Path(pseudo_dir)
        self._validate_pseudo_dir()
        self._z_valence_cache: Dict[str, float] = {}

    def _validate_pseudo_dir(self) -> None:
        if not self.pseudo_dir.exists():
            raise FileNotFoundError(f"pseudo_dir not found: {self.pseudo_dir}")

    def _check_pseudos(self, pseudos: Dict[str, str]) -> List[str]:
        """Check that all required pseudopotentials exist."""
        missing = []
        for element, filename in pseudos.items():
            if not (self.pseudo_dir / filename).exists():
                missing.append(f"{element}: {filename}")
        return missing

    def _pseudo_z_valence(self, pseudo_file: str) -> float:
        """Valence electron count from a UPF header (cached)."""
        cached = self._z_valence_cache.get(pseudo_file)
        if cached is not None:
            return cached
        text = (self.pseudo_dir / pseudo_file).read_text(errors="ignore")
        # UPF PP_HEADER line: e.g. "     19.00000000000      Z valence"
        val = 0.0
        for line in text.splitlines():
            if "Z valence" in line:
                for tok in line.split():
                    try:
                        val = float(tok)
                        break
                    except ValueError:
                        continue
                break
        self._z_valence_cache[pseudo_file] = val
        return val

    def _resolve(self, material: str, atoms: Optional[Atoms]) -> Tuple[Dict[str, Any], Atoms, float]:
        """Material profile + atoms + alat(bohr), from MATERIAL_DB or a raw structure."""
        if atoms is not None:
            db = profile_from_atoms(atoms, self.pseudo_dir)
            return db, atoms, db["_alat_bohr"]
        if material not in MATERIAL_DB:
            raise ValueError(f"Unknown material: {material}. Known: {list(MATERIAL_DB.keys())}")
        db = MATERIAL_DB[material]
        return db, build_atoms(material), get_lattice_constant_angstrom(material) * BOHR_PER_ANG

    def compile_t1(
        self,
        material: str,
        output_path: Path,
        *,
        prefix: Optional[str] = None,
        ecutwfc: Optional[float] = None,
        ecutrho: Optional[float] = None,
        kpoints: Optional[Tuple] = None,
        force_threshold: float = 1.0e-3,
        pressure_threshold: float = 0.5,
        conv_thr: float = 1.0e-8,
        cell_dynamics: str = "bfgs",
        nstep: int = 200,
        cell_optimization: bool = True,
        atoms: Optional[Atoms] = None,
    ) -> str:
        """Compile a T1 vc-relax input file.
        
        Args:
            material: Material key in MATERIAL_DB (label only when atoms is given)
            output_path: Where to write the .in file
            prefix: QE prefix (defaults to material lowercased)
            ecutwfc: Wavefunction cutoff in Ry
            ecutrho: Charge density cutoff in Ry
            kpoints: Override k-points tuple (nx, ny, nz, sx, sy, sz)
            force_threshold: Force convergence threshold Ry/Bohr
            pressure_threshold: Pressure threshold kbar
            conv_thr: SCF convergence threshold
            cell_dynamics: Cell dynamics algorithm
            atoms: Optional ASE Atoms from a structure file (overrides material)
        
        Returns:
            The generated input file content as string
        """
        db, atoms, alat_bohr = self._resolve(material, atoms)
        
        prefix = prefix or material.lower()
        ecutwfc = ecutwfc if ecutwfc is not None else db["ecutwfc_default"]
        ecutrho = ecutrho if ecutrho is not None else db["ecutrho_default"]
        kpoints = tuple(kpoints) if kpoints else tuple(db.get("kpoints_default", (4, 4, 4, 1, 1, 1)))
        
        # Check pseudos
        missing = self._check_pseudos(db["pseudos"])
        if missing:
            raise FileNotFoundError(f"Missing pseudopotentials: {missing}")
        
        # Build QE input
        lines = []
        calc_type = "vc-relax" if cell_optimization else "relax"
        lines.append("&CONTROL")
        lines.append(f'  calculation = "{calc_type}"')
        lines.append(f'  prefix = "{prefix}"')
        lines.append(f'  outdir = "./"')
        lines.append(f'  pseudo_dir = "{self.pseudo_dir}"')
        lines.append(f"  nstep = {nstep}")
        lines.append("/")
        lines.append("")
        lines.append("&SYSTEM")
        lines.append(f"  ibrav = {db['ibrav']}")
        lines.append(f"  celldm(1) = {alat_bohr:.6f}")
        lines.append(f"  nat = {len(atoms)}")
        lines.append(f"  ntyp = {db['nspecies']}")
        lines.append(f"  ecutwfc = {ecutwfc}")
        lines.append(f"  ecutrho = {ecutrho}")
        lines.append(f'  occupations = "{db["occupations"]}"')
        lines.append(f'  smearing = "{db["smearing"]}"')
        lines.append(f"  degauss = {db['degauss']}")
        if db.get("input_dft"):
            lines.append(f'  input_dft = "{db["input_dft"]}"')
        lines.append("/")
        lines.append("")
        lines.append("&ELECTRONS")
        lines.append(f"  conv_thr = {conv_thr:.1e}")
        lines.append("  mixing_beta = 0.7")
        lines.append("/")
        lines.append("")
        lines.append("&IONS")
        lines.append('  ion_dynamics = "bfgs"')
        lines.append("/")
        lines.append("")
        
        # CELL section: only for vc-relax (cell optimization)
        if cell_optimization:
            lines.append("&CELL")
            lines.append(f'  cell_dynamics = "{cell_dynamics}"')
            lines.append("  press = 0.0")
            lines.append("  wmass = 0.01")
            lines.append("/")
            lines.append("")
        
        # ATOMIC_SPECIES
        lines.append("ATOMIC_SPECIES")
        for elem, mass in zip(db["species"], db["masses"]):
            pseudo_file = db["pseudos"][elem]
            lines.append(f"  {elem}  {mass:.3f}  {pseudo_file}")
        lines.append("")
        
        # ATOMIC_POSITIONS (crystal) — primitive cell
        lines.append("ATOMIC_POSITIONS (crystal)")
        for sym, pos in zip(atoms.symbols, atoms.get_scaled_positions()):
            lines.append(f"  {sym}  {pos[0]:.8f}  {pos[1]:.8f}  {pos[2]:.8f}")
        lines.append("")
        
        # CELL_PARAMETERS: only for ibrav=0 (free cell)
        # For ibrav != 0, QE derives the cell from celldm(1) and bravais-lattice index
        if db["ibrav"] == 0:
            # atoms.cell is in Å; alat = celldm(1) is in Bohr — convert both
            # to the same unit before dividing, or the lattice shrinks by
            # BOHR_PER_ANG (cells at 53% size → garbage energies)
            alat_ang = alat_bohr / BOHR_PER_ANG
            lines.append("CELL_PARAMETERS (alat=1.0)")
            for vec in atoms.cell:
                lines.append(f"  {vec[0]/alat_ang:.8f}  {vec[1]/alat_ang:.8f}  {vec[2]/alat_ang:.8f}")
            lines.append("")
        
        # K_POINTS
        lines.append("K_POINTS automatic")
        lines.append(f"  {kpoints[0]} {kpoints[1]} {kpoints[2]}  {kpoints[3]} {kpoints[4]} {kpoints[5]}")
        lines.append("")
        
        content = "\n".join(lines)
        
        # Write to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content)
        
        return content

    def compile_scf(
        self,
        material: str,
        output_path: Path,
        *,
        prefix: Optional[str] = None,
        calculation: str = "scf",
        ecutwfc: Optional[float] = None,
        ecutrho: Optional[float] = None,
        kpoints: Optional[Tuple] = None,
        conv_thr: float = 1.0e-8,
        nbnd: Optional[int] = None,
        nkpoints_bands: int = 60,
        atoms: Optional[Atoms] = None,
        kmode: str = "path",
    ) -> str:
        """Compile a generic SCF or NSCF input file.

        Used for T2 bands/DOS steps. Pass ``atoms`` to compile from an
        arbitrary structure instead of a MATERIAL_DB material. ``kmode``:
        "path" (nscf along the high-symmetry band path, for bands.x) or
        "uniform" (dense automatic mesh, for dos.x / projwfc.x).
        """
        db, atoms, alat_bohr = self._resolve(material, atoms)
        
        prefix = prefix or material.lower()
        ecutwfc = ecutwfc if ecutwfc is not None else db["ecutwfc_default"]
        ecutrho = ecutrho if ecutrho is not None else db["ecutrho_default"]
        kpoints = tuple(kpoints) if kpoints else tuple(db.get("kpoints_default", (4, 4, 4, 1, 1, 1)))
        
        # Check pseudos
        missing = self._check_pseudos(db["pseudos"])
        if missing:
            raise FileNotFoundError(f"Missing pseudopotentials: {missing}")
        
        # Determine nbnd from the pseudopotentials' real valence electron
        # counts — GBRV semicore pseudos (e.g. Ga_sv with 19 valence e-)
        # make atom-count heuristics fatally wrong ("too few bands")
        if nbnd is None:
            nelec = sum(
                self._pseudo_z_valence(db["pseudos"][s]) for s in atoms.symbols
            )
            n_occ = int(math.ceil(nelec / 2.0))
            if db["is_metal"]:
                nbnd = max(len(atoms) * 4, n_occ + 12)
            else:
                nbnd = max(len(atoms) * 4, n_occ + 8)

        lines = []
        lines.append("&CONTROL")
        lines.append(f'  calculation = "{calculation}"')
        lines.append(f'  prefix = "{prefix}"')
        lines.append(f'  outdir = "./"')
        lines.append(f'  pseudo_dir = "{self.pseudo_dir}"')
        if calculation in ("bands", "nscf"):
            lines.append('  restart_mode = "from_scratch"')
        lines.append("/")
        lines.append("")
        lines.append("&SYSTEM")
        lines.append(f"  ibrav = {db['ibrav']}")
        lines.append(f"  celldm(1) = {alat_bohr:.6f}")
        lines.append(f"  nat = {len(atoms)}")
        lines.append(f"  ntyp = {db['nspecies']}")
        lines.append(f"  ecutwfc = {ecutwfc}")
        lines.append(f"  ecutrho = {ecutrho}")
        lines.append(f'  occupations = "{db["occupations"]}"')
        lines.append(f'  smearing = "{db["smearing"]}"')
        lines.append(f"  degauss = {db['degauss']}")
        lines.append(f"  nbnd = {nbnd}")
        lines.append("/")
        lines.append("")
        lines.append("&ELECTRONS")
        lines.append(f"  conv_thr = {conv_thr:.1e}")
        lines.append("  mixing_beta = 0.7")
        lines.append("/")
        lines.append("")
        
        # ATOMIC_SPECIES
        lines.append("ATOMIC_SPECIES")
        for elem, mass in zip(db["species"], db["masses"]):
            pseudo_file = db["pseudos"][elem]
            lines.append(f"  {elem}  {mass:.3f}  {pseudo_file}")
        lines.append("")
        
        # ATOMIC_POSITIONS
        lines.append("ATOMIC_POSITIONS (crystal)")
        for sym, pos in zip(atoms.symbols, atoms.get_scaled_positions()):
            lines.append(f"  {sym}  {pos[0]:.8f}  {pos[1]:.8f}  {pos[2]:.8f}")
        lines.append("")
        
        # CELL_PARAMETERS — only for ibrav=0; for ibrav != 0 QE derives the
        # cell from celldm(1) and emitting both is a fatal "redundant data" error
        if db["ibrav"] == 0:
            # atoms.cell (Å) must be divided by alat in Å, not in Bohr
            alat_ang = alat_bohr / BOHR_PER_ANG
            lines.append("CELL_PARAMETERS (alat=1.0)")
            for vec in atoms.cell:
                lines.append(f"  {vec[0]/alat_ang:.8f}  {vec[1]/alat_ang:.8f}  {vec[2]/alat_ang:.8f}")
            lines.append("")
        
        # K_POINTS — SCF uses a uniform grid; NSCF runs the high-symmetry
        # band path so bands.x can post-process the same k-points
        if calculation == "nscf" and kmode == "path":
            kpts_path = get_bandpath_kpoints(material, npoints=nkpoints_bands, atoms=atoms)
            lines.append("K_POINTS crystal")
            lines.append(f"  {len(kpts_path)}")
            # QE 7.x requires the weight column; equal weights for band paths
            for k in kpts_path:
                lines.append(f"  {k[0]:.8f}  {k[1]:.8f}  {k[2]:.8f}  1.0")
            lines.append("")
        else:
            if calculation == "nscf":
                # uniform nscf for DOS/PDOS: dense mesh, k=1 stays k=1
                kpoints = tuple(
                    (max(1, 2 * n) if n > 1 else 1) for n in kpoints[:3]
                ) + tuple(kpoints[3:])
            lines.append("K_POINTS automatic")
            lines.append(f"  {kpoints[0]} {kpoints[1]} {kpoints[2]}  {kpoints[3]} {kpoints[4]} {kpoints[5]}")
            lines.append("")
        
        content = "\n".join(lines)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content)
        return content

    def compile_bands_input(
        self,
        material: str,
        output_path: Path,
        *,
        prefix: Optional[str] = None,
        kpath: Optional[Tuple] = None,
        klabels: Optional[List[str]] = None,
        nkpoints: int = 100,
        atoms: Optional[Atoms] = None,
    ) -> str:
        """Compile a bands.x post-processing input file.

        Generates a high-symmetry k-path input suitable for bands.x.

        Args:
            material: Material key in MATERIAL_DB (label only when atoms is given)
            output_path: Where to write the .in file
            prefix: QE prefix (must match SCF/NSCF prefix)
            kpath: Override k-points as (N, 3) array in crystal coords
            klabels: Override k-point labels (one per k-point)
            nkpoints: Number of interpolated points between high-symmetry points
            atoms: Optional ASE Atoms from a structure file (overrides material)

        Returns:
            The generated input file content as string
        """
        if atoms is None and material not in MATERIAL_DB:
            raise ValueError(f"Unknown material: {material}")
        prefix = prefix or material.lower()

        # Labeled high-symmetry points from ASE's Bravais-aware bandpath;
        # npoints=0 tells bands.x to reuse the exact pw.x nscf k-point list
        if kpath is None or klabels is None:
            src = atoms if atoms is not None else build_atoms(material)
            segments = bandpath_segments_from_atoms(src)
            flat: List = []
            for seg in segments:
                for pt in seg:
                    if not flat or flat[-1][1] != pt[1]:
                        flat.append(pt)
            kpath = [coords for _label, coords in flat]
            klabels = [_LABEL_MAP.get(label, label) for label, _coords in flat]

        lines = []
        lines.append("&bands")
        lines.append(f'  prefix = "{prefix}"')
        lines.append('  outdir = "./"')
        lines.append(f"  lsym = .true.")
        lines.append(f"  no_overlap = .true.")
        lines.append("/")
        lines.append("")
        lines.append(f"{len(kpath)}")
        for i, (kp, label) in enumerate(zip(kpath, klabels)):
            lines.append(
                f"  {kp[0]:.8f}  {kp[1]:.8f}  {kp[2]:.8f}  {label}"
            )
        lines.append("")
        lines.append("0")
        lines.append("")

        content = "\n".join(lines)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content)
        return content

    def compile_dos_input(
        self,
        material: str,
        output_path: Path,
        *,
        prefix: Optional[str] = None,
        emin: Optional[float] = None,
        emax: Optional[float] = None,
        deltae: float = 0.01,
        fwhm: float = 0.05,
        ngauss: int = 1,
        atoms: Optional[Atoms] = None,
    ) -> str:
        """Compile a dos.x post-processing input file.

        Args:
            material: Material key in MATERIAL_DB (label only when atoms is given)
            output_path: Where to write the .in file
            prefix: QE prefix (must match SCF/NSCF prefix)
            emin: Minimum energy in eV (None = auto)
            emax: Maximum energy in eV (None = auto)
            deltae: Energy grid step in eV
            fwhm: Broadening width in eV
            ngauss: Broadening type (1=Gaussian, 0=Methfessel-Paxton)
            atoms: Optional ASE Atoms (accepted for symmetry with other compilers)

        Returns:
            The generated input file content as string
        """
        if atoms is None and material not in MATERIAL_DB:
            raise ValueError(f"Unknown material: {material}")

        prefix = prefix or material.lower()

        lines = []
        lines.append("&dos")
        lines.append(f'  prefix = "{prefix}"')
        lines.append('  outdir = "./"')
        lines.append(f"  ngauss = {ngauss}")
        lines.append(f"  degauss = {fwhm}")
        lines.append(f"  DeltaE = {deltae}")
        if emin is not None:
            lines.append(f"  Emin = {emin}")
        if emax is not None:
            lines.append(f"  Emax = {emax}")
        lines.append("/")

        content = "\n".join(lines) + "\n"  # trailing newline: QE 7.5 dos.x namelist reader aborts without it
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content)
        return content

    def compile_pdos_input(
        self,
        material: str,
        output_path: Path,
        *,
        prefix: Optional[str] = None,
        deltae: float = 0.01,
        fwhm: float = 0.05,
        ngauss: int = 1,
        atoms: Optional[Atoms] = None,
    ) -> str:
        """Compile a projwfc.x input for projected DOS.

        Requires a prior nscf with the same prefix; produces one
        {prefix}.pdos_atm#N(wfc#M) file per atomic wavefunction plus
        {prefix}.pdos_tot for the total DOS.
        """
        if atoms is None and material not in MATERIAL_DB:
            raise ValueError(f"Unknown material: {material}")

        prefix = prefix or material.lower()

        lines = []
        lines.append("&projwfc")
        lines.append(f'  prefix = "{prefix}"')
        lines.append('  outdir = "./"')
        lines.append(f'  filpdos = "{prefix}"')
        lines.append(f"  ngauss = {ngauss}")
        lines.append(f"  degauss = {fwhm}")
        lines.append(f"  DeltaE = {deltae}")
        # QE 7.x writes the pdos files ONLY when lsym=.true. (projwfc.f90:
        # "ELSE IF ( lsym .OR. kresolveddos ) THEN CALL partialdos")
        lines.append("  lsym = .true.")
        lines.append("/")
        # trailing newline: QE namelist readers abort without it
        content = "\n".join(lines) + "\n"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content)
        return content

    def compute_input_hash(self, content: str) -> str:
        """Compute SHA-256 hash of input file content for reproducibility."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]
