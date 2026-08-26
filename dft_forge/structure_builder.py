"""2D structure builder: monolayers, supercells, doping, vacancies, adsorption sites.

Deterministic geometry on hexagonal lattices (graphene / h-BN):
- top site    : directly above a lattice atom
- bridge site : midpoint between two nearest neighbours
- hollow site : hexagon centre = atom + (v1+v2+v3)/3 of its 3 nearest neighbours

All modifications validate a minimum interatomic distance so obviously
broken structures fail fast instead of producing garbage QE runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from ase import Atom, Atoms
from ase.build import graphene as ase_graphene
from ase.build import make_supercell as ase_make_supercell
from ase.neighborlist import neighbor_list

MIN_DISTANCE_ANGSTROM = 0.9
DEFAULT_VACUUM = 15.0
DEFAULT_ADS_HEIGHT = 1.5

# 1H-phase TMD monolayers: kind -> (metal, chalcogen, a [Å], M-X bond [Å]).
# a is the hexagonal lattice constant; the vertical S-M-S offset follows
# from sqrt(bond² - (a/√3)²), so only (a, bond) is tabulated.
TMD_MONOLAYERS: Dict[str, Tuple[str, str, float, float]] = {
    "mos2": ("Mo", "S", 3.16, 2.41),
    "ws2": ("W", "S", 3.15, 2.41),
    "mose2": ("Mo", "Se", 3.29, 2.38),
    "wse2": ("W", "Se", 3.28, 2.41),
    "mote2": ("Mo", "Te", 3.52, 2.73),
    "wte2": ("W", "Te", 3.51, 2.72),
}


class StructureBuildError(ValueError):
    """Raised when a structure modification would be geometrically invalid."""


@dataclass
class BuildResult:
    atoms: Atoms
    description: str = ""
    sites: Dict[str, List[float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "formula": self.atoms.get_chemical_formula(),
            "natoms": len(self.atoms),
            "species": sorted(set(self.atoms.get_chemical_symbols())),
            "cell": [list(map(float, row)) for row in self.atoms.cell[:]],
            "volume": float(self.atoms.get_volume()),
            "description": self.description,
            "sites": {k: [round(c, 4) for c in v] for k, v in self.sites.items()},
            "min_distance": round(float(min_distance(self.atoms)), 4),
        }


def min_distance(atoms: Atoms) -> float:
    if len(atoms) < 2:
        return float("inf")
    d = neighbor_list("d", atoms, cutoff=6.0)
    return float(d.min()) if len(d) else float("inf")


def check_distances(atoms: Atoms, what: str) -> None:
    dmin = min_distance(atoms)
    if dmin < MIN_DISTANCE_ANGSTROM:
        raise StructureBuildError(
            f"{what}: minimum interatomic distance {dmin:.3f} Å < {MIN_DISTANCE_ANGSTROM} Å"
        )


def build_monolayer(kind: str, *, a: Optional[float] = None, vacuum: float = DEFAULT_VACUUM) -> Atoms:
    kind = kind.lower().replace("-", "").replace("_", "")
    if kind == "graphene":
        atoms = ase_graphene(a=a or 2.46)
    elif kind == "bn" or kind == "hbn":
        atoms = ase_graphene(a=a or 2.51)
        atoms.symbols[0] = "B"
        atoms.symbols[1] = "N"
    elif kind in TMD_MONOLAYERS:
        atoms = _tmd_monolayer(kind, vacuum=vacuum)
    else:
        raise StructureBuildError(
            f"unsupported monolayer kind: {kind} (graphene | bn | {' | '.join(TMD_MONOLAYERS)})"
        )
    # ASE vacuum is per-side; we promise the total vacuum layer
    if kind not in TMD_MONOLAYERS:
        atoms.center(vacuum=vacuum / 2, axis=2)
    return atoms


def _tmd_monolayer(kind: str, *, vacuum: float = DEFAULT_VACUUM) -> Atoms:
    """1H-phase TMD monolayer: X-M-X sandwich on a hexagonal cell.

    M at (1/3, 2/3), X at (2/3, 1/3) ± dz — in-plane M-X distance a/√3,
    dz = sqrt(bond² − (a/√3)²) fixes the S-M-S height.
    """
    import math as _math

    metal, chalc, a, bond = TMD_MONOLAYERS[kind]
    inplane = a / _math.sqrt(3.0)
    dz = _math.sqrt(bond**2 - inplane**2)
    c = 2.0 * dz + vacuum
    cell = [[a, 0, 0], [-a / 2, a * _math.sqrt(3) / 2, 0], [0, 0, c]]
    pos = [
        (1 / 3, 2 / 3, 0.5),
        (2 / 3, 1 / 3, 0.5 + dz / c),
        (2 / 3, 1 / 3, 0.5 - dz / c),
    ]
    return Atoms([metal, chalc, chalc], scaled_positions=pos, cell=cell, pbc=True)


def make_supercell(atoms: Atoms, m: int = 1, n: int = 1) -> Atoms:
    if (m, n) == (1, 1):
        return atoms
    P = [[m, 0, 0], [0, n, 0], [0, 0, 1]]
    return ase_make_supercell(atoms, P)


def substitute(atoms: Atoms, index: int, element: str) -> Atoms:
    if not 0 <= index < len(atoms):
        raise StructureBuildError(f"substitute index {index} out of range (0..{len(atoms) - 1})")
    atoms.symbols[index] = element
    return atoms


def remove_atom(atoms: Atoms, index: int) -> Atoms:
    if not 0 <= index < len(atoms):
        raise StructureBuildError(f"vacancy index {index} out of range (0..{len(atoms) - 1})")
    del atoms[index]
    return atoms


def surface_sites(atoms: Atoms, *, site_index: int = 0) -> Dict[str, List[float]]:
    """top / bridge / hollow sites on the topmost layer of a 2D slab.

    Geometry works for both honeycomb sheets (graphene, h-BN: 3 in-plane
    neighbours, hexagon centre = atom + v1 + v2) and TMD chalcogen
    triangular layers (6 neighbours, triangle centre = atom + (v1+v2)/3).
    Neighbours are filtered to the topmost layer so the metal below a
    TMD chalcogen plane is not mistaken for a surface neighbour.
    """
    z_top = max(atoms.positions[:, 2])
    layer = [i for i in range(len(atoms)) if abs(atoms.positions[i, 2] - z_top) < 0.1]
    ref = layer[site_index % len(layer)]
    pos = atoms.positions[ref]

    # generous cutoff: TMD in-plane X-X spacing (a ≈ 3.2 Å) exceeds the
    # graphene bond (1.42 Å), so 2.0 Å would find nothing
    idx_i, idx_j, idx_D = neighbor_list("ijD", atoms, cutoff=4.0)
    nbrs = sorted(
        [
            (j, tuple(D))
            for i, j, D in zip(idx_i, idx_j, idx_D)
            if i == ref and abs(D[2]) < 0.1 and j in layer
        ],
        key=lambda t: (t[1][0] ** 2 + t[1][1] ** 2 + t[1][2] ** 2),
    )
    if len(nbrs) < 2:
        raise StructureBuildError("reference atom has <2 in-plane neighbours — not a 2D sheet?")

    def xy(v: Tuple[float, ...]) -> List[float]:
        return [pos[0] + v[0], pos[1] + v[1], pos[2]]

    v1, v2 = nbrs[0][1], nbrs[1][1]
    d1 = math.hypot(v1[0], v1[1])
    # coordination within the first shell only — a 4 Å cutoff also returns
    # second-shell atoms (e.g. graphene's 2.46 Å), which must not flip the
    # lattice classification
    coord = sum(1 for _j, v in nbrs if math.hypot(v[0], v[1]) < 1.2 * d1)
    div = 3.0 if coord >= 5 else 1.0
    hollow_v = ((v1[0] + v2[0]) / div, (v1[1] + v2[1]) / div)
    top = xy((0.0, 0.0, 0.0))
    bridge = xy((v1[0] / 2, v1[1] / 2, 0.0))
    hollow = xy(hollow_v)
    return {"top": top, "bridge": bridge, "hollow": hollow}


def add_adsorbate(
    atoms: Atoms,
    element: str,
    site: str = "top",
    *,
    height: float = DEFAULT_ADS_HEIGHT,
    site_index: int = 0,
) -> Atoms:
    site = site.lower()
    if site not in ("top", "bridge", "hollow"):
        raise StructureBuildError(f"unsupported adsorption site: {site} (top | bridge | hollow)")
    pos = surface_sites(atoms, site_index=site_index)[site]
    atoms.append(Atom(element, position=[pos[0], pos[1], pos[2] + height]))
    check_distances(atoms, f"adsorb {element}@{site}")
    return atoms


def build_2d(
    kind: str,
    *,
    supercell: str = "1x1",
    vacancy_index: Optional[int] = None,
    dopants: Optional[List[Dict[str, Any]]] = None,
    adsorbate: Optional[Dict[str, Any]] = None,
    height: float = DEFAULT_ADS_HEIGHT,
    vacuum: float = DEFAULT_VACUUM,
    a: Optional[float] = None,
) -> BuildResult:
    """Build a 2D material with optional supercell / vacancy / doping / adsorbate.

    ``dopants``  : [{"index": int, "element": "N"}, ...]
    ``adsorbate``: {"element": "O", "site": "hollow", "height": 1.5}
    """
    parts: List[str] = [kind]
    atoms = build_monolayer(kind, a=a, vacuum=vacuum)

    m, n = parse_supercell(supercell)
    if (m, n) != (1, 1):
        atoms = make_supercell(atoms, m, n)
        parts.append(f"{m}x{n} supercell")

    if vacancy_index is not None:
        atoms = remove_atom(atoms, vacancy_index)
        parts.append(f"vacancy@{vacancy_index}")

    for d in dopants or []:
        atoms = substitute(atoms, int(d["index"]), str(d["element"]))
        parts.append(f"{d['element']}@{d['index']}")

    check_distances(atoms, "after lattice modifications")

    try:
        sites = surface_sites(atoms, site_index=int((adsorbate or {}).get("site_index", 0)))
    except StructureBuildError:
        # informational only (e.g. vacancy adjacent to the reference atom);
        # a real adsorbate call re-raises with a precise error
        sites = {}
    if adsorbate:
        atoms = add_adsorbate(
            atoms,
            str(adsorbate["element"]),
            str(adsorbate.get("site", "top")),
            height=float(adsorbate.get("height", height)),
            site_index=int(adsorbate.get("site_index", 0)),
        )
        parts.append(f"{adsorbate['element']}@{adsorbate.get('site', 'top')}")

    return BuildResult(atoms=atoms, description=" + ".join(parts), sites=sites)


def parse_supercell(spec: str) -> Tuple[int, int]:
    spec = (spec or "1x1").lower().strip()
    try:
        m, n = spec.split("x", 1)
        m, n = int(m), int(n)
        if m < 1 or n < 1:
            raise ValueError
    except ValueError:
        raise StructureBuildError(f"bad supercell spec: {spec!r} (expected e.g. '3x3')") from None
    return m, n


def parse_supercell_3d(spec: str) -> Tuple[int, int, int]:
    """Parse '2x2x2' (or '2x2') into an (m, n, p) tuple."""
    spec = (spec or "1x1x1").lower().strip()
    parts = spec.split("x")
    try:
        nums = tuple(int(p) for p in parts)
        if len(nums) == 2:
            nums = nums + (1,)
        if len(nums) != 3 or any(p < 1 for p in nums):
            raise ValueError
    except ValueError:
        raise StructureBuildError(f"bad supercell spec: {spec!r} (expected e.g. '2x2x2')") from None
    return nums


def _pick_substitution_site(atoms: Atoms, dopant: str) -> int:
    """Index of the host atom most likely to be substituted: same element,
    else the element closest in atomic number to the dopant."""
    from ase.data import atomic_numbers

    z_d = atomic_numbers.get(dopant)
    syms = atoms.get_chemical_symbols()
    best_el, best_d = None, None
    for el in sorted(set(syms)):
        d = 0 if z_d is None else abs(atomic_numbers.get(el, 999) - z_d)
        if best_d is None or d < best_d:
            best_el, best_d = el, d
    return syms.index(best_el)


def dope_3d(
    source: Any,
    element: str,
    *,
    supercell: str = "2x2x2",
    index: Optional[int] = None,
) -> BuildResult:
    """Dope a bulk crystal: supercell, then substitute one host atom.

    ``source`` is a material key ('Si'), a formula ('CaTiO3'), or a path to
    a CIF/POSCAR file. Returns the doped supercell with a concentration
    estimate in the description. When ``index`` is None the substituted
    site is auto-picked: the host element closest in atomic number to the
    dopant (Nb→Ti in CaTiO3, P→Si in Si).
    """
    from dft_forge.compiler import MATERIAL_DB, build_atoms, formula_atoms, read_structure

    src = str(source)
    path = Path(src)
    if path.suffix.lower() in (".cif", ".poscar", ".xyz") and path.exists():
        atoms = read_structure(path)
    elif src in MATERIAL_DB:
        atoms = build_atoms(src)
    else:
        atoms, _proto = formula_atoms(src)

    m, n, p = parse_supercell_3d(supercell)
    if (m, n, p) != (1, 1, 1):
        P = [[m, 0, 0], [0, n, 0], [0, 0, p]]
        atoms = ase_make_supercell(atoms, P)
    if index is None:
        index = _pick_substitution_site(atoms, str(element))
    host = atoms.symbols[index]
    atoms = substitute(atoms, index, str(element))
    check_distances(atoms, f"dope {element}@{index}")

    frac = 100.0 / len(atoms)
    n_el = atoms.get_chemical_symbols().count(str(element))
    description = f"{host}→{element} in {m}x{n}x{p} supercell ({frac:.1f}% nominal, {n_el} atom)"
    return BuildResult(atoms=atoms, description=description)


# ── Gas molecules (adsorption references) and elemental reference phases ──────

# bond lengths in Å; boxed in a cube with BOX edge so images never interact
MOLECULES: Dict[str, list] = {
    # name: [(element, x, y, z), ...]
    "h2": [("H", -0.370, 0.0, 0.0), ("H", 0.370, 0.0, 0.0)],
    "o2": [("O", -0.605, 0.0, 0.0), ("O", 0.605, 0.0, 0.0)],
    "n2": [("N", -0.550, 0.0, 0.0), ("N", 0.550, 0.0, 0.0)],
    "cl2": [("Cl", -0.995, 0.0, 0.0), ("Cl", 0.995, 0.0, 0.0)],
    "co": [("C", -0.570, 0.0, 0.0), ("O", 0.570, 0.0, 0.0)],
    "oh": [("O", -0.490, 0.0, 0.0), ("H", 0.490, 0.0, 0.0)],
    "no": [("N", -0.575, 0.0, 0.0), ("O", 0.575, 0.0, 0.0)],
    "h2o": [("O", 0.0, 0.117, 0.0), ("H", 0.757, -0.470, 0.0), ("H", -0.757, -0.470, 0.0)],
    "co2": [("C", 0.0, 0.0, 0.0), ("O", 1.160, 0.0, 0.0), ("O", -1.160, 0.0, 0.0)],
    "nh3": [("N", 0.0, 0.0, 0.100), ("H", 0.937, 0.0, -0.265), ("H", -0.469, 0.812, -0.265), ("H", -0.469, -0.812, -0.265)],
}

# elemental reference phases for formation energies: (formula, kind, params)
# kind: diamond | bcc | fcc | hcp | molecule — phase chosen by convention
# (standard DFT reference set, e.g. the Materials Project / AFLOW conventions)
REFERENCE_PHASES: Dict[str, tuple] = {
    "Si": ("diamond", 5.430),
    "Ge": ("diamond", 5.658),
    "C": ("diamond", 3.567),
    "Sn": ("diamond", 6.489),
    "Na": ("bcc", 4.231),
    "K": ("bcc", 5.247),
    "Mo": ("bcc", 3.147),
    "W": ("bcc", 3.165),
    "V": ("bcc", 3.024),
    "Nb": ("bcc", 3.301),
    "Ta": ("bcc", 3.306),
    "Cr": ("bcc", 2.884),
    "Fe": ("bcc", 2.866),
    "Ca": ("fcc", 5.588),
    "Al": ("fcc", 4.050),
    "Ni": ("fcc", 3.524),
    "Cu": ("fcc", 3.615),
    "Ag": ("fcc", 4.085),
    "Au": ("fcc", 4.078),
    "Pt": ("fcc", 3.924),
    "Pd": ("fcc", 3.890),
    "Pb": ("fcc", 4.951),
    "Ti": ("hcp", 2.951, 4.684),
    "Zr": ("hcp", 3.232, 5.147),
    "Hf": ("hcp", 3.195, 5.051),
    "Mg": ("hcp", 3.209, 5.211),
    "Zn": ("hcp", 2.665, 4.947),
    "Co": ("hcp", 2.507, 4.070),
    "O": ("molecule", "o2"),
    "N": ("molecule", "n2"),
    "H": ("molecule", "h2"),
    "Cl": ("molecule", "cl2"),
    "F": ("molecule", "f2"),
}

_BOX = 10.0


def build_molecule(kind: str, *, box: float = _BOX) -> BuildResult:
    """Build a gas-phase molecule in a cubic box (Gamma-point SCF reference)."""
    key = kind.lower().replace("-", "").replace("_", "").replace(" ", "")
    if key not in MOLECULES:
        raise StructureBuildError(
            f"unknown molecule '{kind}'; available: {', '.join(sorted(MOLECULES))}"
        )
    spec = MOLECULES[key]
    symbols = [s for s, *_ in spec]
    pos = [[x + box / 2, y + box / 2, z + box / 2] for _, x, y, z in spec]
    # pbc=False keeps every k-mesh at Γ (molecules have no periodicity)
    atoms = Atoms(symbols, positions=pos, cell=[box, box, box], pbc=False)
    check_distances(atoms, f"molecule {key}")
    name = {"o2": "O₂", "n2": "N₂", "h2": "H₂", "cl2": "Cl₂", "f2": "F₂",
            "co": "CO", "oh": "OH", "no": "NO", "h2o": "H₂O", "co2": "CO₂",
            "nh3": "NH₃"}.get(key, key)
    return BuildResult(atoms=atoms, description=f"{name} 分子（{box:g} Å 立方盒子，Γ 点）")


def build_reference(element: str) -> BuildResult:
    """Elemental reference phase for formation-energy bookkeeping."""
    el = element.strip()
    el = el.capitalize() if len(el) > 1 else el.upper()
    if el not in REFERENCE_PHASES:
        raise StructureBuildError(
            f"no reference phase for '{el}' — supported: "
            + ", ".join(sorted(REFERENCE_PHASES))
        )
    spec = REFERENCE_PHASES[el]
    kind = spec[0]
    if kind == "molecule":
        res = build_molecule(spec[1])
        res.description = f"{el} 参考态: {res.description}"
        return res
    if kind == "diamond":
        a = spec[1]
        cell = [[0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]]
        atoms = Atoms(
            [el, el],
            scaled_positions=[(0, 0, 0), (0.25, 0.25, 0.25)],
            cell=cell, pbc=True,
        )
        phase = "金刚石结构"
    elif kind == "bcc":
        a = spec[1]
        atoms = Atoms(
            [el, el],
            scaled_positions=[(0, 0, 0), (0.5, 0.5, 0.5)],
            cell=np.eye(3) * a, pbc=True,
        )
        phase = "BCC"
    elif kind == "fcc":
        a = spec[1]
        cell = [[0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]]
        atoms = Atoms(
            [el] * 4,
            scaled_positions=[(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)],
            cell=cell, pbc=True,
        )
        phase = "FCC"
    else:  # hcp
        a, c = spec[1], spec[2]
        cell = [
            [a, 0, 0],
            [-a / 2, a * math.sqrt(3) / 2, 0],
            [0, 0, c],
        ]
        atoms = Atoms(
            [el] * 2,
            scaled_positions=[(1 / 3, 2 / 3, 0.25), (2 / 3, 1 / 3, 0.75)],
            cell=cell, pbc=True,
        )
        phase = "HCP"
    check_distances(atoms, f"reference {el}")
    return BuildResult(atoms=atoms, description=f"{el} 参考态: {phase}")


def reference_elements(formula: str) -> list:
    """Elements of a formula, each once, in composition order."""
    from dft_forge.compiler import formula_atoms

    atoms, _proto = formula_atoms(formula)
    seen = []
    for s in atoms.get_chemical_symbols():
        if s not in seen:
            seen.append(s)
    return seen
