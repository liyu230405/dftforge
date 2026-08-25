"""2D structure builder: monolayers, supercells, doping, vacancies, adsorption sites.

Deterministic geometry on hexagonal lattices (graphene / h-BN):
- top site    : directly above a lattice atom
- bridge site : midpoint between two nearest neighbours
- hollow site : hexagon centre = atom + (v1+v2+v3)/3 of its 3 nearest neighbours

All modifications validate a minimum interatomic distance so obviously
broken structures fail fast instead of producing garbage QE runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ase import Atom, Atoms
from ase.build import graphene as ase_graphene
from ase.build import make_supercell as ase_make_supercell
from ase.neighborlist import neighbor_list

MIN_DISTANCE_ANGSTROM = 0.9
DEFAULT_VACUUM = 15.0
DEFAULT_ADS_HEIGHT = 1.5


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
    else:
        raise StructureBuildError(f"unsupported monolayer kind: {kind} (graphene | bn)")
    # ASE vacuum is per-side; we promise the total vacuum layer
    atoms.center(vacuum=vacuum / 2, axis=2)
    return atoms


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

    Geometry assumes a hexagonal sheet; positions are in Å with z above the
    topmost atom layer.
    """
    z_top = max(atoms.positions[:, 2])
    layer = [i for i in range(len(atoms)) if abs(atoms.positions[i, 2] - z_top) < 0.1]
    ref = layer[site_index % len(layer)]
    pos = atoms.positions[ref]

    idx_i, idx_j, d = neighbor_list("ijD", atoms, cutoff=2.0)
    nbrs = sorted(
        [(j, tuple(D)) for i, j, D in zip(idx_i, idx_j, d) if i == ref],
        key=lambda t: (t[1][0] ** 2 + t[1][1] ** 2 + t[1][2] ** 2),
    )
    if len(nbrs) < 3:
        raise StructureBuildError("reference atom has <3 neighbours within 2.0 Å — not a hexagonal sheet?")

    def xy(v: Tuple[float, ...]) -> List[float]:
        return [pos[0] + v[0], pos[1] + v[1], pos[2]]

    # honeycomb geometry: the three neighbour vectors sum to zero, so the
    # hexagon centre is atom + (v1 + v2) — two neighbours 120° apart, whose
    # sum has magnitude equal to the bond length
    v1, v2 = nbrs[0][1], nbrs[1][1]
    top = xy((0.0, 0.0, 0.0))
    bridge = xy((v1[0] / 2, v1[1] / 2, 0.0))
    hollow = xy((v1[0] + v2[0], v1[1] + v2[1], 0.0))
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
