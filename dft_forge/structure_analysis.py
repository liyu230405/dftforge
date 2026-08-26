"""Structure geometry analysis: bonds, angles, coordination, symmetry.

Shared by the CLI (structure.analyze) and the QE engine (post-relax analysis)
so every structure — imported, built, or relaxed — gets the same report.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from ase import Atoms
from ase.data import covalent_radii

# bond cutoff factor: first shell = distances within 1.25x the shortest
# interatomic distance in the structure (robust for ionic crystals like NaCl,
# layered MoS2, and molecules; fixed covalent-radii cutoffs miss Na-Cl bonds
# and create false Mo-Mo bonds)
BOND_SCALE = 1.25


def _neighbor_candidates(atoms: Atoms, pair_cutoff: float):
    """(i, j, offset, distance) bond candidates within pair_cutoff.

    ASE's NeighborList returns pairs within the SUM of per-atom cutoffs plus
    a skin, so callers must distance-filter themselves. Periodic images are
    separate candidates: the six Na-Cl bonds of rocksalt NaCl are six
    instances of one atom pair under different offsets.
    """
    from ase.neighborlist import NeighborList

    nl = NeighborList(
        [pair_cutoff / 2] * len(atoms),
        self_interaction=False,
        bothways=True,
    )
    nl.update(atoms)
    cell = atoms.get_cell().array
    for i in range(len(atoms)):
        for j, offset in zip(*nl.get_neighbors(i)):
            j = int(j)
            d = float(np.linalg.norm(atoms.positions[j] + offset @ cell - atoms.positions[i]))
            if 1e-6 < d <= pair_cutoff:
                yield i, j, np.asarray(offset, dtype=float), d


def _bond_cutoff(atoms: Atoms) -> float:
    """First-shell cutoff: 1.25x the shortest interatomic distance (MIC)."""
    search = max(2.0 * float(covalent_radii[Z]) for Z in atoms.numbers)
    d_min = min((d for _i, _j, _o, d in _neighbor_candidates(atoms, 2 * search)), default=float("inf"))
    return BOND_SCALE * d_min if np.isfinite(d_min) else 0.0


def coordination_and_bonds(atoms: Atoms) -> Dict[str, Any]:
    """Bond lengths, coordination numbers, and bond angles per triplet."""
    symbols = atoms.get_chemical_symbols()
    out: Dict[str, Any] = {}

    try:
        cutoff = _bond_cutoff(atoms)
        if cutoff <= 0:
            return {}
        # unique bond instances: each periodic image counted once (a bond
        # appears as (i,j,off) in i's list and (j,i,-off) in j's list)
        bonds: List[tuple] = []
        seen: set = set()
        neighbor_vectors: Dict[int, List[tuple]] = {i: [] for i in range(len(atoms))}
        cell = atoms.get_cell().array
        for i, j, offset, d in _neighbor_candidates(atoms, cutoff):
            vec = atoms.positions[j] + offset @ cell - atoms.positions[i]
            neighbor_vectors[i].append((vec, symbols[j]))
            if i < j:
                key = (i, j, tuple(int(x) for x in offset))
            elif i > j:
                key = (j, i, tuple(int(-x) for x in offset))
            else:
                off = tuple(int(x) for x in offset)
                key = (i, i, max(off, tuple(-x for x in off)))
            if key not in seen:
                seen.add(key)
                bonds.append((key[0], key[1], d))
    except Exception as exc:  # degenerate cell / overlapping atoms
        return {"bond_analysis_error": str(exc)}

    # ── coordination ──
    coord: Dict[int, int] = {i: 0 for i in range(len(atoms))}
    bond_by_pair: Dict[str, list] = {}
    for i, j, d in bonds:
        coord[i] += 1
        coord[j] += 1
        pair = "-".join(sorted((symbols[i], symbols[j])))
        bond_by_pair.setdefault(pair, []).append(d)

    coord_by_species: Dict[str, list] = {}
    for i, c in coord.items():
        coord_by_species.setdefault(symbols[i], []).append(c)

    out["coordination"] = {
        sp: {"min": min(v), "max": max(v), "mean": float(np.mean(v))}
        for sp, v in sorted(coord_by_species.items())
    }
    out["bond_lengths"] = {
        pair: {
            "count": len(v),
            "min_angstrom": round(min(v), 4),
            "mean_angstrom": round(float(np.mean(v)), 4),
            "max_angstrom": round(max(v), 4),
        }
        for pair, v in sorted(bond_by_pair.items())
    }

    # ── bond angles: A-B-C over bonded neighbor pairs of center B ──
    angle_by_triplet: Dict[str, list] = {}
    for b in range(len(atoms)):
        ns = neighbor_vectors[b]
        for ai in range(len(ns)):
            for ci in range(ai + 1, len(ns)):
                v1, s1 = ns[ai]
                v2, s2 = ns[ci]
                n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
                if n1 < 1e-9 or n2 < 1e-9:
                    continue
                # manual arccos handles exactly collinear bonds (180°), which
                # ase.Atoms.get_angle raises ZeroDivisionError on
                cosang = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
                ang = float(np.degrees(np.arccos(cosang)))
                trip = "-".join((s1, symbols[b], s2))
                angle_by_triplet.setdefault(trip, []).append(ang)

    out["bond_angles"] = {
        trip: {
            "count": len(v),
            "min_deg": round(min(v), 2),
            "mean_deg": round(float(np.mean(v)), 2),
            "max_deg": round(max(v), 2),
        }
        for trip, v in sorted(angle_by_triplet.items())
    }
    return out


def symmetry_info(atoms: Atoms, symprec: float = 0.01) -> Dict[str, Any]:
    """Space group + primitive/conventional cell relationship via spglib."""
    out: Dict[str, Any] = {"available": False}
    try:
        import spglib
    except ImportError:
        out["error"] = "spglib not installed"
        return out

    try:
        cell = (
            atoms.get_cell().array,
            atoms.get_scaled_positions(),
            atoms.numbers,
        )
        ds = spglib.get_symmetry_dataset(cell, symprec=symprec)
        if ds is None:
            out["error"] = "symmetry detection failed"
            return out

        number = int(ds.number)
        intl = str(ds.international).strip()
        out.update({
            "available": True,
            "space_group": intl,
            "space_group_number": number,
            "hall": str(ds.hall).strip(),
            "n_symmetry_operations": len(ds.rotations),
        })

        # conventional (standard) cell
        conv = spglib.standardize_cell(cell, to_primitive=False, no_idealize=False, symprec=symprec)
        prim = spglib.standardize_cell(cell, to_primitive=True, no_idealize=False, symprec=symprec)
        if conv is not None:
            lattice, positions, numbers = conv
            conv_atoms = Atoms(numbers=numbers, scaled_positions=positions, cell=lattice, pbc=atoms.pbc)
            out["conventional_cell"] = {
                "formula": conv_atoms.get_chemical_formula(),
                "natoms": len(conv_atoms),
                "a_angstrom": round(float(np.linalg.norm(lattice[0])), 4),
                "b_angstrom": round(float(np.linalg.norm(lattice[1])), 4),
                "c_angstrom": round(float(np.linalg.norm(lattice[2])), 4),
                "volume_angstrom3": round(float(abs(np.linalg.det(lattice))), 4),
                "multiplicity_vs_input": round(len(conv_atoms) / max(len(atoms), 1), 2),
            }
        if prim is not None:
            lattice, positions, numbers = prim
            out["primitive_cell"] = {
                "natoms": len(numbers),
                "multiplicity_vs_input": round(len(numbers) / max(len(atoms), 1), 2),
            }
    except Exception as exc:
        out["error"] = str(exc)
    return out


def analyze(atoms: Atoms, pair_types=None) -> Dict[str, Any]:
    """Full structure report: geometry + bonding + symmetry."""
    result: Dict[str, Any] = {
        "formula": atoms.get_chemical_formula(),
        "formula_reduced": atoms.get_chemical_formula(mode="reduce"),
        "natoms": len(atoms),
        "species": sorted(set(atoms.get_chemical_symbols())),
        "lattice_vectors_angstrom": atoms.cell.tolist(),
        "lattice_abc_angstrom": [round(float(x), 4) for x in atoms.cell.cellpar()[:3]],
        "lattice_angles_deg": [round(float(x), 2) for x in atoms.cell.cellpar()[3:]],
        "volume_angstrom3": round(float(abs(np.linalg.det(atoms.cell))), 4),
        "pbc": atoms.pbc.tolist() if hasattr(atoms.pbc, "tolist") else list(atoms.pbc),
        "is_2d": bool(any(~np.asarray(atoms.pbc, dtype=bool))),
    }

    # nearest-neighbor floor (raw distances, no MIC — same as before)
    if len(atoms) >= 2:
        positions = atoms.get_positions()
        symbols = atoms.get_chemical_symbols()
        min_dist = float("inf")
        min_pair = None
        pair_dists: Dict[str, list] = {}
        for i in range(len(atoms)):
            for j in range(i + 1, len(atoms)):
                pair = tuple(sorted((symbols[i], symbols[j])))
                allowed = True
                if pair_types:
                    allowed = pair in pair_types or (symbols[i], symbols[j]) in pair_types or (symbols[j], symbols[i]) in pair_types
                if not allowed:
                    continue
                dist = float(np.linalg.norm(positions[i] - positions[j]))
                if dist <= 1e-12:
                    continue
                pair_dists.setdefault(str(pair), []).append(dist)
                if dist < min_dist:
                    min_dist = dist
                    min_pair = pair

        result["minimum_distance_angstrom"] = round(min_dist, 4) if min_dist != float("inf") else None
        result["minimum_distance_pair"] = min_pair
        result["pair_distances"] = {
            pair: {
                "count": len(vals),
                "min_angstrom": round(min(vals), 4),
                "max_angstrom": round(max(vals), 4),
                "mean_angstrom": round(float(np.mean(vals)), 4),
            }
            for pair, vals in pair_dists.items()
        }

    result.update(coordination_and_bonds(atoms))
    result["symmetry"] = symmetry_info(atoms)
    return result
