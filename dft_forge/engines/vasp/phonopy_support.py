"""Phonopy bridge for finite-displacement phonon dispersion with VASP.

Phonopy is an optional, separately-installed dependency (BSD, freely
installable — unlike VASP/POTCAR). It is imported lazily so the VASP engine
works without it; every entry point fails closed with an actionable message
when phonopy is missing. No fabrication: displacements, forces and the
resulting dynamical matrix are all real phonopy/VASP artifacts.

The finite-displacement fan-out (an a-priori-unknown number of displaced
supercells) is deliberately kept *inside* a single graph node, so the static
JSON graph runtime needs no dynamic node expansion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

class PhonopyUnavailable(RuntimeError):
    """Raised when phonopy is not importable in the current environment."""


def _import_phonopy():
    """Import phonopy lazily; fail closed with an install hint."""
    try:
        import phonopy  # noqa: F401
        from phonopy import Phonopy  # noqa: F401
        from phonopy.structure.atoms import PhonopyAtoms  # noqa: F401
    except Exception as exc:  # ImportError or a broken install
        raise PhonopyUnavailable(
            "phonon dispersion needs the 'phonopy' package, which is not "
            "installed. Install it (`pip install phonopy`) to enable the "
            "v4_vasp_phonon_dispersion template. Γ-point frequencies via "
            "v3_vasp_phonon (native VASP DFPT) need no extra dependency."
        ) from exc
    return phonopy


def _to_phonopy_atoms(atoms) -> Any:
    """ASE Atoms -> PhonopyAtoms (both are simple cell/positions/species holders)."""
    _import_phonopy()
    from phonopy.structure.atoms import PhonopyAtoms

    return PhonopyAtoms(
        symbols=list(atoms.get_chemical_symbols()),
        cell=atoms.get_cell().array if hasattr(atoms.get_cell(), "array") else atoms.get_cell(),
        scaled_positions=atoms.get_scaled_positions(),
    )


def _to_ase_atoms(ph_atoms) -> Any:
    """PhonopyAtoms -> ASE Atoms for reuse of the deterministic VASP writer."""
    from ase import Atoms

    return Atoms(
        symbols=list(ph_atoms.symbols),
        scaled_positions=ph_atoms.scaled_positions,
        cell=ph_atoms.cell,
        pbc=True,
    )


def build_phonopy(atoms, supercell_matrix: List[List[int]], distance: float = 0.01):
    """Construct a Phonopy object and generate symmetry-reduced displacements."""
    _import_phonopy()
    from phonopy import Phonopy

    phonon = Phonopy(_to_phonopy_atoms(atoms), supercell_matrix=supercell_matrix)
    phonon.generate_displacements(distance=distance)
    return phonon


def displacement_atoms(phonon) -> List[Any]:
    """The displaced supercells to run VASP on, as ASE Atoms in phonopy order."""
    return [_to_ase_atoms(sc) for sc in phonon.supercells_with_displacements]


def set_forces_and_produce(phonon, force_sets: List[List[List[float]]]) -> None:
    """Attach per-displacement force arrays and build the force constants."""
    phonon.forces = force_sets
    phonon.produce_force_constants()


def band_structure(
    phonon, band_path: List[List[Tuple[float, float, float]]], labels: List[str],
    npoints: int = 51,
) -> Dict[str, Any]:
    """Compute the phonon dispersion (THz) along a q-path."""
    from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections

    qpoints, connections = get_band_qpoints_and_path_connections(band_path, npoints=npoints)
    phonon.run_band_structure(qpoints, path_connections=connections, labels=labels)
    bs = phonon.get_band_structure_dict()
    freqs = bs["frequencies"]  # list of (npoints, nbands) arrays per segment
    flat: List[float] = []
    for seg in freqs:
        for row in seg:
            flat.extend(float(f) for f in row)
    return {
        "distances": [[float(x) for x in seg] for seg in bs["distances"]],
        "frequencies_thz": [[[float(f) for f in row] for row in seg] for seg in freqs],
        "labels": labels,
        "min_frequency_thz": min(flat) if flat else None,
        "max_frequency_thz": max(flat) if flat else None,
        "all_frequencies_thz": flat,
    }


def total_dos(phonon, mesh: List[int]) -> Dict[str, Any]:
    """Phonon DOS on a q-mesh."""
    phonon.run_mesh(mesh)
    phonon.run_total_dos()
    dos = phonon.get_total_dos_dict()
    return {
        "frequency_points_thz": [float(x) for x in dos["frequency_points"]],
        "total_dos": [float(x) for x in dos["total_dos"]],
    }


def save(phonon, path: Path) -> None:
    """Persist force constants so a downstream node can reload them."""
    phonon.save(str(path), settings={"force_constants": True})


def load(path: Path):
    """Reload a saved Phonopy object (with force constants)."""
    _import_phonopy()
    from phonopy import load as phonopy_load

    return phonopy_load(str(path))
