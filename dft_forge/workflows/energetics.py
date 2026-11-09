"""Pure energy-reduction and convergence-selection helpers.

Every function in this module is a deterministic reducer over supplied
evidence; no graph execution happens here.
"""

from __future__ import annotations

import math
import numbers
from typing import Any, Dict, List, Sequence

from dft_forge.parser import RY_TO_EV

from .common import _MAX_CONVERGENCE_JOBS, WorkflowError, _normalise_kpoint, _positive_grid


def convergence_grid(
    *,
    ecutwfc_values: Any = None,
    kpoints_values: Any = None,
    smearing_values: Any = None,
    degauss_values: Any = None,
    max_jobs: int = 64,
) -> List[Dict[str, Any]]:
    """Build a reproducible Cartesian parameter grid with a hard job cap."""
    def finite_positive(values: Any, name: str, default: Sequence[Any]) -> List[float]:
        converted: List[float] = []
        for raw in _positive_grid(values, name, default):
            if isinstance(raw, bool):
                raise WorkflowError(f"{name} values must be finite and positive")
            try:
                value = float(raw)
            except (TypeError, ValueError) as exc:
                raise WorkflowError(f"{name} values must be finite and positive") from exc
            if not math.isfinite(value) or value <= 0:
                raise WorkflowError(f"{name} values must be finite and positive")
            converted.append(value)
        return converted

    ecut = finite_positive(ecutwfc_values, "ecutwfc_values", [30, 40, 50, 60])
    kpts = [_normalise_kpoint(x) for x in _positive_grid(kpoints_values, "kpoints_values", [4, 6, 8])]
    smearing = [str(x) for x in _positive_grid(smearing_values, "smearing_values", ["mv"])]
    if any(not value.strip() for value in smearing):
        raise WorkflowError("smearing values must be non-empty")
    degauss = finite_positive(degauss_values, "degauss_values", [0.01])
    if isinstance(max_jobs, bool) or not isinstance(max_jobs, numbers.Integral):
        raise WorkflowError("max_jobs must be a positive integer")
    max_jobs = int(max_jobs)
    if max_jobs < 1 or max_jobs > _MAX_CONVERGENCE_JOBS:
        raise WorkflowError(
            f"max_jobs must be between 1 and {_MAX_CONVERGENCE_JOBS}"
        )
    n_points = len(ecut) * len(kpts) * len(smearing) * len(degauss)
    if n_points > max_jobs:
        raise WorkflowError(
            f"convergence grid has {n_points} jobs, above max_jobs={max_jobs}; "
            "reduce the grid explicitly"
        )
    points = [
        {"ecutwfc": e, "kpoints": k, "smearing": s, "degauss": d}
        for e in ecut for k in kpts for s in smearing for d in degauss
    ]
    return points


def select_converged_point(
    results: Sequence[Dict[str, Any]],
    *,
    threshold_mev_per_atom: float = 1.0,
) -> Dict[str, Any]:
    """Select the least-cost point converged to the densest/highest reference.

    Every successful point is compared with the highest cutoff + densest
    k-grid point for the same smearing/degauss pair.  A point is eligible only
    when that reference exists and its per-atom energy difference is within the
    requested tolerance.
    """
    if threshold_mev_per_atom <= 0:
        raise WorkflowError("threshold_mev_per_atom must be positive")
    good = [
        r for r in results
        if r.get("state") in (None, "succeeded", "success")
        and isinstance(r.get("energy_ry"), (int, float))
        and isinstance(r.get("natoms"), (int, float))
        and r.get("natoms", 0) > 0
    ]
    if not good:
        raise WorkflowError("no successful SCF points with energy and natoms evidence")

    def key(r: Dict[str, Any]) -> Tuple[float, int, int]:
        kp = r.get("kpoints") or [1, 1, 1]
        return float(r.get("ecutwfc") or 0), sum(int(x) for x in kp[:3]), len(kp)

    enriched = []
    groups: Dict[Tuple[str, float], List[Dict[str, Any]]] = {}
    for r in good:
        group = (str(r.get("smearing", "")), float(r.get("degauss", 0.0) or 0.0))
        groups.setdefault(group, []).append(r)
    references = {group: max(items, key=key) for group, items in groups.items()}
    for r in good:
        group = (str(r.get("smearing", "")), float(r.get("degauss", 0.0) or 0.0))
        reference = references[group]
        ref_per_atom = float(reference["energy_ry"]) / float(reference["natoms"])
        per_atom = float(r["energy_ry"]) / float(r["natoms"])
        delta = abs(per_atom - ref_per_atom) * RY_TO_EV * 1000.0
        kp = r.get("kpoints") or [1, 1, 1]
        cost = float(r.get("ecutwfc") or 0) * math.prod(int(x) for x in kp[:3])
        enriched.append({**r, "delta_mev_per_atom": delta, "cost_proxy": cost})
    eligible = [r for r in enriched if r["delta_mev_per_atom"] <= threshold_mev_per_atom]
    if not eligible:
        raise WorkflowError(
            f"no point converged to {threshold_mev_per_atom:g} meV/atom against the reference"
        )
    chosen = min(eligible, key=lambda r: (r["cost_proxy"], float(r.get("ecutwfc") or 0)))
    return {
        "selected": chosen,
        "reference": max(references.values(), key=key),
        "references_by_condition": [
            {"smearing": group[0], "degauss": group[1], "point": ref}
            for group, ref in references.items()
        ],
        "threshold_mev_per_atom": float(threshold_mev_per_atom),
        "points": enriched,
    }


def vacancy_formation_energy(
    *,
    defect_energy_ry: float,
    pristine_energy_ry: float,
    chemical_potential_ry: float,
    removed_atoms: int = 1,
) -> Dict[str, float]:
    """Neutral vacancy formation energy using a per-atom chemical potential.

    For a vacancy, ``E_f = E_def - E_bulk + n_removed * mu_removed``.
    The caller must provide ``mu_removed`` explicitly (for example half the
    total O2 energy); this function never guesses a reference state.
    """
    if removed_atoms < 1:
        raise WorkflowError("removed_atoms must be >= 1")
    ry = float(defect_energy_ry) - float(pristine_energy_ry) + removed_atoms * float(chemical_potential_ry)
    return {"formation_energy_ry": ry, "formation_energy_ev": ry * RY_TO_EV}


def makov_payne_correction_ev(
    cell: Any,
    *,
    charge: int,
    dielectric_constant: float,
    madelung_alpha: float = 2.837297,
    cubic_tolerance: float = 0.02,
) -> Dict[str, float]:
    """First-order Makov-Payne image-charge correction for a cubic cell.

    ``charge`` follows the QE convention: positive values remove electrons.
    The formula is ``q² α e² / (2 ε L)`` in eV.  Non-cubic cells are rejected
    because substituting ``V**(1/3)`` would hide a material approximation.
    """
    if isinstance(charge, bool) or not isinstance(charge, numbers.Integral):
        raise WorkflowError("charge must be an integer")
    try:
        epsilon = float(dielectric_constant)
        alpha = float(madelung_alpha)
        tolerance = float(cubic_tolerance)
    except (TypeError, ValueError) as exc:
        raise WorkflowError(
            "dielectric_constant, madelung_alpha and cubic_tolerance must be finite numbers"
        ) from exc
    if not all(math.isfinite(value) for value in (epsilon, alpha, tolerance)):
        raise WorkflowError("Makov-Payne inputs must be finite")
    if epsilon <= 0 or alpha <= 0 or not 0 < tolerance <= 0.1:
        raise WorkflowError("dielectric_constant, madelung_alpha and cubic_tolerance must be positive")
    lengths = [float(value) for value in cell.lengths()]
    angles = [float(value) for value in cell.angles()]
    mean_length = sum(lengths) / 3.0
    if mean_length <= 0:
        raise WorkflowError("defect cell has no finite length")
    if (max(abs(value - mean_length) for value in lengths) > tolerance * mean_length
            or max(abs(value - 90.0) for value in angles) > max(0.5, 90.0 * tolerance)):
        raise WorkflowError(
            "makov_payne_cubic requires a near-cubic cell; provide an explicit electrostatic correction for this cell"
        )
    coulomb_ev_angstrom = 14.3996454784255
    correction = int(charge) ** 2 * alpha * coulomb_ev_angstrom / (2.0 * epsilon * mean_length)
    return {
        "electrostatic_correction_ev": correction,
        "charge": int(charge),
        "dielectric_constant": epsilon,
        "madelung_alpha": alpha,
        "cell_length_angstrom": mean_length,
        "cubic_tolerance": tolerance,
    }


def charged_vacancy_formation_energy(
    *,
    defect_energy_ry: float,
    pristine_energy_ry: float,
    chemical_potential_ry: float,
    charge: int,
    fermi_level_from_vbm_ev: float,
    potential_alignment_ev: float,
    electrostatic_correction_ev: float,
) -> Dict[str, float]:
    """Charged vacancy formation energy with explicit reference terms.

    ``E_f = E_def - E_bulk + mu + q(E_F + ΔV) + E_corr``.  ``E_F`` is
    measured from the VBM; ``ΔV`` and ``E_corr`` must be supplied or derived
    by the workflow and are never guessed here.
    """
    values = {
        "defect_energy_ry": defect_energy_ry,
        "pristine_energy_ry": pristine_energy_ry,
        "chemical_potential_ry": chemical_potential_ry,
        "fermi_level_from_vbm_ev": fermi_level_from_vbm_ev,
        "potential_alignment_ev": potential_alignment_ev,
        "electrostatic_correction_ev": electrostatic_correction_ev,
    }
    if isinstance(charge, bool) or not isinstance(charge, numbers.Integral):
        raise WorkflowError("charge must be an integer")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
           for value in values.values()):
        raise WorkflowError("charged-defect formation inputs must be finite numbers")
    q = int(charge)
    raw_ev = (float(defect_energy_ry) - float(pristine_energy_ry)
              + float(chemical_potential_ry)) * RY_TO_EV
    fermi_term = q * float(fermi_level_from_vbm_ev)
    alignment_term = q * float(potential_alignment_ev)
    formation = raw_ev + fermi_term + alignment_term + float(electrostatic_correction_ev)
    return {
        "charge": q,
        "formation_energy_ev": formation,
        "raw_energy_and_mu_ev": raw_ev,
        "fermi_level_from_vbm_ev": float(fermi_level_from_vbm_ev),
        "fermi_term_ev": fermi_term,
        "potential_alignment_ev": float(potential_alignment_ev),
        "potential_alignment_term_ev": alignment_term,
        "electrostatic_correction_ev": float(electrostatic_correction_ev),
    }


def adsorption_energy(
    *, adsorbed_energy_ry: float, surface_energy_ry: float, molecule_energy_ry: float,
    adsorbate_atoms: int, reference_atoms: int,
) -> Dict[str, float]:
    if adsorbate_atoms < 1 or reference_atoms < 1:
        raise WorkflowError("adsorption atom counts must be positive")
    ratio = float(adsorbate_atoms) / float(reference_atoms)
    value_ry = float(adsorbed_energy_ry) - float(surface_energy_ry) - ratio * float(molecule_energy_ry)
    return {
        "e_ads_ry": value_ry,
        "e_ads_ev": value_ry * RY_TO_EV,
        "e_ads_ev_per_adsorbate": value_ry * RY_TO_EV / adsorbate_atoms,
        "molecule_ratio": ratio,
    }
