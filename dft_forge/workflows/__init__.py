"""Deterministic scientific workflow macros.

The chat planner selects one of these high-level capabilities.  The macro
then builds a typed graph and performs all scientific bookkeeping in Python;
the LLM never expands a composite calculation into an ad-hoc executable DAG.

This module intentionally keeps the implementation conservative: missing QE
programs, missing outputs, incomplete competitor sets, and ambiguous
thermodynamic references fail closed or return an explicit partial status.
Charged-defect and adsorption reducers expose every supplied correction and
reference term rather than silently filling material observations.

Implementations live in focused submodules; this package re-exports the
public surface (and private helpers used by tests) for compatibility.
"""

from .adsorption import run_adsorption_scan
from .chemical_potential import chemical_potential_scan
from .common import (
    WorkflowError,
    _base_params,
    _charge_node_id,
    _charge_value,
    _finite_real,
    _graph_payload,
    _jsonable,
    _matching_atom_count,
    _normalise_kpoint,
    _positive_grid,
    _profile_common_method,
    _require_method_compatibility,
    _run_graph,
    _runtime_controls,
    _workflow_id,
    _workflow_workdir,
    _write_structure,
)
from .convergence import run_convergence
from .energetics import (
    adsorption_energy,
    charged_vacancy_formation_energy,
    convergence_grid,
    makov_payne_correction_ev,
    select_converged_point,
    vacancy_formation_energy,
)
from .magnetic import compare_magnetic_results, run_magnetic_compare
from .phonon import phonon_stability, run_phonon_dfpt
from .vacancy import run_charged_vacancy, run_neutral_vacancy

__all__ = [
    "WorkflowError", "convergence_grid", "select_converged_point",
    "vacancy_formation_energy", "makov_payne_correction_ev",
    "charged_vacancy_formation_energy", "adsorption_energy", "chemical_potential_scan",
    "compare_magnetic_results", "phonon_stability", "run_convergence",
    "run_neutral_vacancy", "run_charged_vacancy", "run_adsorption_scan",
    "run_magnetic_compare", "run_phonon_dfpt",
]
