"""Recovery controller: deterministic failure classification and bounded recovery actions.

The RecoveryController inspects a ConvergenceReport, classifies the failure
into a FailureKind, and produces bounded RecoveryAction objects.  All limits
are hard-coded to prevent runaway parameter growth.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from dft_forge.verifier import ConvergenceReport

logger = logging.getLogger(__name__)


class FailureKind(str, Enum):
    """Deterministic classification of a failed DFT run."""
    SCF_NOT_CONVERGED = "scf_not_converged"
    NSCF_NOT_CONVERGED = "nscf_not_converged"
    FORCES_TOO_HIGH = "forces_too_high"
    PRESSURE_TOO_HIGH = "pressure_too_high"
    BAD_CELL = "bad_cell"
    BANDS_XML_MISSING = "bands_xml_missing"
    DOS_FILE_MISSING = "dos_file_missing"
    TOOL_FAILURE = "tool_failure"
    JOB_TIMEOUT = "job_timeout"
    PARSER_MISMATCH = "parser_mismatch"
    UNKNOWN = "unknown"


@dataclass
class RecoveryAction:
    """A single bounded recovery action with optional parameter overrides."""
    action_type: str
    target: str
    params: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    applied: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type,
            "target": self.target,
            "params": self.params,
            "reason": self.reason,
            "applied": self.applied,
        }

    def __str__(self) -> str:
        if self.params:
            params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
            return f"{self.action_type}({params_str}): {self.reason}"
        return f"{self.action_type}: {self.reason}"


class RecoveryController:
    """Classify failures and generate bounded recovery actions.

    Usage:
        controller = RecoveryController()
        kind = controller.classify(report, task_type="T2")
        actions = controller.plan_actions(kind, baseline_params, attempt=1)
        retry_params = controller.apply_to_params(actions, baseline_params)
    """

    def __init__(
        self,
        max_ecut_scale: float = 3.0,
        max_nbnd_scale: float = 2.0,
        max_kpoints_scale: float = 2.0,
        max_attempts: int = 3,
    ):
        self.max_ecut_scale = max_ecut_scale
        self.max_nbnd_scale = max_nbnd_scale
        self.max_kpoints_scale = max_kpoints_scale
        self.max_attempts = max_attempts
        self._baseline: Dict[str, Any] = {}
        self._attempt: int = 0

    def classify(
        self,
        report: ConvergenceReport,
        task_type: str,
        parsed: Any = None,
    ) -> FailureKind:
        """Classify a failure from a ConvergenceReport."""
        reasons = [r.lower() for r in report.failure_reasons]

        if not report.passed and not reasons:
            return FailureKind.UNKNOWN

        if any("scf stage failed" in r or "scf did not converge" in r for r in reasons):
            return FailureKind.SCF_NOT_CONVERGED
        if any("nscf stage did not converge" in r for r in reasons):
            return FailureKind.NSCF_NOT_CONVERGED
        if any("force" in r for r in reasons):
            return FailureKind.FORCES_TOO_HIGH
        if any("pressure" in r for r in reasons):
            return FailureKind.PRESSURE_TOO_HIGH
        if any("cell" in r for r in reasons):
            return FailureKind.BAD_CELL
        if any("bands.xml not found" in r for r in reasons):
            return FailureKind.BANDS_XML_MISSING
        if any("dos file not found" in r for r in reasons):
            return FailureKind.DOS_FILE_MISSING
        if any("timed out" in r for r in reasons) or any("timeout" in r for r in reasons):
            return FailureKind.JOB_TIMEOUT
        if any("dos.x" in r or "bands.x" in r for r in reasons):
            return FailureKind.TOOL_FAILURE
        if any("parser failed" in r or "format mismatch" in r for r in reasons):
            return FailureKind.PARSER_MISMATCH
        return FailureKind.UNKNOWN

    def plan_actions(
        self,
        kind: FailureKind,
        baseline_params: Dict[str, Any],
        attempt: int,
    ) -> List[RecoveryAction]:
        """Produce bounded recovery actions for the given failure kind."""
        if attempt >= self.max_attempts:
            return [RecoveryAction(
                action_type="give_up",
                target="all",
                reason=f"Max recovery attempts ({self.max_attempts}) reached",
            )]

        self._baseline = dict(baseline_params)
        self._attempt = attempt
        actions: List[RecoveryAction] = []

        if kind == FailureKind.SCF_NOT_CONVERGED:
            actions.extend(self._plan_scf_failure())
        elif kind == FailureKind.NSCF_NOT_CONVERGED:
            actions.extend(self._plan_nscf_failure())
        elif kind == FailureKind.FORCES_TOO_HIGH:
            actions.extend(self._plan_forces_failure())
        elif kind == FailureKind.PRESSURE_TOO_HIGH:
            actions.extend(self._plan_pressure_failure())
        elif kind == FailureKind.BAD_CELL:
            actions.extend(self._plan_bad_cell())
        elif kind == FailureKind.BANDS_XML_MISSING:
            actions.extend(self._plan_bands_missing())
        elif kind == FailureKind.DOS_FILE_MISSING:
            actions.extend(self._plan_dos_missing())
        elif kind == FailureKind.TOOL_FAILURE:
            actions.extend(self._plan_tool_failure())
        elif kind == FailureKind.JOB_TIMEOUT:
            actions.extend(self._plan_timeout())
        elif kind == FailureKind.PARSER_MISMATCH:
            actions.extend(self._plan_parser_mismatch())
        else:
            actions.append(RecoveryAction(
                action_type="retry",
                target="all",
                reason="Unknown failure; retry with same parameters",
            ))

        return actions

    def apply_to_params(
        self,
        actions: List[RecoveryAction],
        baseline_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply recovery actions to produce new parameters for a retry."""
        params = dict(baseline_params)
        for action in actions:
            if not action.applied:
                params.update(action.params)
                action.applied = True
        return params

    # ── Private planners ─────────────────────────────────────────────────────

    def _plan_scf_failure(self) -> List[RecoveryAction]:
        actions: List[RecoveryAction] = []
        ecut = self._baseline.get("ecutwfc", 25.0)
        max_ecut = ecut * self.max_ecut_scale
        new_ecut = min(ecut * 1.5, max_ecut)
        if new_ecut > ecut:
            actions.append(RecoveryAction(
                action_type="increase_ecut",
                target="ecutwfc",
                params={"ecutwfc": round(new_ecut, 2)},
                reason=f"Increase ecutwfc from {ecut} to {round(new_ecut, 2)} Ry (max {round(max_ecut, 2)})",
            ))

        kp = self._baseline.get("kpoints", [4, 4, 4, 1, 1, 1])
        if isinstance(kp, list) and len(kp) >= 3:
            max_kp = int(kp[0] * self.max_kpoints_scale)
            new_kp = min(int(kp[0] * 2), max_kp)
            if new_kp > kp[0]:
                actions.append(RecoveryAction(
                    action_type="increase_kpoints",
                    target="kpoints",
                    params={"kpoints": [new_kp, new_kp, new_kp, 1, 1, 1]},
                    reason=f"Increase kpoints from {kp[:3]} to {[new_kp]*3} (max {max_kp})",
                ))

        actions.append(RecoveryAction(
            action_type="check_pseudos",
            target="pseudos",
            reason="Verify pseudopotentials are correct for the material",
        ))
        return actions

    def _plan_nscf_failure(self) -> List[RecoveryAction]:
        actions: List[RecoveryAction] = []
        nbnd = self._baseline.get("nbnd", 16)
        max_nbnd = int(nbnd * self.max_nbnd_scale)
        new_nbnd = min(int(nbnd * 1.5), max_nbnd)
        if new_nbnd > nbnd:
            actions.append(RecoveryAction(
                action_type="increase_nbnd",
                target="nbnd",
                params={"nbnd": new_nbnd},
                reason=f"Increase nbnd from {nbnd} to {new_nbnd} (max {max_nbnd})",
            ))

        kp = self._baseline.get("kpoints_nscf", [8, 8, 8, 1, 1, 1])
        if isinstance(kp, list) and len(kp) >= 3:
            max_kp = int(kp[0] * self.max_kpoints_scale)
            new_kp = min(int(kp[0] * 2), max_kp)
            if new_kp > kp[0]:
                actions.append(RecoveryAction(
                    action_type="increase_kpoints",
                    target="kpoints_nscf",
                    params={"kpoints_nscf": [new_kp, new_kp, new_kp, 1, 1, 1]},
                    reason=f"Increase kpoints from {kp[:3]} to {[new_kp]*3} (max {max_kp})",
                ))
        return actions

    def _plan_forces_failure(self) -> List[RecoveryAction]:
        actions: List[RecoveryAction] = []
        ecut = self._baseline.get("ecutwfc", 25.0)
        max_ecut = ecut * self.max_ecut_scale
        new_ecut = min(ecut * 1.5, max_ecut)
        if new_ecut > ecut:
            actions.append(RecoveryAction(
                action_type="increase_ecut",
                target="ecutwfc",
                params={"ecutwfc": round(new_ecut, 2)},
                reason=f"Increase ecutwfc from {ecut} to {round(new_ecut, 2)} Ry (max {round(max_ecut, 2)})",
            ))
        actions.append(RecoveryAction(
            action_type="tighten_conv_thr",
            target="conv_thr",
            params={"conv_thr": 1.0e-9},
            reason="Tighten SCF convergence threshold to 1e-9",
        ))
        return actions

    def _plan_pressure_failure(self) -> List[RecoveryAction]:
        return [
            RecoveryAction(
                action_type="check_structure",
                target="structure",
                reason="High pressure: verify initial structure and cell constraints",
            )
        ]

    def _plan_bad_cell(self) -> List[RecoveryAction]:
        return [
            RecoveryAction(
                action_type="check_structure",
                target="structure",
                reason="Unreasonable cell parameters: check initial lattice constant",
            )
        ]

    def _plan_bands_missing(self) -> List[RecoveryAction]:
        return [
            RecoveryAction(
                action_type="check_nscf_output",
                target="nscf",
                reason="bands.xml missing: verify NSCF completed and prefix matches",
            )
        ]

    def _plan_dos_missing(self) -> List[RecoveryAction]:
        return [
            RecoveryAction(
                action_type="check_nscf_output",
                target="nscf",
                reason="DOS file missing: verify NSCF output completeness and prefix matches",
            )
        ]

    def _plan_tool_failure(self) -> List[RecoveryAction]:
        return [
            RecoveryAction(
                action_type="retry_tool",
                target="tool",
                reason="Post-processing tool failed; retry execution",
            )
        ]

    def _plan_timeout(self) -> List[RecoveryAction]:
        return [
            RecoveryAction(
                action_type="reduce_kpoints",
                target="kpoints",
                params={"kpoints": [4, 4, 4, 1, 1, 1]},
                reason="Job timed out: reduce k-point density to lower cost",
            )
        ]

    def _plan_parser_mismatch(self) -> List[RecoveryAction]:
        return [
            RecoveryAction(
                action_type="retry",
                target="all",
                reason="Parser mismatch; retry to capture fresh output",
            )
        ]
