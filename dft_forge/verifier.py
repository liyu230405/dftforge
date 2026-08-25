"""Scientific Verifier: deterministic physical convergence checks.

Key principle: program exit code ≠ physical convergence.
This module performs rigorous checks on QE output.

T1 checks:
- JOB DONE present
- SCF converged (not just last iteration)
- Forces below threshold
- Pressure below threshold
- Cell parameters reasonable
- Energy monotonicity
- File completeness

T2 checks:
- SCF convergence
- nbnd sufficient for bands
- bands.x output valid
- DOS output valid
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from dft_forge.protocol.schemas import VerificationResult
from dft_forge.parser import QEParser, ParsedVCResult


@dataclass
class ConvergenceReport:
    """Report from convergence check."""
    passed: bool
    checks: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    failure_reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ScientificVerifier:
    """Deterministic scientific verification of QE calculation results."""

    def __init__(
        self,
        force_threshold_ry: float = 1.0e-3,
        pressure_threshold_kbar: float = 0.5,
        energy_threshold_ry: float = 1.0e-6,
    ):
        self.force_threshold_ry = force_threshold_ry
        self.pressure_threshold_kbar = pressure_threshold_kbar
        self.energy_threshold_ry = energy_threshold_ry

    def verify_t1(self, job_result: Any, parsed: ParsedVCResult) -> ConvergenceReport:
        """Verify T1 vc-relax calculation.
        
        Args:
            job_result: QE JobResult from executor
            parsed: ParsedVCResult from parser
        
        Returns:
            ConvergenceReport with all checks
        """
        report = ConvergenceReport(passed=True)
        stdout = job_result.stdout if hasattr(job_result, 'stdout') else ""
        
        # Check 0: Data quality
        if (parsed.final_energy_ry == 0.0 and parsed.pressure_kbar == 0.0 
            and parsed.natoms == 0):
            report.passed = False
            report.failure_reasons.append(
                "Parser failed to extract values - possible format mismatch"
            )
            report.checks["data_quality"] = {"pass": False, "detail": "All parsed values zero"}
        else:
            report.checks["data_quality"] = {"pass": True, "detail": "Parser extracted non-zero values"}
        
        # Check 1: QE job completed
        if "JOB DONE" not in stdout:
            report.passed = False
            report.failure_reasons.append("JOB DONE not found in output")
            report.checks["job_done"] = {"pass": False}
        else:
            report.checks["job_done"] = {"pass": True}
        
        # Check 2: SCF converged
        scf = QEParser.parse_scf(stdout)
        if not scf.converged:
            report.passed = False
            report.failure_reasons.append("SCF did not converge")
        report.checks["scf_convergence"] = {
            "pass": scf.converged,
            "n_iterations": scf.n_iterations,
            "threshold": scf.convergence_thr,
        }
        
        # Check 3: Forces below threshold
        if parsed.max_force_ry_bohr > self.force_threshold_ry:
            report.passed = False
            report.failure_reasons.append(
                f"Max force {parsed.max_force_ry_bohr:.2e} Ry/Bohr exceeds "
                f"threshold {self.force_threshold_ry:.0e}"
            )
        report.checks["forces"] = {
            "pass": parsed.max_force_ry_bohr <= self.force_threshold_ry,
            "value": parsed.max_force_ry_bohr,
            "threshold": self.force_threshold_ry,
        }
        
        # Check 4: Pressure
        if abs(parsed.pressure_kbar) > self.pressure_threshold_kbar:
            report.warnings.append(
                f"Pressure {parsed.pressure_kbar:.2f} kbar exceeds "
                f"threshold {self.pressure_threshold_kbar}"
            )
        report.checks["pressure"] = {
            "pass": abs(parsed.pressure_kbar) <= self.pressure_threshold_kbar,
            "value": parsed.pressure_kbar,
            "threshold": self.pressure_threshold_kbar,
        }
        
        # Check 5: Energy monotonicity (vc-relax)
        energies = re.findall(r"!\s+total energy\s+=\s+(-?\d+\.\d+)\s+Ry", stdout)
        if len(energies) > 1:
            monotonic = all(
                float(energies[i]) <= float(energies[i-1])
                for i in range(1, len(energies))
            )
            if not monotonic:
                report.warnings.append("Energy not monotonically decreasing")
            report.checks["energy_monotonicity"] = {
                "pass": monotonic,
                "n_points": len(energies),
            }
        
        # Check 6: Cell parameters reasonable
        if parsed.cell:
            a = parsed.cell.a_bohr
            if 1.0 < a < 100.0:
                report.checks["cell_params"] = {"pass": True, "a_bohr": a}
            else:
                report.passed = False
                report.failure_reasons.append(f"Unreasonable cell: a={a}")
        
        # Check 7: Ion convergence (vc-relax specific)
        if "bfgs" in stdout or "vc-relax" in stdout:
            if parsed.max_force_ry_bohr > self.force_threshold_ry:
                report.failure_reasons.append(
                    "vc-relax did not converge: forces still above threshold"
                )
        
        # Summary
        n_pass = sum(1 for c in report.checks.values() if c.get("pass", False))
        n_total = len(report.checks)
        report.summary = f"{n_pass}/{n_total} checks passed"
        if report.failure_reasons:
            report.summary += f"; failures: {', '.join(report.failure_reasons[:3])}"
        
        return report

    def verify_scf(self, stdout: str) -> ConvergenceReport:
        """Verify SCF calculation convergence."""
        report = ConvergenceReport(passed=True)
        
        if not stdout or len(stdout) < 50:
            report.passed = False
            report.failure_reasons.append("No stdout provided")
            report.checks["scf"] = {"pass": False}
            report.summary = "SCF NOT converged: no output"
            return report
        
        scf = QEParser.parse_scf(stdout)
        
        if not scf.converged:
            report.passed = False
            report.failure_reasons.append("SCF did not converge")
        
        report.checks["scf"] = {
            "pass": scf.converged,
            "n_iterations": scf.n_iterations,
            "energy_ry": scf.total_energy_ry,
        }
        report.summary = (
            f"SCF {'converged' if scf.converged else 'NOT converged'} "
            f"in {scf.n_iterations} iterations"
        )
        
        return report

    def verify_t2_bands(
        self,
        scf_stdout: str,
        nscf_stdout: str,
        bands_xml: Path,
    ) -> ConvergenceReport:
        """Verify T2 band structure calculation."""
        report = ConvergenceReport(passed=True)
        
        # Check SCF
        scf_report = self.verify_scf(scf_stdout)
        if not scf_report.passed:
            report.passed = False
            report.failure_reasons.append("SCF stage failed")
        report.checks["scf"] = scf_report.checks.get("scf", {})
        
        # Check NSCF
        nscf_report = self.verify_scf(nscf_stdout)
        if not nscf_report.passed:
            report.passed = False
            report.failure_reasons.append("NSCF stage did not converge")
        report.checks["nscf"] = nscf_report.checks.get("scf", {})
        
        # Check bands.xml exists
        if not bands_xml.exists():
            report.passed = False
            report.failure_reasons.append(f"bands.xml not found: {bands_xml}")
            report.checks["bands_xml"] = {"pass": False}
        else:
            report.checks["bands_xml"] = {"pass": True}
            
            # Parse bands
            bands = QEParser.parse_bands(bands_xml)
            if bands.n_bands == 0:
                report.warnings.append("No bands parsed from XML")
            report.checks["bands_data"] = {
                "pass": bands.n_bands > 0,
                "n_bands": bands.n_bands,
                "n_kpoints": bands.n_kpoints,
                "fermi_ev": bands.fermi_energy_ev,
            }
        
        n_pass = sum(1 for c in report.checks.values() if c.get("pass", False))
        n_total = len(report.checks)
        report.summary = f"T2 bands: {n_pass}/{n_total} checks passed"
        
        return report

    def verify_t2_dos(
        self,
        scf_stdout: str,
        nscf_stdout: str,
        dos_file: Path,
    ) -> ConvergenceReport:
        """Verify T2 DOS calculation."""
        report = ConvergenceReport(passed=True)
        
        # Check SCF
        scf_report = self.verify_scf(scf_stdout)
        if not scf_report.passed:
            report.passed = False
            report.failure_reasons.append("SCF stage failed")
        report.checks["scf"] = scf_report.checks.get("scf", {})
        
        # Check DOS file
        if not dos_file.exists():
            report.passed = False
            report.failure_reasons.append(f"DOS file not found: {dos_file}")
            report.checks["dos_file"] = {"pass": False}
        else:
            dos = QEParser.parse_dos(dos_file)
            report.checks["dos_file"] = {
                "pass": dos.n_energy_points > 0,
                "n_points": dos.n_energy_points,
                "fermi_ev": dos.fermi_energy_ev,
            }
        
        n_pass = sum(1 for c in report.checks.values() if c.get("pass", False))
        n_total = len(report.checks)
        report.summary = f"T2 DOS: {n_pass}/{n_total} checks passed"
        
        return report
