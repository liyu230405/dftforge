"""Deterministic task runner: compiles input, executes QE, parses output, verifies.

This is the core Phase 1/2 pipeline with NO LLM involvement for the
deterministic core.  Execution is delegated to pluggable backends via
the ``Executor`` interface.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dft_forge.catalog import get_material_profile, get_task_spec
from dft_forge.compiler import QECompiler, MATERIAL_DB
from dft_forge.executor import (
    Executor,
    FakeExecutor,
    JobHandle,
    JobResult,
    JobStatus,
    LocalExecutor,
    QEExecutor,
)
from dft_forge.parser import QEParser, ParsedVCResult, ParsedBands, ParsedDOS
from dft_forge.protocol.schemas import EvidenceBundle, TaskSpec
from dft_forge.recovery import RecoveryController
from dft_forge.verifier import ScientificVerifier, ConvergenceReport


@dataclass
class RunRecord:
    """Record of a single task execution attempt."""
    attempt: int
    success: bool
    input_hash: str
    walltime_sec: float
    verifier_passed: bool
    failure_reasons: List[str] = field(default_factory=list)
    recovery_actions: List[str] = field(default_factory=list)


class TaskRunner:
    """Deterministic task runner.

    Pipeline:
    1. Load TaskSpec
    2. Compile QE input (deterministic)
    3. Execute QE via pluggable Executor
    4. Parse output (deterministic)
    5. Verify convergence (deterministic)
    6. Build evidence bundle
    """

    def __init__(
        self,
        pseudo_dir: Path,
        executor: Optional[Executor] = None,
        max_walltime_sec: int = 600,
        max_retries: int = 2,
        ledger: Optional["EvidenceLedger"] = None,
    ):
        self.pseudo_dir = Path(pseudo_dir)
        self.max_walltime_sec = max_walltime_sec
        self.max_retries = max_retries
        self.compiler = QECompiler(pseudo_dir=self.pseudo_dir)
        self.executor = executor or LocalExecutor(
            qe_bin_dir=Path("/opt/homebrew/bin"),
            max_walltime_sec=self.max_walltime_sec,
            max_retries=self.max_retries,
        )
        self.verifier = ScientificVerifier()
        self.run_records: List[RunRecord] = []
        self.recovery_controller = RecoveryController()
        self.ledger = ledger

    def run_t1(
        self,
        task_id: str,
        workdir: Path,
        *,
        retry_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run a T1 vc-relax task."""
        task_spec = get_task_spec(task_id)
        material = task_spec.material
        db = MATERIAL_DB[material]

        workdir = Path(workdir).resolve()
        workdir.mkdir(parents=True, exist_ok=True)

        input_file = workdir / f"{material}_vcrelax.in"
        params = dict(task_spec.parameters)
        if retry_params:
            params.update(retry_params)

        input_content = self.compiler.compile_t1(
            material,
            input_file,
            ecutwfc=params.get("ecutwfc"),
            ecutrho=params.get("ecutrho"),
            kpoints=tuple(params.get("kpoints", [4, 4, 4, 1, 1, 1])),
            conv_thr=params.get("conv_thr", 1.0e-8),
            nstep=params.get("nstep", 200),
            cell_optimization=params.get("cell_optimization", True),
        )
        input_hash = self.compiler.compute_input_hash(input_content)

        job_result = self._execute_pw(input_file, workdir)

        xml_path = workdir / f"{material}_vcrelax.save" / "data-file-schema.xml"
        parsed = QEParser.parse_vc_relax(job_result.stdout, xml_path=xml_path if xml_path.exists() else None)
        report = self.verifier.verify_t1(job_result, parsed)

        kind = self.recovery_controller.classify(report, task_spec.task_type, parsed)
        actions = self.recovery_controller.plan_actions(kind, task_spec.parameters, len(self.run_records) + 1)
        recovery_strings = [str(a) for a in actions]

        record = RunRecord(
            attempt=len(self.run_records) + 1,
            success=report.passed,
            input_hash=input_hash,
            walltime_sec=job_result.walltime_sec,
            verifier_passed=report.passed,
            failure_reasons=report.failure_reasons,
            recovery_actions=recovery_strings,
        )
        self.run_records.append(record)

        evidence = self._build_t1_evidence(task_spec, job_result, parsed, report, input_hash, db, recovery_strings)
        evidence_path = workdir / "evidence.json"
        evidence.to_json(evidence_path)

        result = self._build_t1_result(task_spec, job_result, parsed, report, evidence, input_file, input_hash, workdir)
        result_path = workdir / "result.json"
        result_path.write_text(json.dumps(result, indent=2, default=str))

        if self.ledger:
            self.ledger.record_run(
                record,
                task_id=task_spec.task_id,
                task_type=task_spec.task_type,
                failure_kind=kind.value,
                evidence_path=str(evidence_path),
                output_files=job_result.output_files,
            )

        return result

    def run_t2_bands(
        self,
        task_id: str,
        workdir: Path,
        *,
        retry_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run a T2 band structure task: SCF → NSCF → bands.x."""
        task_spec = get_task_spec(task_id)
        material = task_spec.material
        db = MATERIAL_DB[material]
        params = dict(task_spec.parameters)
        if retry_params:
            params.update(retry_params)

        workdir = Path(workdir).resolve()
        workdir.mkdir(parents=True, exist_ok=True)

        prefix = material.lower()
        all_walltimes: List[float] = []
        all_commands: List[str] = []
        all_output_files: List[str] = []
        all_errors: List[str] = []

        scf_input = workdir / f"{prefix}_scf.in"
        scf_content = self.compiler.compile_scf(
            material,
            scf_input,
            calculation="scf",
            ecutwfc=params.get("ecutwfc"),
            ecutrho=params.get("ecutrho"),
            kpoints=tuple(params.get("kpoints_scf", [8, 8, 8, 1, 1, 1])),
            conv_thr=params.get("conv_thr", 1.0e-8),
            nbnd=params.get("nbnd"),
        )
        scf_hash = self.compiler.compute_input_hash(scf_content)
        scf_result = self._execute_pw(scf_input, workdir)
        all_walltimes.append(scf_result.walltime_sec)
        all_commands.append(f"pw.x -in {scf_input.name}")
        all_output_files.extend(scf_result.output_files)
        if scf_result.error_message:
            all_errors.append(f"SCF: {scf_result.error_message}")

        scf_parsed = QEParser.parse_scf(scf_result.stdout)
        scf_report = self.verifier.verify_scf(scf_result.stdout)

        nscf_input = workdir / f"{prefix}_nscf_bands.in"
        nscf_content = self.compiler.compile_scf(
            material,
            nscf_input,
            calculation="nscf",
            ecutwfc=params.get("ecutwfc"),
            ecutrho=params.get("ecutrho"),
            kpoints=tuple(params.get("kpoints_nscf", [8, 8, 8, 1, 1, 1])),
            conv_thr=params.get("conv_thr", 1.0e-8),
            nbnd=params.get("nbnd"),
        )
        nscf_hash = self.compiler.compute_input_hash(nscf_content)
        nscf_result = self._execute_pw(nscf_input, workdir)
        all_walltimes.append(nscf_result.walltime_sec)
        all_commands.append(f"pw.x -in {nscf_input.name}")
        all_output_files.extend(nscf_result.output_files)
        if nscf_result.error_message:
            all_errors.append(f"NSCF: {nscf_result.error_message}")

        nscf_parsed = QEParser.parse_scf(nscf_result.stdout)
        nscf_report = self.verifier.verify_scf(nscf_result.stdout)

        bands_input = workdir / f"{prefix}_bands.in"
        bands_content = self.compiler.compile_bands_input(
            material,
            bands_input,
            prefix=prefix,
            nkpoints=params.get("nkpoints_bands", 100),
        )
        bands_hash = self.compiler.compute_input_hash(bands_content)
        bands_result = self._execute_tool("bands.x", bands_input, workdir)
        all_walltimes.append(bands_result.walltime_sec)
        all_commands.append(f"bands.x -in {bands_input.name}")
        all_output_files.extend(bands_result.output_files)
        if bands_result.error_message:
            all_errors.append(f"bands.x: {bands_result.error_message}")

        bands_xml = workdir / f"{prefix}_bands.xml"
        bands_parsed = QEParser.parse_bands(bands_xml)

        report = self.verifier.verify_t2_bands(
            scf_result.stdout,
            nscf_result.stdout,
            bands_xml,
        )

        kind = self.recovery_controller.classify(report, task_spec.task_type)
        actions = self.recovery_controller.plan_actions(kind, task_spec.parameters, len(self.run_records) + 1)
        recovery_actions = [str(a) for a in actions]

        record = RunRecord(
            attempt=len(self.run_records) + 1,
            success=report.passed,
            input_hash=f"{scf_hash}:{nscf_hash}:{bands_hash}",
            walltime_sec=sum(all_walltimes),
            verifier_passed=report.passed,
            failure_reasons=report.failure_reasons,
            recovery_actions=recovery_actions,
        )
        self.run_records.append(record)

        pseudo_hashes = {}
        for elem, pseudo_file in db["pseudos"].items():
            pseudo_path = self.pseudo_dir / pseudo_file
            if pseudo_path.exists():
                pseudo_hashes[elem] = hashlib.sha256(pseudo_path.read_bytes()).hexdigest()[:16]

        evidence = EvidenceBundle(
            task_id=task_spec.task_id,
            task_type=task_spec.task_type,
            status="pass" if report.passed else "fail",
            input_hash=f"{scf_hash}:{nscf_hash}:{bands_hash}",
            pseudo_hashes=pseudo_hashes,
            qe_version="7.5",
            parser_version="0.1.0",
            commands=all_commands,
            walltimes_sec=all_walltimes,
            total_walltime_sec=sum(all_walltimes),
            tokens_used=0,
            errors=all_errors,
            recovery_actions=recovery_actions,
            verifier_verdict=report.to_dict(),
            physical_results={
                "band_gap_ev": bands_parsed.band_gap_ev,
                "fermi_energy_ev": bands_parsed.fermi_energy_ev,
                "n_bands": bands_parsed.n_bands,
                "n_kpoints": bands_parsed.n_kpoints,
                "is_metal": bands_parsed.is_metal,
                "scf_energy_ry": scf_parsed.total_energy_ry,
                "nscf_energy_ry": nscf_parsed.total_energy_ry,
            },
            output_files=all_output_files,
            log_files=[
                str(scf_input.parent / f"{scf_input.stem}.out"),
                str(nscf_input.parent / f"{nscf_input.stem}.out"),
                str(bands_input.parent / f"{bands_input.stem}.out"),
            ],
        )
        evidence_path = workdir / "evidence.json"
        evidence.to_json(evidence_path)

        physical_results = {
            "band_gap_ev": bands_parsed.band_gap_ev,
            "fermi_energy_ev": bands_parsed.fermi_energy_ev,
            "n_bands": bands_parsed.n_bands,
            "n_kpoints": bands_parsed.n_kpoints,
            "is_metal": bands_parsed.is_metal,
            "scf_energy_ry": scf_parsed.total_energy_ry,
            "nscf_energy_ry": nscf_parsed.total_energy_ry,
        }
        if bands_parsed.eigenvalues is not None:
            physical_results["eigenvalues_shape"] = list(bands_parsed.eigenvalues.shape)

        result = {
            "task_id": task_spec.task_id,
            "task_type": task_spec.task_type,
            "subtype": "bands",
            "status": "pass" if report.passed else "fail",
            "input_hash": f"{scf_hash}:{nscf_hash}:{bands_hash}",
            "verifier_summary": report.summary,
            "verifier_passed": report.passed,
            "checks": report.checks,
            "failure_reasons": report.failure_reasons,
            "warnings": report.warnings,
            "physical_results": physical_results,
            "evidence_path": str(evidence_path),
            "output_files": all_output_files,
            "walltime_sec": round(sum(all_walltimes), 3),
            "qe_calls": 3,
            "input_files": {
                "scf": str(scf_input),
                "nscf": str(nscf_input),
                "bands": str(bands_input),
            },
            "stdout_files": {
                "scf": str(workdir / f"{scf_input.stem}.out"),
                "nscf": str(workdir / f"{nscf_input.stem}.out"),
                "bands": str(workdir / f"{bands_input.stem}.out"),
            },
        }
        result_path = workdir / "result.json"
        result_path.write_text(json.dumps(result, indent=2, default=str))

        if self.ledger:
            self.ledger.record_run(
                record,
                task_id=task_spec.task_id,
                task_type=task_spec.task_type,
                failure_kind=kind.value,
                evidence_path=str(evidence_path),
                output_files=all_output_files,
            )

        return result

    def run_t2_dos(
        self,
        task_id: str,
        workdir: Path,
        *,
        retry_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run a T2 DOS task: SCF → NSCF → dos.x."""
        task_spec = get_task_spec(task_id)
        material = task_spec.material
        db = MATERIAL_DB[material]
        params = dict(task_spec.parameters)
        if retry_params:
            params.update(retry_params)

        workdir = Path(workdir).resolve()
        workdir.mkdir(parents=True, exist_ok=True)

        prefix = material.lower()
        all_walltimes: List[float] = []
        all_commands: List[str] = []
        all_output_files: List[str] = []
        all_errors: List[str] = []

        scf_input = workdir / f"{prefix}_scf.in"
        scf_content = self.compiler.compile_scf(
            material,
            scf_input,
            calculation="scf",
            ecutwfc=params.get("ecutwfc"),
            ecutrho=params.get("ecutrho"),
            kpoints=tuple(params.get("kpoints_scf", [8, 8, 8, 1, 1, 1])),
            conv_thr=params.get("conv_thr", 1.0e-8),
            nbnd=params.get("nbnd"),
        )
        scf_hash = self.compiler.compute_input_hash(scf_content)
        scf_result = self._execute_pw(scf_input, workdir)
        all_walltimes.append(scf_result.walltime_sec)
        all_commands.append(f"pw.x -in {scf_input.name}")
        all_output_files.extend(scf_result.output_files)
        if scf_result.error_message:
            all_errors.append(f"SCF: {scf_result.error_message}")

        scf_parsed = QEParser.parse_scf(scf_result.stdout)
        scf_report = self.verifier.verify_scf(scf_result.stdout)

        nscf_input = workdir / f"{prefix}_nscf_dos.in"
        nscf_content = self.compiler.compile_scf(
            material,
            nscf_input,
            calculation="nscf",
            ecutwfc=params.get("ecutwfc"),
            ecutrho=params.get("ecutrho"),
            kpoints=tuple(params.get("kpoints_nscf", [8, 8, 8, 1, 1, 1])),
            conv_thr=params.get("conv_thr", 1.0e-8),
            nbnd=params.get("nbnd"),
        )
        nscf_hash = self.compiler.compute_input_hash(nscf_content)
        nscf_result = self._execute_pw(nscf_input, workdir)
        all_walltimes.append(nscf_result.walltime_sec)
        all_commands.append(f"pw.x -in {nscf_input.name}")
        all_output_files.extend(nscf_result.output_files)
        if nscf_result.error_message:
            all_errors.append(f"NSCF: {nscf_result.error_message}")

        nscf_parsed = QEParser.parse_scf(nscf_result.stdout)
        nscf_report = self.verifier.verify_scf(nscf_result.stdout)

        dos_input = workdir / f"{prefix}_dos.in"
        dos_content = self.compiler.compile_dos_input(
            material,
            dos_input,
            prefix=prefix,
            deltae=params.get("dos_deltae", 0.01),
            fwhm=params.get("dos_fwhm", 0.05),
        )
        dos_hash = self.compiler.compute_input_hash(dos_content)
        dos_result = self._execute_tool("dos.x", dos_input, workdir)
        all_walltimes.append(dos_result.walltime_sec)
        all_commands.append(f"dos.x -in {dos_input.name}")
        all_output_files.extend(dos_result.output_files)
        if dos_result.error_message:
            all_errors.append(f"dos.x: {dos_result.error_message}")

        dos_file = workdir / f"{prefix}_dos.dat"
        dos_parsed = QEParser.parse_dos(dos_file)

        report = self.verifier.verify_t2_dos(
            scf_result.stdout,
            nscf_result.stdout,
            dos_file,
        )

        kind = self.recovery_controller.classify(report, task_spec.task_type)
        actions = self.recovery_controller.plan_actions(kind, task_spec.parameters, len(self.run_records) + 1)
        recovery_actions = [str(a) for a in actions]

        record = RunRecord(
            attempt=len(self.run_records) + 1,
            success=report.passed,
            input_hash=f"{scf_hash}:{nscf_hash}:{dos_hash}",
            walltime_sec=sum(all_walltimes),
            verifier_passed=report.passed,
            failure_reasons=report.failure_reasons,
            recovery_actions=recovery_actions,
        )
        self.run_records.append(record)

        pseudo_hashes = {}
        for elem, pseudo_file in db["pseudos"].items():
            pseudo_path = self.pseudo_dir / pseudo_file
            if pseudo_path.exists():
                pseudo_hashes[elem] = hashlib.sha256(pseudo_path.read_bytes()).hexdigest()[:16]

        evidence = EvidenceBundle(
            task_id=task_spec.task_id,
            task_type=task_spec.task_type,
            status="pass" if report.passed else "fail",
            input_hash=f"{scf_hash}:{nscf_hash}:{dos_hash}",
            pseudo_hashes=pseudo_hashes,
            qe_version="7.5",
            parser_version="0.1.0",
            commands=all_commands,
            walltimes_sec=all_walltimes,
            total_walltime_sec=sum(all_walltimes),
            tokens_used=0,
            errors=all_errors,
            recovery_actions=recovery_actions,
            verifier_verdict=report.to_dict(),
            physical_results={
                "dos_at_fermi": dos_parsed.dos_at_fermi,
                "fermi_energy_ev": dos_parsed.fermi_energy_ev,
                "n_energy_points": dos_parsed.n_energy_points,
                "scf_energy_ry": scf_parsed.total_energy_ry,
                "nscf_energy_ry": nscf_parsed.total_energy_ry,
            },
            output_files=all_output_files,
            log_files=[
                str(scf_input.parent / f"{scf_input.stem}.out"),
                str(nscf_input.parent / f"{nscf_input.stem}.out"),
                str(dos_input.parent / f"{dos_input.stem}.out"),
            ],
        )
        evidence_path = workdir / "evidence.json"
        evidence.to_json(evidence_path)

        physical_results = {
            "dos_at_fermi": dos_parsed.dos_at_fermi,
            "fermi_energy_ev": dos_parsed.fermi_energy_ev,
            "n_energy_points": dos_parsed.n_energy_points,
            "scf_energy_ry": scf_parsed.total_energy_ry,
            "nscf_energy_ry": nscf_parsed.total_energy_ry,
            "scf_n_iterations": scf_parsed.n_iterations,
            "nscf_n_iterations": nscf_parsed.n_iterations,
        }
        if dos_parsed.energies is not None and dos_parsed.dos is not None:
            physical_results["energy_range_ev"] = [
                float(dos_parsed.energies[0]),
                float(dos_parsed.energies[-1]),
            ]
            physical_results["dos_shape"] = list(dos_parsed.dos.shape)

        result = {
            "task_id": task_spec.task_id,
            "task_type": task_spec.task_type,
            "subtype": "dos",
            "status": "pass" if report.passed else "fail",
            "input_hash": f"{scf_hash}:{nscf_hash}:{dos_hash}",
            "verifier_summary": report.summary,
            "verifier_passed": report.passed,
            "checks": report.checks,
            "failure_reasons": report.failure_reasons,
            "warnings": report.warnings,
            "physical_results": physical_results,
            "evidence_path": str(evidence_path),
            "output_files": all_output_files,
            "walltime_sec": round(sum(all_walltimes), 3),
            "qe_calls": 3,
            "input_files": {
                "scf": str(scf_input),
                "nscf": str(nscf_input),
                "dos": str(dos_input),
            },
            "stdout_files": {
                "scf": str(workdir / f"{scf_input.stem}.out"),
                "nscf": str(workdir / f"{nscf_input.stem}.out"),
                "dos": str(workdir / f"{dos_input.stem}.out"),
            },
        }
        result_path = workdir / "result.json"
        result_path.write_text(json.dumps(result, indent=2, default=str))

        if self.ledger:
            self.ledger.record_run(
                record,
                task_id=task_spec.task_id,
                task_type=task_spec.task_type,
                failure_kind=kind.value,
                evidence_path=str(evidence_path),
                output_files=all_output_files,
            )

        return result

    def _build_t1_evidence(
        self,
        task_spec: TaskSpec,
        job_result: JobResult,
        parsed: ParsedVCResult,
        report: ConvergenceReport,
        input_hash: str,
        material_db: Dict[str, Any],
        recovery_actions: Optional[List[str]] = None,
    ) -> EvidenceBundle:
        pseudo_hashes = {}
        for elem, pseudo_file in material_db["pseudos"].items():
            pseudo_path = self.pseudo_dir / pseudo_file
            if pseudo_path.exists():
                pseudo_hashes[elem] = hashlib.sha256(pseudo_path.read_bytes()).hexdigest()[:16]

        return EvidenceBundle(
            task_id=task_spec.task_id,
            task_type=task_spec.task_type,
            status="pass" if report.passed else "fail",
            input_hash=input_hash,
            pseudo_hashes=pseudo_hashes,
            qe_version="7.5",
            parser_version="0.1.0",
            commands=[f"pw.x -in {task_spec.material}_vcrelax.in"],
            walltimes_sec=[job_result.walltime_sec],
            total_walltime_sec=job_result.walltime_sec,
            errors=[job_result.error_message] if job_result.error_message else [],
            recovery_actions=recovery_actions or [],
            verifier_verdict=report.to_dict(),
            physical_results={
                "final_energy_ry": parsed.final_energy_ry,
                "final_energy_ev_per_atom": parsed.final_energy_ev_per_atom,
                "max_force_ry_bohr": parsed.max_force_ry_bohr,
                "pressure_kbar": parsed.pressure_kbar,
                "natoms": parsed.natoms,
            },
            output_files=job_result.output_files,
            log_files=[],
        )

    def _build_t1_result(
        self,
        task_spec: TaskSpec,
        job_result: JobResult,
        parsed: ParsedVCResult,
        report: ConvergenceReport,
        evidence: EvidenceBundle,
        input_file: Path,
        input_hash: str,
        workdir: Path,
    ) -> Dict[str, Any]:
        physical_results = {
            "final_energy_ry": parsed.final_energy_ry,
            "final_energy_ev_per_atom": parsed.final_energy_ev_per_atom,
            "max_force_ry_bohr": parsed.max_force_ry_bohr,
            "pressure_kbar": parsed.pressure_kbar,
            "natoms": parsed.natoms,
            "n_iterations": parsed.n_iterations,
        }
        if parsed.cell:
            physical_results["cell"] = {
                "a_bohr": parsed.cell.a_bohr,
                "b_bohr": parsed.cell.b_bohr,
                "c_bohr": parsed.cell.c_bohr,
                "volume_bohr3": parsed.cell.volume_bohr3,
            }

        return {
            "task_id": task_spec.task_id,
            "task_type": task_spec.task_type,
            "status": "pass" if report.passed else "fail",
            "input_hash": input_hash,
            "verifier_summary": report.summary,
            "verifier_passed": report.passed,
            "checks": report.checks,
            "failure_reasons": report.failure_reasons,
            "warnings": report.warnings,
            "physical_results": physical_results,
            "evidence_path": str(workdir / "evidence.json"),
            "output_files": job_result.output_files,
            "walltime_sec": round(job_result.walltime_sec, 3),
            "qe_calls": 1,
            "input_file": str(input_file),
            "stdout_file": str(input_file.parent / f"{input_file.stem}.out"),
        }

    # ── Internal execution helpers ─────────────────────────────────────────────

    def _execute_pw(self, input_file: Path, workdir: Path) -> JobResult:
        if hasattr(self.executor, 'run_pw'):
            return self.executor.run_pw(input_file, workdir)

        handle = self.executor.stage({"input_file": str(input_file)})
        spec = {"input_file": str(input_file), "workdir": str(workdir)}
        handle = self.executor.submit(handle, spec)
        for _ in range(1200):
            status = self.executor.status(handle)
            if status in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.TIMEOUT, JobStatus.CANCELLED}:
                break
            time.sleep(1)
        else:
            self.executor.cancel(handle)
            return JobResult(
                success=False,
                exit_code=-1,
                error_message="Executor polling timed out",
                remote_path=handle.remote_path,
            )

        fetch = self.executor.fetch(handle, workdir)
        stdout = ""
        stdout_file = workdir / f"{input_file.stem}.out"
        if stdout_file.exists():
            stdout = stdout_file.read_text()
        return JobResult(
            success=fetch.success,
            exit_code=0 if fetch.success else -1,
            stdout=stdout,
            output_files=fetch.output_files,
            walltime_sec=0.0,
            job_done="JOB DONE" in stdout,
            error_message=fetch.error_message,
            remote_path=handle.remote_path,
        )

    def _execute_tool(self, tool: str, input_file: Path, workdir: Path) -> JobResult:
        if hasattr(self.executor, 'run_pw'):
            if tool == "bands.x":
                return self.executor.run_bands_x(input_file, workdir)
            if tool == "dos.x":
                return self.executor.run_dos_x(input_file, workdir)
            if tool == "projwfc.x":
                return self.executor.run_projwfc_x(input_file, workdir)

        handle = self.executor.stage({"input_file": str(input_file), "tool": tool})
        handle = self.executor.submit(handle, {"input_file": str(input_file), "workdir": str(workdir)})
        fetch = self.executor.fetch(handle, workdir)
        stdout = ""
        stdout_file = workdir / f"{input_file.stem}.out"
        if stdout_file.exists():
            stdout = stdout_file.read_text()
        return JobResult(
            success=fetch.success,
            exit_code=0 if fetch.success else -1,
            stdout=stdout,
            output_files=fetch.output_files,
            error_message=fetch.error_message,
            remote_path=handle.remote_path,
        )


def run_task_deterministic(
    task_id: str,
    workdir: Path,
    *,
    pseudo_dir: Optional[Path] = None,
    qe_bin_dir: Optional[Path] = None,
    retry_params: Optional[Dict[str, Any]] = None,
    executor: Optional[Executor] = None,
) -> Dict[str, Any]:
    """Run a DFT task deterministically (no LLM).

    Dispatches to the appropriate runner based on task_type.
    """
    if pseudo_dir is None:
        pseudo_dir = Path(__file__).resolve().parent.parent / "assets" / "pseudos"
    if qe_bin_dir is None:
        qe_bin_dir = Path("/opt/homebrew/bin")

    task_spec = get_task_spec(task_id)
    local_executor = executor or LocalExecutor(
        qe_bin_dir=qe_bin_dir,
        max_walltime_sec=600,
        max_retries=1,
    )

    runner = TaskRunner(
        pseudo_dir=pseudo_dir,
        executor=local_executor,
        max_walltime_sec=600,
        max_retries=1,
    )

    if task_spec.task_type == "T1":
        return runner.run_t1(task_id, workdir, retry_params=retry_params)
    elif task_spec.task_type == "T2":
        subtype = task_spec.parameters.get("subtype")
        if subtype is None:
            subtype = "dos" if "dos" in task_id.lower() else "bands"
        if subtype == "dos":
            return runner.run_t2_dos(task_id, workdir, retry_params=retry_params)
        return runner.run_t2_bands(task_id, workdir, retry_params=retry_params)
    else:
        raise ValueError(f"Unsupported task_type: {task_spec.task_type}")


def run_smoke(task_id: str, workdir: Path = Path("./smoke_test")) -> Dict[str, Any]:
    """Run a smoke test for a task."""
    from dft_forge.cli import probe_environment

    env = probe_environment(workdir)

    if not env.get("qe_binary"):
        return {
            "task_id": task_id,
            "status": "error",
            "error": "QE binary not found",
        }

    return run_task_deterministic(task_id, workdir)
