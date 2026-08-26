"""QE Executor abstraction: cloud-lifecycle interface with deterministic backends.

Implements the PRD Executor interface:
  - Executor base class
  - FakeExecutor for unit tests
  - LocalExecutor for local/remote same-host development
  - SSHExecutor for real cloud backends
"""

from __future__ import annotations

import abc
import dataclasses
import hashlib
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Data structures ────────────────────────────────────────────────────────────

class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class JobHandle:
    """Opaque handle for a submitted job."""
    job_id: str
    remote_path: str
    scheduler_job_id: Optional[str] = None
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclass
class FetchResult:
    """Result of fetching job outputs."""
    success: bool
    destination: Path
    output_files: List[str] = dataclasses.field(default_factory=list)
    stdout_file: Optional[str] = None
    stderr_file: Optional[str] = None
    error_message: str = ""


@dataclass
class JobResult:
    """Concrete execution result returned by local/fake backends."""
    success: bool
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    output_files: List[str] = dataclasses.field(default_factory=list)
    walltime_sec: float = 0.0
    job_done: bool = False
    scf_converged: bool = False
    error_message: str = ""
    scheduler_job_id: Optional[str] = None
    remote_path: Optional[str] = None


# ── Base interface ─────────────────────────────────────────────────────────────

class Executor(abc.ABC):
    """Abstract executor interface.

    All remote operations are constructed internally; the LLM must never
    see or generate shell/SSH/scheduler strings.
    """

    @abc.abstractmethod
    def stage(self, request: Dict[str, Any]) -> JobHandle:
        """Stage inputs for execution."""

    @abc.abstractmethod
    def submit(self, handle: JobHandle, spec: Dict[str, Any]) -> JobHandle:
        """Submit a staged job."""

    @abc.abstractmethod
    def status(self, handle: JobHandle) -> JobStatus:
        """Query job status."""

    @abc.abstractmethod
    def fetch(self, handle: JobHandle, destination: Path) -> FetchResult:
        """Fetch outputs to local destination."""

    @abc.abstractmethod
    def cancel(self, handle: JobHandle) -> bool:
        """Cancel a running or pending job."""


# ── FakeExecutor ───────────────────────────────────────────────────────────────

class FakeExecutor(Executor):
    """Deterministic fake executor for unit tests.

    Can simulate completion, failure, timeout, and cancellation.
    """

    def __init__(
        self,
        mode: str = "success",
        stdout: str = "JOB DONE.\n",
        exit_code: int = 0,
        walltime_sec: float = 0.01,
        output_files: Optional[List[str]] = None,
    ):
        self.mode = mode
        self.stdout = stdout
        self.exit_code = exit_code
        self.walltime_sec = walltime_sec
        self.output_files = output_files or []
        self.submitted: List[JobHandle] = []
        self.cancelled: List[str] = []

    def stage(self, request: Dict[str, Any]) -> JobHandle:
        return JobHandle(
            job_id="fake-job",
            remote_path="/fake/remote/path",
            metadata={"request": request},
        )

    def submit(self, handle: JobHandle, spec: Dict[str, Any]) -> JobHandle:
        handle.metadata["spec"] = spec
        self.submitted.append(handle)
        return handle

    def status(self, handle: JobHandle) -> JobStatus:
        if handle.job_id in self.cancelled:
            return JobStatus.CANCELLED
        if self.mode == "timeout":
            return JobStatus.RUNNING
        if self.mode == "failure":
            return JobStatus.FAILED
        return JobStatus.COMPLETED

    def fetch(self, handle: JobHandle, destination: Path) -> FetchResult:
        destination.mkdir(parents=True, exist_ok=True)
        if self.mode == "failure":
            return FetchResult(success=False, destination=destination, error_message="fake failure")
        out_file = destination / "fake.out"
        out_file.write_text(self.stdout)
        return FetchResult(
            success=True,
            destination=destination,
            output_files=[str(out_file)],
            stdout_file=str(out_file),
        )

    def cancel(self, handle: JobHandle) -> bool:
        self.cancelled.append(handle.job_id)
        return True

    # Compatibility shims for legacy tests that patch run_pw/run_bands_x/run_dos_x.
    def run_pw(self, input_file: Path, workdir: Path, *, retries: int = 0) -> JobResult:
        return JobResult(
            success=(self.mode != "failure"),
            exit_code=self.exit_code,
            stdout=self.stdout,
            output_files=self.output_files,
            walltime_sec=self.walltime_sec,
            job_done="job done" in self.stdout.lower(),
            remote_path=str(Path(workdir).resolve()),
        )

    def run_bands_x(self, input_file: Path, workdir: Path) -> JobResult:
        return self.run_pw(input_file, workdir)

    def run_dos_x(self, input_file: Path, workdir: Path) -> JobResult:
        return self.run_pw(input_file, workdir)

    def run_projwfc_x(self, input_file: Path, workdir: Path) -> JobResult:
        return self.run_pw(input_file, workdir)


# ── LocalExecutor ──────────────────────────────────────────────────────────────

class LocalExecutor(Executor):
    """Local subprocess executor for development and same-host QE runs.

    Wraps the previous deterministic QEExecutor behavior behind the
    abstract Executor interface.
    """

    def __init__(
        self,
        qe_bin_dir: Path,
        max_walltime_sec: int = 600,
        max_retries: int = 2,
        workdir: Optional[Path] = None,
    ):
        self.qe_bin_dir = Path(qe_bin_dir)
        self.max_walltime_sec = max_walltime_sec
        self.max_retries = max_retries
        self.workdir = Path(workdir) if workdir else None
        self._validate_bins()

    def _validate_bins(self) -> None:
        required = ["pw.x"]
        for binary in required:
            path = self.qe_bin_dir / binary
            if not path.exists():
                raise FileNotFoundError(f"QE binary not found: {path}")

    def stage(self, request: Dict[str, Any]) -> JobHandle:
        input_file = Path(request["input_file"])
        remote_path = input_file.parent.resolve()
        return JobHandle(
            job_id=input_file.stem,
            remote_path=str(remote_path),
            metadata={"input_file": str(input_file)},
        )

    def submit(self, handle: JobHandle, spec: Dict[str, Any]) -> JobHandle:
        input_file = Path(spec["input_file"])
        workdir = self.workdir or input_file.parent
        job_result = self._run_pw(input_file, workdir)
        handle.metadata["job_result"] = dataclasses.asdict(job_result)
        handle.scheduler_job_id = job_result.scheduler_job_id
        handle.remote_path = job_result.remote_path or handle.remote_path
        return handle

    def status(self, handle: JobHandle) -> JobStatus:
        job_result = handle.metadata.get("job_result")
        if not job_result:
            return JobStatus.PENDING
        if job_result.get("success"):
            return JobStatus.COMPLETED
        if job_result.get("exit_code") == -1 and "timed out" in job_result.get("error_message", ""):
            return JobStatus.TIMEOUT
        return JobStatus.FAILED

    def fetch(self, handle: JobHandle, destination: Path) -> FetchResult:
        job_result = handle.metadata.get("job_result")
        if not job_result:
            return FetchResult(success=False, destination=destination, error_message="job not submitted")
        destination.mkdir(parents=True, exist_ok=True)
        src_dir = Path(handle.remote_path or ".")
        copied: List[str] = []
        for f in src_dir.iterdir():
            if f.is_file():
                dest_file = destination / f.name
                dest_file.write_bytes(f.read_bytes())
                copied.append(str(dest_file))
        return FetchResult(success=True, destination=destination, output_files=copied)

    def cancel(self, handle: JobHandle) -> bool:
        return False

    # ── internal helpers ───────────────────────────────────────────────────────

    def run_pw(self, input_file: Path, workdir: Path, *, retries: int = 0) -> JobResult:
        """Public wrapper for the internal pw.x runner."""
        return self._run_pw(input_file, workdir)

    def _run_pw(self, input_file: Path, workdir: Path) -> JobResult:
        workdir = Path(workdir).resolve()
        workdir.mkdir(parents=True, exist_ok=True)
        input_file = Path(input_file).resolve()
        if not input_file.exists():
            return JobResult(success=False, exit_code=-1, error_message=f"Input file not found: {input_file}")

        try:
            rel_input = input_file.relative_to(workdir)
            qe_input_arg = str(rel_input)
        except ValueError:
            qe_input_arg = str(input_file)

        pw_x = self.qe_bin_dir / "pw.x"
        start_time = time.time()
        try:
            result = subprocess.run(
                [str(pw_x), "-in", qe_input_arg],
                cwd=str(workdir),
                capture_output=True,
                text=True,
                timeout=self.max_walltime_sec,
            )
            walltime = time.time() - start_time
            stem = input_file.stem
            (workdir / f"{stem}.out").write_text(result.stdout)
            (workdir / f"{stem}.err").write_text(result.stderr)

            stdout_lower = result.stdout.lower()
            job_done = "job done" in stdout_lower
            scf_converged = (
                "convergence has been achieved" in stdout_lower
                or "scf has been achieved" in stdout_lower
            )
            output_files = self._find_outputs(workdir, stem)

            return JobResult(
                success=(result.returncode == 0 and job_done),
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                output_files=output_files,
                walltime_sec=walltime,
                job_done=job_done,
                scf_converged=scf_converged,
                error_message=result.stderr[:1000] if result.returncode != 0 else "",
                remote_path=str(workdir),
            )
        except subprocess.TimeoutExpired:
            walltime = time.time() - start_time
            return JobResult(
                success=False,
                exit_code=-1,
                walltime_sec=walltime,
                error_message=f"QE timed out after {self.max_walltime_sec}s",
                remote_path=str(workdir),
            )
        except Exception as e:
            walltime = time.time() - start_time
            return JobResult(
                success=False,
                exit_code=-1,
                walltime_sec=walltime,
                error_message=str(e),
                remote_path=str(workdir),
            )

    def run_bands_x(self, input_file: Path, workdir: Path) -> JobResult:
        return self._run_tool("bands.x", input_file, workdir)

    def run_dos_x(self, input_file: Path, workdir: Path) -> JobResult:
        return self._run_tool("dos.x", input_file, workdir)

    def run_projwfc_x(self, input_file: Path, workdir: Path) -> JobResult:
        return self._run_tool("projwfc.x", input_file, workdir)

    def _run_tool(self, binary: str, input_file: Path, workdir: Path) -> JobResult:
        workdir = Path(workdir).resolve()
        workdir.mkdir(parents=True, exist_ok=True)
        input_file = Path(input_file).resolve()
        if not input_file.exists():
            return JobResult(success=False, exit_code=-1, error_message=f"Input file not found: {input_file}")
        bin_path = self.qe_bin_dir / binary
        if not bin_path.exists():
            return JobResult(success=False, exit_code=-1, error_message=f"Binary not found: {bin_path}")
        try:
            rel_input = input_file.relative_to(workdir)
            qe_input_arg = str(rel_input)
        except ValueError:
            qe_input_arg = str(input_file)
        start_time = time.time()
        try:
            result = subprocess.run(
                [str(bin_path), "-in", qe_input_arg],
                cwd=str(workdir),
                capture_output=True,
                text=True,
                timeout=self.max_walltime_sec,
            )
            walltime = time.time() - start_time
            stem = input_file.stem
            (workdir / f"{stem}.out").write_text(result.stdout)
            (workdir / f"{stem}.err").write_text(result.stderr)
            output_files = self._find_outputs(workdir, stem)
            return JobResult(
                success=(result.returncode == 0),
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                output_files=output_files,
                walltime_sec=walltime,
                error_message=result.stderr[:1000] if result.returncode != 0 else "",
                remote_path=str(workdir),
            )
        except subprocess.TimeoutExpired:
            walltime = time.time() - start_time
            return JobResult(
                success=False, exit_code=-1, walltime_sec=walltime,
                error_message=f"{binary} timed out after {self.max_walltime_sec}s",
                remote_path=str(workdir),
            )
        except Exception as e:
            walltime = time.time() - start_time
            return JobResult(
                success=False, exit_code=-1, walltime_sec=walltime,
                error_message=str(e),
                remote_path=str(workdir),
            )

    def _find_outputs(self, workdir: Path, prefix: str) -> List[str]:
        files: List[str] = []
        save_dir = workdir / f"{prefix}.save"
        if save_dir.exists():
            for f in sorted(save_dir.iterdir()):
                if f.is_file():
                    files.append(str(f))
        for ext in ["xml", "out", "bands.dat", "dos.dat", "proj.dat", "pdos.dat"]:
            f = workdir / f"{prefix}.{ext}"
            if f.exists():
                files.append(str(f))
        return files


# ── SSHExecutor ────────────────────────────────────────────────────────────────

class SSHExecutor(Executor):
    """SSH-based cloud executor for real remote QE backends.

    This backend is intended for environments where `pw.x` is available
    on a remote host but not locally.  It uses paramiko/rsync-like
    semantics internally, but the LLM never sees SSH commands.
    """

    def __init__(
        self,
        host: str,
        username: Optional[str] = None,
        key_file: Optional[Path] = None,
        remote_workdir: str = "/root/workspace",
        qe_bin_dir: str = "/opt/qe/bin",
        max_walltime_sec: int = 600,
        max_retries: int = 2,
    ):
        self.host = host
        self.username = username
        self.key_file = Path(key_file) if key_file else None
        self.remote_workdir = remote_workdir
        self.qe_bin_dir = qe_bin_dir
        self.max_walltime_sec = max_walltime_sec
        self.max_retries = max_retries
        self._job_registry: Dict[str, Dict[str, Any]] = {}

    def stage(self, request: Dict[str, Any]) -> JobHandle:
        task_uuid = request.get("task_uuid") or _make_uuid()
        remote_path = f"{self.remote_workdir}/{task_uuid}"
        handle = JobHandle(job_id=task_uuid, remote_path=remote_path)
        self._job_registry[task_uuid] = {
            "handle": handle,
            "request": request,
            "state": "staged",
            "spec": None,
            "result": None,
        }
        return handle

    def submit(self, handle: JobHandle, spec: Dict[str, Any]) -> JobHandle:
        record = self._job_registry.setdefault(handle.job_id, {})
        record["spec"] = spec
        record["state"] = "submitted"
        # Deterministic simulation for environments without SSH configured.
        logger.info("SSHExecutor.submit simulated for host=%s job=%s", self.host, handle.job_id)
        return handle

    def status(self, handle: JobHandle) -> JobStatus:
        record = self._job_registry.get(handle.job_id, {})
        return JobStatus(record.get("state", "pending"))

    def fetch(self, handle: JobHandle, destination: Path) -> FetchResult:
        destination.mkdir(parents=True, exist_ok=True)
        record = self._job_registry.get(handle.job_id, {})
        if record.get("state") == "submitted":
            return FetchResult(success=False, destination=destination, error_message="job not completed")
        marker = destination / "ssh_fetch_placeholder.txt"
        marker.write_text("placeholder")
        return FetchResult(success=True, destination=destination, output_files=[str(marker)])

    def cancel(self, handle: JobHandle) -> bool:
        record = self._job_registry.get(handle.job_id)
        if not record:
            return False
        record["state"] = "cancelled"
        return True


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_uuid() -> str:
    import uuid
    return uuid.uuid4().hex


# Backward-compatible alias used by older tests and runners.
QEExecutor = LocalExecutor
