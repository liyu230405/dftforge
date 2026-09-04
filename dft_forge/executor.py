"""QE Executor abstraction: cloud-lifecycle interface with deterministic backends.

Implements the PRD Executor interface:
  - Executor base class
  - FakeExecutor for unit tests
  - LocalExecutor for local/remote same-host development
  - SSHExecutor for real remote backends (grafted onto hpc/runner.SSHRunner)
"""

from __future__ import annotations

import abc
import asyncio
import base64
import binascii
import dataclasses
import hashlib
import json
import logging
import os
import re
import shlex
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

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


# ── Async facade ──────────────────────────────────────────────────────────────

class AsyncExecutor(Protocol):
    """Awaitable executor interface for asyncio callers.

    A future asyncssh-based remote executor implements this directly; sync
    executors are wrapped with AsyncLocalExecutor.
    """

    async def stage(self, request: Dict[str, Any]) -> JobHandle: ...

    async def submit(self, handle: JobHandle, spec: Dict[str, Any]) -> JobHandle: ...

    async def status(self, handle: JobHandle) -> JobStatus: ...

    async def fetch(self, handle: JobHandle, destination: Path) -> FetchResult: ...

    async def cancel(self, handle: JobHandle) -> bool: ...


class AsyncLocalExecutor:
    """Wrap any sync Executor as an AsyncExecutor via asyncio.to_thread."""

    def __init__(self, executor: Executor):
        self._sync = executor

    async def stage(self, request: Dict[str, Any]) -> JobHandle:
        return await asyncio.to_thread(self._sync.stage, request)

    async def submit(self, handle: JobHandle, spec: Dict[str, Any]) -> JobHandle:
        return await asyncio.to_thread(self._sync.submit, handle, spec)

    async def status(self, handle: JobHandle) -> JobStatus:
        return await asyncio.to_thread(self._sync.status, handle)

    async def fetch(self, handle: JobHandle, destination: Path) -> FetchResult:
        return await asyncio.to_thread(self._sync.fetch, handle, destination)

    async def cancel(self, handle: JobHandle) -> bool:
        return await asyncio.to_thread(self._sync.cancel, handle)


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
    def run_pw(self, input_file: Path, workdir: Path, *, retries: int = 0, timeout: float | None = None) -> JobResult:
        return JobResult(
            success=(self.mode != "failure"),
            exit_code=self.exit_code,
            stdout=self.stdout,
            output_files=self.output_files,
            walltime_sec=self.walltime_sec,
            job_done="job done" in self.stdout.lower(),
            remote_path=str(Path(workdir).resolve()),
        )

    def run_bands_x(self, input_file: Path, workdir: Path, timeout: float | None = None) -> JobResult:
        return self.run_pw(input_file, workdir, timeout=timeout)

    def run_dos_x(self, input_file: Path, workdir: Path, timeout: float | None = None) -> JobResult:
        return self.run_pw(input_file, workdir, timeout=timeout)

    def run_projwfc_x(self, input_file: Path, workdir: Path, timeout: float | None = None) -> JobResult:
        return self.run_pw(input_file, workdir, timeout=timeout)


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
        self._running: Dict[int, subprocess.Popen] = {}
        self._proc_lock = threading.Lock()
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

    def run_pw(self, input_file: Path, workdir: Path, *, retries: int = 0, timeout: float | None = None) -> JobResult:
        """Public wrapper for the internal pw.x runner."""
        return self._run_pw(input_file, workdir, timeout=timeout)

    def _run_pw(self, input_file: Path, workdir: Path, timeout: float | None = None) -> JobResult:
        result = self._run_tool("pw.x", input_file, workdir, timeout=timeout)
        stdout_lower = result.stdout.lower()
        result.job_done = "job done" in stdout_lower
        result.scf_converged = (
            "convergence has been achieved" in stdout_lower
            or "scf has been achieved" in stdout_lower
        )
        # pw.x can exit 0 without finishing the calculation; JOB DONE is
        # the authoritative success marker
        result.success = result.exit_code == 0 and result.job_done
        return result

    def run_bands_x(self, input_file: Path, workdir: Path, timeout: float | None = None) -> JobResult:
        return self._run_tool("bands.x", input_file, workdir, timeout=timeout)

    def run_dos_x(self, input_file: Path, workdir: Path, timeout: float | None = None) -> JobResult:
        return self._run_tool("dos.x", input_file, workdir, timeout=timeout)

    def run_projwfc_x(self, input_file: Path, workdir: Path, timeout: float | None = None) -> JobResult:
        return self._run_tool("projwfc.x", input_file, workdir, timeout=timeout)

    def _run_tool(self, binary: str, input_file: Path, workdir: Path, timeout: float | None = None) -> JobResult:
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
        # Popen (not subprocess.run) so cancel_all() can terminate a run that
        # is minutes from finishing; start_new_session puts pw.x (and any MPI
        # children it spawns) in its own process group for group-wide kills
        try:
            proc = subprocess.Popen(
                [str(bin_path), "-in", qe_input_arg],
                cwd=str(workdir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
        except Exception as e:
            return JobResult(
                success=False, exit_code=-1, walltime_sec=time.time() - start_time,
                error_message=f"failed to launch {binary}: {e}",
                remote_path=str(workdir),
            )
        with self._proc_lock:
            self._running[proc.pid] = proc
        try:
            stdout, stderr = proc.communicate(timeout=timeout or self.max_walltime_sec)
            cancelled = self._was_cancelled(proc)
        except subprocess.TimeoutExpired:
            self._terminate_group(proc)
            stdout, stderr = proc.communicate()
            walltime = time.time() - start_time
            return JobResult(
                success=False, exit_code=-1, walltime_sec=walltime,
                stdout=stdout or "", stderr=stderr or "",
                error_message=f"{binary} timed out after {timeout or self.max_walltime_sec}s",
                remote_path=str(workdir),
            )
        finally:
            with self._proc_lock:
                self._running.pop(proc.pid, None)
        walltime = time.time() - start_time
        stem = input_file.stem
        (workdir / f"{stem}.out").write_text(stdout)
        (workdir / f"{stem}.err").write_text(stderr)
        output_files = self._find_outputs(workdir, stem)
        if cancelled:
            return JobResult(
                success=False, exit_code=proc.returncode, stdout=stdout, stderr=stderr,
                output_files=output_files, walltime_sec=walltime,
                error_message=f"{binary} cancelled by user",
                remote_path=str(workdir),
            )
        return JobResult(
            success=(proc.returncode == 0),
            exit_code=proc.returncode,
            stdout=stdout,
            stderr=stderr,
            output_files=output_files,
            walltime_sec=walltime,
            error_message=stderr[:1000] if proc.returncode != 0 else "",
            remote_path=str(workdir),
        )

    # ── process-level cancellation ────────────────────────────────────────────

    def _was_cancelled(self, proc: subprocess.Popen) -> bool:
        """A negative returncode means death by signal — SIGTERM/SIGKILL is
        what cancel_all() sends, so any negative exit here is a cancellation."""
        return proc.returncode is not None and proc.returncode < 0

    def _terminate_group(self, proc: subprocess.Popen, grace_sec: float = 5.0) -> None:
        """SIGTERM the whole process group, then SIGKILL after a grace period."""
        if proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            proc.terminate()
        try:
            proc.wait(timeout=grace_sec)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            proc.kill()
        try:
            proc.wait(timeout=grace_sec)
        except subprocess.TimeoutExpired:
            logger.warning("process %s survived SIGKILL", proc.pid)

    def cancel_all(self) -> int:
        """Terminate every in-flight subprocess. Returns how many were killed."""
        with self._proc_lock:
            procs = [p for p in self._running.values() if p.poll() is None]
        for proc in procs:
            self._terminate_group(proc)
        return len(procs)

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


# ── SSHExecutor (real remote execution via hpc/runner.SSHRunner) ────────────

# b64 chars per runner.run call: keeps each command far below ARG_MAX even
# after ssh wrapping and shlex quoting
_B64_CHUNK = 48000

_SCHED_POLL_INTERVAL = 5.0


class SSHExecutor(Executor):
    """Remote QE executor grafted onto ``hpc/runner.SSHRunner``.

    Every remote operation goes through an injectable CommandRunner, so
    tests script a fake runner and no network is ever touched.

    Two execution modes:
    - ``scheduler="none"`` (direct): blocking ssh run of the QE binary under
      remote-side ``timeout``; outputs are tarred back afterwards.
    - ``scheduler="slurm"|"pbs"``: renders an sbatch/qsub script and submits
      through the hpc scheduler adapters; status maps squeue/sacct/qstat.

    File transport is chunked base64+tar over the text-based CommandResult
    API (the runner has no stdin channel). ``-h`` dereferences the .save
    symlinks on upload so chained nodes (scf→nscf→bands) carry real
    wavefunctions to the remote side. Multi-GB wavefunction dirs will be
    slow over this transport; prefer scheduler mode for production runs.
    """

    _STATE_TO_STATUS = {
        "staged": JobStatus.PENDING,
        "submitted": JobStatus.PENDING,
        "running": JobStatus.RUNNING,
        "pending": JobStatus.PENDING,
        "completed": JobStatus.COMPLETED,
        "failed": JobStatus.FAILED,
        "cancelled": JobStatus.CANCELLED,
        "timeout": JobStatus.TIMEOUT,
    }

    def __init__(
        self,
        host: str,
        username: Optional[str] = None,
        ssh_key: Optional[Path] = None,
        key_file: Optional[Path] = None,  # legacy alias
        remote_workdir: str | Path = "/root/workspace",
        qe_bin_dir: str | Path = "/opt/qe/bin",
        scheduler: str = "slurm",
        port: int = 22,
        max_walltime_sec: int = 600,
        max_retries: int = 2,
        jump_host: Optional[str] = None,
        runner: Optional[Any] = None,
    ):
        self.host = host
        self.username = username
        self.ssh_key = Path(ssh_key or key_file) if (ssh_key or key_file) else None
        self.remote_workdir = str(remote_workdir)
        self.qe_bin_dir = str(qe_bin_dir)
        self.scheduler = (scheduler or "slurm").lower()
        self.port = port
        self.max_walltime_sec = max_walltime_sec
        self.max_retries = max_retries
        self._job_registry: Dict[str, Dict[str, Any]] = {}

        # lazy import: hpc.scheduler imports executor.JobStatus at module
        # level, so importing it here avoids a cycle at import time
        if runner is not None:
            self.runner = runner
        else:
            from dft_forge.hpc.runner import SSHRunner

            self.runner = SSHRunner(
                host,
                username=username,
                port=port,
                key_file=str(self.ssh_key) if self.ssh_key else None,
                jump_host=jump_host,
            )
        self.scheduler_adapter = None
        if self.scheduler == "slurm":
            from dft_forge.hpc.scheduler import SlurmScheduler

            self.scheduler_adapter = SlurmScheduler(self.runner)
        elif self.scheduler == "pbs":
            from dft_forge.hpc.scheduler import PbsScheduler

            self.scheduler_adapter = PbsScheduler(self.runner)

    # ── Cloud-lifecycle interface ─────────────────────────────────────────────

    def stage(self, request: Dict[str, Any]) -> JobHandle:
        task_uuid = request.get("task_uuid") or _make_uuid()
        remote_path = self._remote_job_dir(task_uuid)
        handle = JobHandle(job_id=task_uuid, remote_path=remote_path)
        self._job_registry[task_uuid] = {
            "handle": handle,
            "request": request,
            "state": "staged",
            "spec": None,
            "scheduler_job_id": None,
            "error": None,
        }
        return handle

    def submit(self, handle: JobHandle, spec: Dict[str, Any]) -> JobHandle:
        record = self._job_registry.setdefault(
            handle.job_id,
            {"handle": handle, "request": {}, "state": "staged", "spec": None,
             "scheduler_job_id": None, "error": None},
        )
        record["spec"] = spec
        input_file = Path(str(spec.get("input_file", "")))
        workdir = Path(str(spec.get("workdir") or (input_file.parent if input_file.exists() else ".")))
        remote_dir = handle.remote_path or self._remote_job_dir(handle.job_id)

        if not input_file.exists():
            record["state"] = "failed"
            record["error"] = f"input file not found: {input_file}"
            raise FileNotFoundError(record["error"])

        try:
            self._upload_dir(workdir if workdir.is_dir() else input_file.parent, remote_dir)
        except Exception as exc:
            record["state"] = "failed"
            record["error"] = f"upload failed: {exc}"
            raise

        binary = str(spec.get("tool", "pw.x"))
        if self.scheduler_adapter is not None:
            ok, msg, sched_id = self._submit_scheduler(remote_dir, input_file.name, handle.job_id, binary)
            record["scheduler_job_id"] = sched_id
            if not ok:
                record["state"] = "failed"
                record["error"] = msg
                raise RuntimeError(f"scheduler submit failed: {msg}")
            record["state"] = "submitted"
        else:
            record["state"] = "running"
            result = self._run_direct(remote_dir, binary, input_file.name, timeout=None)
            record["state"] = "completed" if result.ok else "failed"
            record["error"] = None if result.ok else result.stderr.strip()[:500]
        return handle

    def status(self, handle: JobHandle) -> JobStatus:
        record = self._job_registry.get(handle.job_id)
        if not record:
            return JobStatus.PENDING
        sched_id = record.get("scheduler_job_id")
        if sched_id and self.scheduler_adapter is not None:
            sched_status = self.scheduler_adapter.get_job_status(sched_id)
            if sched_status is not None:
                # remember the freshest scheduler state so fetch/cancel after
                # the fact see a consistent record
                record["state"] = sched_status.value
                return sched_status
        return self._STATE_TO_STATUS.get(record.get("state", "pending"), JobStatus.PENDING)

    def fetch(self, handle: JobHandle, destination: Path) -> FetchResult:
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        record = self._job_registry.get(handle.job_id, {})
        if record.get("state") in ("staged", "submitted", "running"):
            return FetchResult(success=False, destination=destination, error_message="job not completed")
        try:
            files = self._download_dir(handle.remote_path, destination)
        except Exception as exc:
            return FetchResult(success=False, destination=destination, error_message=f"download failed: {exc}")
        return FetchResult(
            success=True,
            destination=destination,
            output_files=files,
            stdout_file=str(destination / "stdout.txt") if (destination / "stdout.txt").exists() else None,
            stderr_file=str(destination / "stderr.txt") if (destination / "stderr.txt").exists() else None,
        )

    def cancel(self, handle: JobHandle) -> bool:
        record = self._job_registry.get(handle.job_id)
        if not record:
            return False
        sched_id = record.get("scheduler_job_id")
        if sched_id and self.scheduler_adapter is not None:
            ok, _msg = self.scheduler_adapter.cancel_job(sched_id)
        else:
            # direct mode: kill whatever is still running inside the job dir
            # ('; true' keeps the exit code clean when nothing matches)
            self._run_remote(f"pkill -f {shlex.quote(handle.remote_path)}; true")
            ok = True
        record["state"] = "cancelled"
        return bool(ok)

    def cancel_all(self) -> int:
        """Cancel every live job (web stop-button chain)."""
        n = 0
        for job_id, record in list(self._job_registry.items()):
            if record.get("state") in ("staged", "submitted", "running"):
                if self.cancel(record["handle"]):
                    n += 1
        return n

    # ── Direct runner bindings (graph-engine path) ────────────────────────────

    def run_pw(self, input_file: Path, workdir: Path, *, retries: int = 0, timeout: float | None = None) -> JobResult:
        return self._run_remote_tool("pw.x", input_file, workdir, timeout)

    def run_bands_x(self, input_file: Path, workdir: Path, timeout: float | None = None) -> JobResult:
        return self._run_remote_tool("bands.x", input_file, workdir, timeout)

    def run_dos_x(self, input_file: Path, workdir: Path, timeout: float | None = None) -> JobResult:
        return self._run_remote_tool("dos.x", input_file, workdir, timeout)

    def run_projwfc_x(self, input_file: Path, workdir: Path, timeout: float | None = None) -> JobResult:
        return self._run_remote_tool("projwfc.x", input_file, workdir, timeout)

    def _run_remote_tool(self, binary: str, input_file: Path, workdir: Path, timeout: float | None = None) -> JobResult:
        input_file = Path(input_file).resolve()
        workdir = Path(workdir).resolve()
        if not input_file.exists():
            return JobResult(success=False, exit_code=-1, error_message=f"Input file not found: {input_file}")

        job_id = f"{binary.replace('.x', '')}_{_make_uuid()[:8]}"
        remote_dir = self._remote_job_dir(job_id)
        handle = self.stage({"task_uuid": job_id, "input_file": str(input_file)})
        record = self._job_registry[job_id]
        record["state"] = "running"
        start = time.time()

        try:
            if workdir.is_dir():
                self._upload_dir(workdir, remote_dir)
            else:
                self._upload_dir(input_file.parent, remote_dir)
        except Exception as exc:
            record["state"] = "failed"
            return JobResult(
                success=False, exit_code=-1, walltime_sec=time.time() - start,
                error_message=f"upload failed: {exc}", remote_path=remote_dir,
            )

        if self.scheduler_adapter is not None:
            ok, msg, sched_id = self._submit_scheduler(remote_dir, input_file.name, job_id, binary)
            record["scheduler_job_id"] = sched_id
            if not ok:
                record["state"] = "failed"
                return JobResult(
                    success=False, exit_code=-1, walltime_sec=time.time() - start,
                    error_message=f"scheduler submit failed: {msg}", remote_path=remote_dir,
                )
            terminal = self._poll_scheduler(sched_id, timeout or self.max_walltime_sec)
            if terminal not in (JobStatus.COMPLETED,):
                record["state"] = "failed" if terminal else "timeout"
                state_name = terminal.value if terminal else "timeout"
                return JobResult(
                    success=False, exit_code=-1, walltime_sec=time.time() - start,
                    error_message=f"job ended in state {state_name}", remote_path=remote_dir,
                    scheduler_job_id=sched_id,
                )
        else:
            result = self._run_direct(remote_dir, binary, input_file.name, timeout=timeout)
            if not result.ok:
                record["state"] = "failed"
                # bring partial outputs back even on failure — the .out file
                # often holds the QE error block the repairer pattern-matches
                try:
                    self._download_dir(remote_dir, workdir)
                except Exception:
                    pass
                return JobResult(
                    success=False, exit_code=result.exit_code, walltime_sec=time.time() - start,
                    error_message=result.stderr.strip()[:500] or f"{binary} exited {result.exit_code}",
                    remote_path=remote_dir,
                )

        try:
            files = self._download_dir(remote_dir, workdir)
        except Exception as exc:
            record["state"] = "failed"
            return JobResult(
                success=False, exit_code=-1, walltime_sec=time.time() - start,
                error_message=f"download failed: {exc}", remote_path=remote_dir,
            )
        stdout = self._read_local(workdir / f"{input_file.stem}.out")
        stderr = self._read_local(workdir / f"{input_file.stem}.err")
        record["state"] = "completed"
        return JobResult(
            success=(binary != "pw.x") or ("JOB DONE" in stdout),
            exit_code=0,
            stdout=stdout,
            stderr=stderr,
            output_files=files,
            walltime_sec=time.time() - start,
            job_done="JOB DONE" in stdout,
            scf_converged="convergence has been achieved" in stdout.lower(),
            remote_path=remote_dir,
            scheduler_job_id=record.get("scheduler_job_id"),
        )

    # ── Transport helpers ─────────────────────────────────────────────────────

    def _remote_job_dir(self, job_id: str) -> str:
        return f"{self.remote_workdir}/{job_id}"

    def _run_remote(self, cmd: str, timeout: Optional[float] = None):
        return self.runner.run(cmd, timeout=timeout or 120.0)

    def _upload_dir(self, local_dir: Path, remote_dir: str) -> None:
        local_dir = Path(local_dir)
        if not local_dir.is_dir():
            raise FileNotFoundError(f"local dir not found: {local_dir}")
        # -h dereferences symlinks: the chained .save links upload as real dirs
        proc = subprocess.run(
            ["tar", "-C", str(local_dir), "-chf", "-", "."], capture_output=True
        )
        if proc.returncode != 0:
            raise RuntimeError(f"local tar failed: {proc.stderr.decode(errors='replace')[:200]}")
        b64 = base64.b64encode(proc.stdout).decode("ascii")
        archive = f"{remote_dir}.b64"
        self._run_remote(
            f"mkdir -p {shlex.quote(remote_dir)} && rm -f {shlex.quote(archive)}"
        )
        for i in range(0, len(b64), _B64_CHUNK):
            self._run_remote(
                f"printf '%s' {shlex.quote(b64[i:i + _B64_CHUNK])} >> {shlex.quote(archive)}"
            )
        res = self._run_remote(
            f"base64 -d {shlex.quote(archive)} | tar -C {shlex.quote(remote_dir)} -xf - "
            f"&& rm -f {shlex.quote(archive)}"
        )
        if not res.ok:
            raise RuntimeError(f"remote untar failed: {res.stderr.strip()[:200]}")

    def _download_dir(self, remote_dir: str, destination: Path) -> List[str]:
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        res = self._run_remote(
            f"tar -C {shlex.quote(remote_dir)} -cf - . 2>/dev/null | base64", timeout=300.0
        )
        if not res.ok or not res.stdout.strip():
            return []
        try:
            payload = base64.b64decode(res.stdout.strip())
        except (binascii.Error, ValueError):
            return []
        proc = subprocess.run(
            ["tar", "-C", str(destination), "-xf", "-"], input=payload, capture_output=True
        )
        if proc.returncode != 0:
            return []
        return [str(p) for p in sorted(destination.rglob("*")) if p.is_file()]

    @staticmethod
    def _read_local(path: Path) -> str:
        return path.read_text(errors="replace") if path.exists() else ""

    # ── Execution helpers ─────────────────────────────────────────────────────

    def _run_direct(self, remote_dir: str, binary: str, input_name: str, timeout: Optional[float]):
        from dft_forge.hpc.runner import CommandResult

        walltime = int(timeout or self.max_walltime_sec)
        stem = Path(input_name).stem
        cmd = (
            f"cd {shlex.quote(remote_dir)} && "
            f"timeout {walltime} {shlex.quote(self.qe_bin_dir + '/' + binary)} "
            f"-in {shlex.quote(input_name)} > {shlex.quote(stem)}.out "
            f"2> {shlex.quote(stem)}.err; echo DFTFORGE_EXIT_$?"
        )
        # remote walltime + transfer margin for the ssh call itself
        res = self._run_remote(cmd, timeout=walltime + 120)
        exit_code = 0
        m = re.search(r"DFTFORGE_EXIT_(-?\d+)", res.stdout)
        if m:
            exit_code = int(m.group(1))
        stderr = res.stderr
        # timeout(1) exits 124 when the walltime kills the binary
        if exit_code == 124:
            stderr = f"{binary} timed out after {walltime}s"
        return CommandResult(exit_code, stdout=res.stdout, stderr=stderr)

    def _submit_scheduler(self, remote_dir: str, input_name: str, job_id: str, binary: str):
        from dft_forge.hpc.job_script import render_sbatch, render_qsub

        stem = Path(input_name).stem
        launcher = "mpirun" if self.scheduler == "pbs" else "srun"
        params = {
            "input_file": input_name,
            "output_file": f"{stem}.out",
            "env_setup": f"export PATH={shlex.quote(self.qe_bin_dir)}:$PATH",
            # The post-processing nodes must invoke their own executables.
            # Falling back to engine="qe" here always rendered pw.x, so a
            # scheduler-backed bands/DOS/PDOS workflow submitted the wrong
            # program while direct SSH mode happened to work correctly.
            "run_command": (
                f"{launcher} {shlex.quote(binary)} -in {shlex.quote(input_name)} "
                f"> {shlex.quote(stem + '.out')}"
            ),
        }
        if self.scheduler == "pbs":
            script = render_qsub(params, job_name=job_id[:12], work_dir=remote_dir, engine="qe")
        else:
            script = render_sbatch(params, job_name=job_id[:12], work_dir=remote_dir, engine="qe")
        return self.scheduler_adapter.submit_job(script, remote_dir, job_id[:12])

    def _poll_scheduler(self, sched_id: str, walltime: float) -> Optional[JobStatus]:
        deadline = time.time() + walltime + 60
        while time.time() < deadline:
            status = self.scheduler_adapter.get_job_status(sched_id)
            if status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.TIMEOUT):
                return status
            time.sleep(_SCHED_POLL_INTERVAL)
        return None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_uuid() -> str:
    import uuid
    return uuid.uuid4().hex


# Backward-compatible alias used by older tests and runners.
QEExecutor = LocalExecutor
