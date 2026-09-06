"""SLURM and PBS scheduler adapters.

submit -> remote sbatch/qsub; status -> squeue with sacct fallback
(SLURM's squeue lags for finished jobs); cancel -> scancel/qdel.
"""

from __future__ import annotations

import shlex
import time
from typing import Optional, Tuple

from dft_forge.executor import JobStatus
from dft_forge.hpc.job_script import render_qsub, render_sbatch
from dft_forge.hpc.runner import CommandRunner, CommandResult


class SchedulerInterface:
    """Abstract HPC job scheduler."""

    def submit_job(
        self,
        script_text: str,
        work_dir: str,
        job_name: str,
        *,
        partition: Optional[str] = None,
        nodes: int = 1,
        ntasks: int = 8,
        cpus_per_task: int = 1,
        time_limit: str = "24:00:00",
        memory: Optional[str] = None,
    ) -> Tuple[bool, str, Optional[str]]:
        """Submit a job script; returns (success, message, job_id)."""
        raise NotImplementedError

    def get_job_status(self, job_id: str) -> Optional[JobStatus]:
        raise NotImplementedError

    def cancel_job(self, job_id: str) -> Tuple[bool, str]:
        raise NotImplementedError


class SlurmScheduler(SchedulerInterface):
    def __init__(self, runner: CommandRunner, poll_timeout: float = 30.0):
        self.runner = runner
        self.poll_timeout = poll_timeout

    STATUS_MAP = {
        "PENDING": JobStatus.PENDING,
        "RUNNING": JobStatus.RUNNING,
        "COMPLETED": JobStatus.COMPLETED,
        "FAILED": JobStatus.FAILED,
        "CANCELLED": JobStatus.CANCELLED,
        "TIMEOUT": JobStatus.FAILED,
        "NODE_FAIL": JobStatus.FAILED,
        "OUT_OF_MEMORY": JobStatus.FAILED,
        "PREEMPTED": JobStatus.CANCELLED,
        "SUSPENDED": JobStatus.PENDING,
        "CONFIGURING": JobStatus.PENDING,
        "COMPLETING": JobStatus.RUNNING,
    }

    def submit_job(self, script_text, work_dir, job_name, *, partition=None, nodes=1,
                   ntasks=8, cpus_per_task=1, time_limit="24:00:00", memory=None):
        script_name = f"submit_{job_name}_{int(time.time())}.sh"
        script_path = f"{work_dir}/{script_name}"
        heredoc = (
            f"mkdir -p {shlex.quote(work_dir)} && cat > {shlex.quote(script_path)} "
            f"<< 'DFTFORGE_EOF'\n{script_text}\nDFTFORGE_EOF"
        )
        result = self.runner.run(heredoc)
        if not result.ok:
            return False, f"failed to write script: {result.stderr.strip()}", None
        result = self.runner.run(
            f"cd {shlex.quote(work_dir)} && sbatch {shlex.quote(script_name)}"
        )
        if not result.ok:
            return False, result.stderr.strip() or "sbatch failed", None
        # stdout like "Submitted batch job 12345"
        job_id = next((tok for tok in result.stdout.split() if tok.isdigit()), None)
        if job_id is None:
            return False, f"could not parse job id from: {result.stdout.strip()}", None
        return True, f"submitted {job_id}", job_id

    def get_job_status(self, job_id: str) -> Optional[JobStatus]:
        result = self.runner.run(f"squeue -j {shlex.quote(job_id)} -h -o '%i|%T'")
        if result.ok:
            for line in result.stdout.strip().splitlines():
                parts = line.split("|")
                if len(parts) >= 2 and parts[0] == job_id:
                    return self._map(parts[1])
        # squeue lags for finished jobs — fall back to sacct
        result = self.runner.run(
            f"sacct -j {shlex.quote(job_id)} -n -o 'JobID,State' --parsable2"
        )
        if not result.ok:
            return None
        for line in result.stdout.strip().splitlines():
            if "|" not in line or "." in line.split("|")[0]:
                continue
            jid, raw_state = line.split("|", 1)
            if jid == job_id:
                return self._map(raw_state.split("|")[0].strip())
        return None

    def cancel_job(self, job_id: str) -> Tuple[bool, str]:
        result = self.runner.run(f"scancel {shlex.quote(job_id)}")
        return result.ok, "cancelled" if result.ok else result.stderr.strip()

    @classmethod
    def _map(cls, state: str) -> JobStatus:
        token = state.split()[0].rstrip("+") if state.split() else ""
        return cls.STATUS_MAP.get(token, JobStatus.FAILED)


class PbsScheduler(SchedulerInterface):
    def __init__(self, runner: CommandRunner):
        self.runner = runner

    STATUS_MAP = {
        "Q": JobStatus.PENDING, "H": JobStatus.PENDING,
        "W": JobStatus.PENDING, "T": JobStatus.PENDING,
        "R": JobStatus.RUNNING, "E": JobStatus.RUNNING, "S": JobStatus.RUNNING,
        "C": JobStatus.COMPLETED,
        "F": JobStatus.FAILED,
    }

    def submit_job(self, script_text, work_dir, job_name, *, partition=None, nodes=1,
                   ntasks=8, cpus_per_task=1, time_limit="24:00:00", memory=None):
        script_name = f"submit_{job_name}_{int(time.time())}.sh"
        script_path = f"{work_dir}/{script_name}"
        heredoc = (
            f"mkdir -p {shlex.quote(work_dir)} && cat > {shlex.quote(script_path)} "
            f"<< 'DFTFORGE_EOF'\n{script_text}\nDFTFORGE_EOF"
        )
        result = self.runner.run(heredoc)
        if not result.ok:
            return False, f"failed to write script: {result.stderr.strip()}", None
        result = self.runner.run(
            f"cd {shlex.quote(work_dir)} && qsub {shlex.quote(script_name)}"
        )
        if not result.ok or not result.stdout.strip():
            return False, result.stderr.strip() or "qsub failed", None
        return True, "submitted", result.stdout.strip().split(".")[0]

    def get_job_status(self, job_id: str) -> Optional[JobStatus]:
        result = self.runner.run(f"qstat -f {shlex.quote(job_id)}")
        if not result.ok:
            return None
        for line in result.stdout.splitlines():
            if "job_state" in line and "=" in line:
                state = line.split("=", 1)[1].strip()[:1]
                return self.STATUS_MAP.get(state, JobStatus.FAILED)
        return None

    def cancel_job(self, job_id: str) -> Tuple[bool, str]:
        result = self.runner.run(f"qdel {shlex.quote(job_id)}")
        return result.ok, "cancelled" if result.ok else result.stderr.strip()


def build_sbatch_for_qe(
    runner: CommandRunner,
    work_dir: str,
    input_file: str,
    output_file: str,
    *,
    job_name: str = "dftforge_qe",
    params: Optional[dict] = None,
) -> Tuple[bool, str, Optional[str]]:
    """Convenience: render + submit one QE pw.x job to SLURM."""
    params = dict(params or {})
    params.setdefault("input_file", input_file)
    params.setdefault("output_file", output_file)
    script = render_sbatch(params, job_name=job_name, work_dir=work_dir, engine="qe")
    return SlurmScheduler(runner).submit_job(
        script,
        work_dir,
        job_name,
        partition=params.get("partition") or None,
        nodes=int(params.get("nodes", 1)),
        ntasks=int(params.get("ntasks", 8)),
        cpus_per_task=int(params.get("cpus_per_task", 1)),
        time_limit=str(params.get("walltime", "24:00:00")),
    )
