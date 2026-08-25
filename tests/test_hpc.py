"""Tests for the HPC layer: job scripts, SLURM/PBS schedulers with a fake runner."""

import pytest

from dft_forge.executor import JobStatus
from dft_forge.hpc.job_script import render_qsub, render_sbatch
from dft_forge.hpc.runner import CommandResult, LocalRunner
from dft_forge.hpc.scheduler import PbsScheduler, SlurmScheduler


class FakeRunner:
    """Records commands; replies from a scripted queue or a callable."""

    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.calls = []

    def run(self, cmd, timeout=60.0):
        self.calls.append(cmd)
        if self.responses:
            resp = self.responses.pop(0)
            if callable(resp):
                return resp(cmd)
            return resp
        return CommandResult(0, "", "")


class TestJobScript:
    def test_sbatch_defaults(self):
        script = render_sbatch({}, job_name="nacl", work_dir="/scratch/x")
        assert "#SBATCH --job-name=nacl" in script
        assert "#SBATCH --ntasks=8" in script
        assert "cd /scratch/x" in script
        assert "srun pw.x -in pw.in > pw.out" in script

    def test_sbatch_param_priority(self):
        script = render_sbatch(
            {"ntasks": 16, "walltime": "01:00:00", "partition": "gpu"},
            {"ntasks": 4, "walltime": "12:00:00"},
        )
        assert "#SBATCH --ntasks=16" in script
        assert "#SBATCH --time=01:00:00" in script
        assert "#SBATCH --partition=gpu" in script

    def test_sbatch_defaults_fallback(self):
        script = render_sbatch({}, {"ntasks": 4, "account": "mygrant"})
        assert "#SBATCH --ntasks=4" in script
        assert "#SBATCH --account=mygrant" in script

    def test_empty_optionals_stripped(self):
        script = render_sbatch({})
        assert "--account" not in script
        assert "--mem" not in script

    def test_custom_run_command(self):
        script = render_sbatch({"run_command": "srun pw.x -in scf.in > scf.out"})
        assert "srun pw.x -in scf.in > scf.out" in script

    def test_input_output_files(self):
        script = render_sbatch({"input_file": "nacl_scf.in", "output_file": "nacl_scf.out"})
        assert "srun pw.x -in nacl_scf.in > nacl_scf.out" in script

    def test_qsub_render(self):
        script = render_qsub({"cpus_per_task": 4}, job_name="si", work_dir="/work")
        assert "#PBS -N si" in script
        assert "#PBS -l nodes=1:ppn=4" in script
        assert "cd /work" in script

    def test_ppn_alias(self):
        assert "#SBATCH --cpus-per-task=8" in render_sbatch({"ppn": 8})


class TestSlurmScheduler:
    def test_submit_parses_job_id(self):
        runner = FakeRunner([
            CommandResult(0),  # heredoc write
            CommandResult(0, "Submitted batch job 123456\n"),
        ])
        ok, msg, job_id = SlurmScheduler(runner).submit_job("#!/bin/bash\ntrue", "/w", "nacl")
        assert ok and job_id == "123456"
        assert "sbatch" in runner.calls[1]

    def test_submit_failure(self):
        runner = FakeRunner([
            CommandResult(0),
            CommandResult(1, stderr="sbatch: error\n"),
        ])
        ok, msg, job_id = SlurmScheduler(runner).submit_job("x", "/w", "j")
        assert not ok and job_id is None and "error" in msg

    def test_status_from_squeue(self):
        runner = FakeRunner([CommandResult(0, "123456|RUNNING\n")])
        assert SlurmScheduler(runner).get_job_status("123456") == JobStatus.RUNNING

    def test_status_squeue_to_sacct_fallback(self):
        # squeue empty (job finished) -> sacct reports COMPLETED
        runner = FakeRunner([
            CommandResult(0, ""),
            CommandResult(0, "123456|COMPLETED|\n123456.batch|COMPLETED|\n"),
        ])
        assert SlurmScheduler(runner).get_job_status("123456") == JobStatus.COMPLETED
        assert "sacct" in runner.calls[1]

    def test_status_skips_substeps(self):
        runner = FakeRunner([
            CommandResult(0, ""),
            CommandResult(0, "123456.external|FAILED\n123456|FAILED|\n"),
        ])
        assert SlurmScheduler(runner).get_job_status("123456") == JobStatus.FAILED

    @pytest.mark.parametrize("raw,expected", [
        ("RUNNING", JobStatus.RUNNING),
        ("PENDING", JobStatus.PENDING),
        ("CANCELLED by 1234", JobStatus.CANCELLED),
        ("TIMEOUT", JobStatus.FAILED),
        ("OUT_OF_MEMORY", JobStatus.FAILED),
        ("COMPLETING+", JobStatus.RUNNING),
        ("NODE_FAIL", JobStatus.FAILED),
    ])
    def test_status_mapping(self, raw, expected):
        assert SlurmScheduler._map(raw) == expected

    def test_cancel(self):
        runner = FakeRunner([CommandResult(0, "")])
        ok, msg = SlurmScheduler(runner).cancel_job("42")
        assert ok and "scancel 42" in runner.calls[0]


class TestPbsScheduler:
    def test_submit(self):
        runner = FakeRunner([
            CommandResult(0),
            CommandResult(0, "12345.server\n"),
        ])
        ok, msg, job_id = PbsScheduler(runner).submit_job("x", "/w", "j")
        assert ok and job_id == "12345"
        assert "qsub" in runner.calls[1]

    def test_status(self):
        runner = FakeRunner([CommandResult(0, "    job_state = R\n")])
        assert PbsScheduler(runner).get_job_status("1") == JobStatus.RUNNING

    def test_cancel(self):
        runner = FakeRunner([CommandResult(0)])
        assert PbsScheduler(runner).cancel_job("1")[0]


class TestRunners:
    def test_local_runner_echo(self):
        result = LocalRunner().run("echo hello-graph")
        assert result.ok and "hello-graph" in result.stdout

    def test_local_runner_login_shell(self):
        result = LocalRunner().run("echo $0")
        assert "bash" in result.stdout
