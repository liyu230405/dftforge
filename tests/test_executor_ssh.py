"""SSHExecutor semantics tests against the REAL transport protocol.

SSHExecutor is grafted onto ``hpc/runner.SSHRunner``: every remote operation
is a shell command over the runner's text channel.  ``FakeRemoteHost``
implements just enough of that wire protocol — tar+base64 chunked transport,
timeout-wrapped direct runs, sbatch/qsub + squeue/sacct/qstat/scancel — backed
by a real local directory, so uploads and downloads genuinely round-trip and
no network is ever touched.

These tests pin:
- upload: chunked base64+tar, symlink dereference (-h), big payloads
- direct mode (scheduler="none"): timeout wrapper, exit-code marker,
  stem-named outputs, partial-output download on failure
- scheduler mode: sbatch/qsub script rendering, poll loop, scancel/qdel
- cloud lifecycle: stage → submit → status → fetch → cancel
"""

from __future__ import annotations

import base64
import io
import re
import tarfile
from pathlib import Path
from typing import List, Optional

import pytest

from dft_forge.executor import JobHandle, JobStatus, SSHExecutor
from dft_forge.hpc.runner import CommandResult, CommandRunner


# ── Fake remote host ──────────────────────────────────────────────────────────

def _unq(tok: str) -> str:
    """Strip the single quotes shlex.quote() may have added."""
    if len(tok) >= 2 and tok[0] == "'" and tok[-1] == "'":
        return tok[1:-1]
    return tok


def _safe_extract(tf: tarfile.TarFile, dest: Path) -> None:
    try:
        tf.extractall(dest, filter="data")
    except TypeError:  # Python < 3.12 has no filter kwarg
        tf.extractall(dest)


# shlex.quote() emits bare tokens for [A-Za-z0-9_@%+=:,./-]+ and 'quoted'
# tokens otherwise — patterns accept both forms.
_TOK = r"(?:'[^']*'|\S+)"

_UPLOAD_PREP_RE = re.compile(rf"^mkdir -p ({_TOK}) && rm -f ({_TOK})$")
_UPLOAD_CHUNK_RE = re.compile(rf"^printf '%s' ({_TOK}) >> ({_TOK})$")
_UPLOAD_UNTAR_RE = re.compile(rf"^base64 -d ({_TOK}) \| tar -C ({_TOK}) -xf - && rm -f ({_TOK})$")
_DIRECT_RUN_RE = re.compile(
    rf"^cd ({_TOK}) && timeout (?P<secs>\d+) ({_TOK}) -in ({_TOK}) "
    rf"> ({_TOK})\.out 2> ({_TOK})\.err; echo DFTFORGE_EXIT_\$\?$"
)
_DOWNLOAD_RE = re.compile(rf"^tar -C ({_TOK}) -cf - \. 2>/dev/null \| base64$")
_SQUEUE_RE = re.compile(r"^squeue -j (?P<job>\d+) -h -o '%i\|%T'$")
_SACCT_RE = re.compile(r"^sacct -j (?P<job>\d+) -n -o 'JobID,State' --parsable2$")
_QSTAT_RE = re.compile(r"^qstat -f (\S+)$")
_BATCH_RE = re.compile(r"^cd (\S+) && (sbatch|qsub) (\S+)$")
_CANCEL_RE = re.compile(r"^(?:scancel|qdel) (\S+)$")
_PKILL_RE = re.compile(rf"^pkill -f ({_TOK}); true$")


class FakeRemoteHost(CommandRunner):
    """Scriptable stand-in for the SSH remote side."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.commands: List[str] = []
        self.pending_b64 = {}  # archive path -> b64 chunks
        self.direct_exit_codes: List[int] = []  # consumed per direct run
        self.direct_stdout_files = {}  # stem -> .out content written on direct run
        self.sbatch_stdout_files = {}  # output name -> content written on submit
        self.squeue_states: List[str] = []  # consumed per squeue call
        self.sacct_state = "COMPLETED"
        self.qstat_state = "C"
        self.job_id = "42"
        self.cancelled_jobs: List[str] = []
        self.pkilled: List[str] = []
        self.submitted_scripts = {}  # script path -> text
        self.fail_substr: Optional[str] = None  # commands containing it fail

    def _local(self, remote_path: str) -> Path:
        return self.root / remote_path.lstrip("/")

    def run(self, cmd: str, timeout: float = 60.0) -> CommandResult:
        self.commands.append(cmd)
        if self.fail_substr and self.fail_substr in cmd:
            return CommandResult(1, stderr="fake remote: scripted failure")

        m = _UPLOAD_PREP_RE.match(cmd)
        if m:
            self._local(_unq(m.group(1))).mkdir(parents=True, exist_ok=True)
            self.pending_b64[_unq(m.group(2))] = []
            return CommandResult(0)

        m = _UPLOAD_CHUNK_RE.match(cmd)
        if m:
            self.pending_b64.setdefault(_unq(m.group(2)), []).append(_unq(m.group(1)))
            return CommandResult(0)

        m = _UPLOAD_UNTAR_RE.match(cmd)
        if m:
            dest = self._local(_unq(m.group(2)))
            if not dest.is_dir():
                return CommandResult(1, stderr="fake remote: no such directory")
            payload = base64.b64decode("".join(self.pending_b64.get(_unq(m.group(1)), [])))
            with tarfile.open(fileobj=io.BytesIO(payload)) as tf:
                _safe_extract(tf, dest)
            return CommandResult(0)

        if "<< 'DFTFORGE_EOF'" in cmd:
            header, _, body = cmd.partition("<< 'DFTFORGE_EOF'\n")
            script = body.rsplit("DFTFORGE_EOF", 1)[0].rstrip("\n")
            hm = re.match(r"^mkdir -p (\S+) && cat > (\S+)\s*$", header)
            if not hm:
                return CommandResult(1, stderr="fake remote: bad heredoc header")
            path = self._local(hm.group(2))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(script)
            self.submitted_scripts[hm.group(2)] = script
            self._fake_run_engine(script, hm.group(1))
            return CommandResult(0)

        m = _BATCH_RE.match(cmd)
        if m:
            if not (self._local(m.group(1)) / m.group(3)).exists():
                return CommandResult(1, stderr=f"fake remote: {m.group(2)} script not staged")
            if m.group(2) == "sbatch":
                return CommandResult(0, stdout=f"Submitted batch job {self.job_id}\n")
            return CommandResult(0, stdout=f"{self.job_id}.server\n")

        m = _SQUEUE_RE.match(cmd)
        if m:
            if self.squeue_states:
                return CommandResult(0, stdout=f"{m['job']}|{self.squeue_states.pop(0)}\n")
            return CommandResult(0, stdout="")  # job left the queue

        m = _SACCT_RE.match(cmd)
        if m:
            return CommandResult(0, stdout=f"{m['job']}|{self.sacct_state}\n")

        m = _QSTAT_RE.match(cmd)
        if m:
            return CommandResult(0, stdout=f"    job_state = {self.qstat_state}\n")

        m = _CANCEL_RE.match(cmd)
        if m:
            self.cancelled_jobs.append(m.group(1))
            return CommandResult(0)

        m = _PKILL_RE.match(cmd)
        if m:
            self.pkilled.append(_unq(m.group(1)))
            return CommandResult(0)

        m = _DIRECT_RUN_RE.match(cmd)
        if m:
            code = self.direct_exit_codes.pop(0) if self.direct_exit_codes else 0
            # groups: 1=dir, 2=secs(named), 3=binary, 4=input, 5=out-stem, 6=err-stem
            out_stem, err_stem = _unq(m.group(5)), _unq(m.group(6))
            job_dir = self._local(_unq(m.group(1)))
            job_dir.mkdir(parents=True, exist_ok=True)
            (job_dir / f"{out_stem}.out").write_text(
                self.direct_stdout_files.get(out_stem, "JOB DONE.\n")
            )
            (job_dir / f"{err_stem}.err").write_text("")
            return CommandResult(0, stdout=f"DFTFORGE_EXIT_{code}\n")

        m = _DOWNLOAD_RE.match(cmd)
        if m:
            src = self._local(_unq(m.group(1)))
            if not src.is_dir():
                return CommandResult(1, stderr="fake remote: no such directory")
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w") as tf:
                tf.add(str(src), arcname=".")
            return CommandResult(0, stdout=base64.b64encode(buf.getvalue()).decode("ascii"))

        return CommandResult(127, stderr=f"fake remote: unhandled command: {cmd[:120]}")

    def _fake_run_engine(self, script: str, work_dir: str) -> None:
        """The submitted job 'runs': its redirected output lands in the job dir."""
        m = re.search(r"-in (\S+) > (\S+)", script)
        if not m:
            return
        out_name = m.group(2)
        (self._local(work_dir) / out_name).write_text(
            self.sbatch_stdout_files.get(out_name, "JOB DONE.\n")
        )


# ── Fixtures / helpers ────────────────────────────────────────────────────────

@pytest.fixture
def fake_remote(tmp_path):
    return FakeRemoteHost(tmp_path / "remote-root")


def _direct_executor(fake, **kw):
    return SSHExecutor(
        host="fake", remote_workdir="/remote/ws", qe_bin_dir="/opt/qe/bin",
        scheduler="none", max_walltime_sec=60, runner=fake, **kw
    )


def _slurm_executor(fake, **kw):
    return SSHExecutor(
        host="fake", remote_workdir="/remote/ws", qe_bin_dir="/opt/qe/bin",
        scheduler="slurm", max_walltime_sec=60, runner=fake, **kw
    )


def _make_workdir(tmp_path, input_name="si_vcrelax.in"):
    wd = tmp_path / "work"
    wd.mkdir()
    (wd / input_name).write_text("&CONTROL\n/\n")
    return wd


# ── CLI wiring ────────────────────────────────────────────────────────────────

class TestCLIWiring:
    def test_constructor_accepts_cli_kwargs(self):
        # mirrors _resolve_executor in cli.py for backend == "ssh"
        ex = SSHExecutor(
            host="cluster.example",
            username="zlp",
            remote_workdir=Path("/home/zlp/dft"),
            ssh_key=None,
            qe_bin_dir=Path("/opt/software/q-e/bin"),
            scheduler="slurm",
            max_walltime_sec=3600,
            max_retries=1,
            port=22,
        )
        assert ex.host == "cluster.example"
        assert ex.remote_workdir == "/home/zlp/dft"
        assert ex.qe_bin_dir == "/opt/software/q-e/bin"
        assert ex.scheduler == "slurm"
        assert ex.port == 22
        assert ex.scheduler_adapter is not None

    def test_key_file_legacy_alias(self):
        ex = SSHExecutor(host="h", key_file=Path("/tmp/id_rsa"))
        assert ex.ssh_key == Path("/tmp/id_rsa")

    def test_direct_mode_has_no_scheduler_adapter(self, fake_remote):
        assert _direct_executor(fake_remote).scheduler_adapter is None

    def test_injected_runner_is_used_by_executor_and_scheduler(self, fake_remote):
        ex = _slurm_executor(fake_remote)
        assert ex.runner is fake_remote
        assert ex.scheduler_adapter.runner is fake_remote


# ── Direct mode (scheduler="none") ────────────────────────────────────────────

class TestDirectMode:
    def test_round_trip_success(self, tmp_path, fake_remote):
        ex = _direct_executor(fake_remote)
        wd = _make_workdir(tmp_path)
        res = ex.run_pw(wd / "si_vcrelax.in", wd)

        assert res.success and res.job_done
        assert "JOB DONE" in res.stdout
        # the downloaded .out landed next to the input
        assert (wd / "si_vcrelax.out").exists()
        # the upload genuinely delivered the input to the remote side
        assert (fake_remote._local(res.remote_path) / "si_vcrelax.in").exists()

    def test_output_files_are_stem_named(self, tmp_path, fake_remote):
        # regression: the redirect used to produce si_vcrelax.in.out, which
        # the post-download reader (stem.out) never found
        ex = _direct_executor(fake_remote)
        wd = _make_workdir(tmp_path)
        res = ex.run_pw(wd / "si_vcrelax.in", wd)

        remote_names = [p.name for p in fake_remote._local(res.remote_path).iterdir()]
        assert "si_vcrelax.out" in remote_names
        assert "si_vcrelax.in.out" not in remote_names
        assert "si_vcrelax.err" in remote_names

    def test_run_command_carries_walltime_wrapper(self, tmp_path, fake_remote):
        ex = _direct_executor(fake_remote)
        wd = _make_workdir(tmp_path)
        ex.run_pw(wd / "si_vcrelax.in", wd)

        run_cmd = next(c for c in fake_remote.commands if "pw.x -in" in c)
        assert "timeout 60 " in run_cmd
        assert "/opt/qe/bin/pw.x" in run_cmd
        assert "DFTFORGE_EXIT_$?" in run_cmd  # exit code travels via stdout marker

    def test_timeout_exit_124_maps_to_failure(self, tmp_path, fake_remote):
        fake_remote.direct_exit_codes = [124]
        ex = _direct_executor(fake_remote)
        wd = _make_workdir(tmp_path)
        res = ex.run_pw(wd / "si_vcrelax.in", wd)
        assert not res.success
        assert "timed out" in res.error_message

    def test_binary_failure_downloads_partial_outputs(self, tmp_path, fake_remote):
        # the .out file often holds the QE error block the repairer matches on
        fake_remote.direct_exit_codes = [1]
        fake_remote.direct_stdout_files = {"si_vcrelax": "Error in routine scf (1)\n"}
        ex = _direct_executor(fake_remote)
        wd = _make_workdir(tmp_path)
        res = ex.run_pw(wd / "si_vcrelax.in", wd)

        assert not res.success
        assert "exited 1" in res.error_message
        assert (wd / "si_vcrelax.out").read_text().startswith("Error in routine")

    def test_missing_input_short_circuits_without_network(self, tmp_path, fake_remote):
        ex = _direct_executor(fake_remote)
        res = ex.run_pw(tmp_path / "nope.in", tmp_path)
        assert not res.success
        assert "Input file not found" in res.error_message
        assert fake_remote.commands == []

    def test_non_pw_tools_do_not_require_job_done(self, tmp_path, fake_remote):
        fake_remote.direct_stdout_files = {"si_dos.dos": ""}  # dos.x prints nothing
        ex = _direct_executor(fake_remote)
        wd = _make_workdir(tmp_path, input_name="si_dos.dos.in")
        res = ex.run_dos_x(wd / "si_dos.dos.in", wd)
        assert res.success


# ── Upload semantics ──────────────────────────────────────────────────────────

class TestUploadSemantics:
    def test_save_symlinks_are_dereferenced(self, tmp_path, fake_remote):
        # chained nodes link upstream .save dirs; -h uploads real wavefunctions
        wd = _make_workdir(tmp_path)
        (wd / "upstream.save").mkdir()
        (wd / "upstream.save" / "data-file-schema.xml").write_text("<xml/>")
        (wd / "si_vcrelax.save").symlink_to("upstream.save")

        ex = _direct_executor(fake_remote)
        res = ex.run_pw(wd / "si_vcrelax.in", wd)

        remote = fake_remote._local(res.remote_path)
        assert (remote / "si_vcrelax.save" / "data-file-schema.xml").is_file()

    def test_large_payload_is_chunked_and_round_trips(self, tmp_path, fake_remote):
        wd = _make_workdir(tmp_path)
        payload = bytes(range(256)) * 800  # 200 KB → several b64 chunks
        (wd / "charge-density.dat").write_bytes(payload)

        ex = _direct_executor(fake_remote)
        res = ex.run_pw(wd / "si_vcrelax.in", wd)

        chunks = [c for c in fake_remote.commands if c.startswith("printf '%s' ")]
        assert len(chunks) >= 3
        assert (fake_remote._local(res.remote_path) / "charge-density.dat").read_bytes() == payload


# ── SLURM mode ────────────────────────────────────────────────────────────────

class TestSlurmMode:
    def test_run_pw_round_trip(self, tmp_path, fake_remote):
        fake_remote.squeue_states = ["COMPLETED"]
        ex = _slurm_executor(fake_remote)
        wd = _make_workdir(tmp_path)
        res = ex.run_pw(wd / "si_vcrelax.in", wd)

        assert res.success and res.job_done
        assert res.scheduler_job_id == "42"
        assert (wd / "si_vcrelax.out").exists()
        # sbatch ran from the uploaded job dir
        assert any(c.startswith("cd /remote/ws/") and "sbatch submit_" in c
                   for c in fake_remote.commands)

    def test_submitted_script_shape(self, tmp_path, fake_remote):
        fake_remote.squeue_states = ["COMPLETED"]
        ex = _slurm_executor(fake_remote)
        wd = _make_workdir(tmp_path)
        res = ex.run_pw(wd / "si_vcrelax.in", wd)

        script = next(iter(fake_remote.submitted_scripts.values()))
        assert "srun pw.x -in si_vcrelax.in > si_vcrelax.out" in script
        assert f"cd {res.remote_path}" in script
        assert "export PATH=/opt/qe/bin:$PATH" in script
        assert script.startswith("#!/bin/bash\n#SBATCH")

    def test_scheduler_failure_maps_to_job_result(self, tmp_path, fake_remote, monkeypatch):
        monkeypatch.setattr("dft_forge.executor._SCHED_POLL_INTERVAL", 0.01)
        fake_remote.squeue_states = ["RUNNING", "FAILED"]
        ex = _slurm_executor(fake_remote)
        wd = _make_workdir(tmp_path)
        res = ex.run_pw(wd / "si_vcrelax.in", wd)

        assert not res.success
        assert "failed" in res.error_message
        assert res.scheduler_job_id == "42"

    def test_squeue_exhausted_falls_back_to_sacct(self, tmp_path, fake_remote):
        fake_remote.squeue_states = []  # job already left the queue
        fake_remote.sacct_state = "COMPLETED"
        ex = _slurm_executor(fake_remote)
        wd = _make_workdir(tmp_path)
        res = ex.run_pw(wd / "si_vcrelax.in", wd)
        assert res.success
        assert any(c.startswith("sacct -j 42 ") for c in fake_remote.commands)


# ── PBS mode ──────────────────────────────────────────────────────────────────

class TestPbsMode:
    def test_run_pw_round_trip(self, tmp_path, fake_remote):
        ex = SSHExecutor(
            host="fake", remote_workdir="/remote/ws", qe_bin_dir="/opt/qe/bin",
            scheduler="pbs", max_walltime_sec=60, runner=fake_remote,
        )
        wd = _make_workdir(tmp_path)
        res = ex.run_pw(wd / "si_vcrelax.in", wd)

        assert res.success and res.job_done
        assert res.scheduler_job_id == "42"
        assert any("qsub submit_" in c for c in fake_remote.commands)
        script = next(iter(fake_remote.submitted_scripts.values()))
        assert script.startswith("#!/bin/bash\n#PBS")

    def test_qstat_failure_state_maps_to_failed(self, tmp_path, fake_remote):
        fake_remote.qstat_state = "F"
        ex = SSHExecutor(
            host="fake", remote_workdir="/remote/ws", qe_bin_dir="/opt/qe/bin",
            scheduler="pbs", max_walltime_sec=60, runner=fake_remote,
        )
        wd = _make_workdir(tmp_path)
        res = ex.run_pw(wd / "si_vcrelax.in", wd)
        assert not res.success
        assert "failed" in res.error_message


# ── Cloud lifecycle (stage/submit/status/fetch/cancel) ────────────────────────

class TestCloudLifecycle:
    def test_slurm_lifecycle(self, tmp_path, fake_remote):
        ex = _slurm_executor(fake_remote)
        wd = _make_workdir(tmp_path)
        spec = {"input_file": str(wd / "si_vcrelax.in"), "workdir": str(wd)}

        handle = ex.stage({"task_uuid": "job1"})
        assert handle.remote_path == "/remote/ws/job1"
        assert ex.status(handle) == JobStatus.PENDING  # staged

        fake_remote.squeue_states = ["RUNNING", "COMPLETED"]
        ex.submit(handle, spec)
        assert ex._job_registry["job1"]["scheduler_job_id"] == "42"
        assert ex.status(handle) == JobStatus.RUNNING

        early = ex.fetch(handle, tmp_path / "dest")
        assert not early.success
        assert "not completed" in early.error_message

        assert ex.status(handle) == JobStatus.COMPLETED
        got = ex.fetch(handle, tmp_path / "dest")
        assert got.success
        assert "si_vcrelax.in" in [Path(f).name for f in got.output_files]

    def test_direct_lifecycle(self, tmp_path, fake_remote):
        ex = _direct_executor(fake_remote)
        wd = _make_workdir(tmp_path)
        handle = ex.stage({"task_uuid": "job3"})
        ex.submit(handle, {"input_file": str(wd / "si_vcrelax.in"), "workdir": str(wd)})
        # direct submit blocks until the run finishes
        assert ex.status(handle) == JobStatus.COMPLETED
        got = ex.fetch(handle, tmp_path / "dest")
        assert got.success

    def test_cancel_slurm_issues_scancel(self, tmp_path, fake_remote):
        ex = _slurm_executor(fake_remote)
        wd = _make_workdir(tmp_path)
        handle = ex.stage({"task_uuid": "job2"})
        ex.submit(handle, {"input_file": str(wd / "si_vcrelax.in"), "workdir": str(wd)})

        fake_remote.squeue_states = ["CANCELLED"]
        assert ex.cancel(handle) is True
        assert fake_remote.cancelled_jobs == ["42"]
        assert ex.status(handle) == JobStatus.CANCELLED

    def test_cancel_direct_mode_pkills_job_dir(self, tmp_path, fake_remote):
        ex = _direct_executor(fake_remote)
        handle = ex.stage({"task_uuid": "job5"})
        assert ex.cancel(handle) is True
        assert fake_remote.pkilled == ["/remote/ws/job5"]
        assert ex.status(handle) == JobStatus.CANCELLED

    def test_cancel_pbs_issues_qdel(self, tmp_path, fake_remote):
        ex = SSHExecutor(
            host="fake", remote_workdir="/remote/ws", qe_bin_dir="/opt/qe/bin",
            scheduler="pbs", max_walltime_sec=60, runner=fake_remote,
        )
        wd = _make_workdir(tmp_path)
        handle = ex.stage({"task_uuid": "job6"})
        ex.submit(handle, {"input_file": str(wd / "si_vcrelax.in"), "workdir": str(wd)})
        assert ex.cancel(handle) is True
        assert fake_remote.cancelled_jobs == ["42"]

    def test_upload_failure_marks_job_failed(self, tmp_path, fake_remote):
        fake_remote.fail_substr = "mkdir -p"
        ex = _direct_executor(fake_remote)
        wd = _make_workdir(tmp_path)
        handle = ex.stage({"task_uuid": "job4"})
        with pytest.raises(Exception):
            ex.submit(handle, {"input_file": str(wd / "si_vcrelax.in"), "workdir": str(wd)})
        record = ex._job_registry["job4"]
        assert record["state"] == "failed"
        assert "upload failed" in record["error"]

    def test_scheduler_submit_failure_raises(self, tmp_path, fake_remote):
        fake_remote.fail_substr = "sbatch"
        ex = _slurm_executor(fake_remote)
        wd = _make_workdir(tmp_path)
        handle = ex.stage({"task_uuid": "job7"})
        with pytest.raises(RuntimeError, match="scheduler submit failed"):
            ex.submit(handle, {"input_file": str(wd / "si_vcrelax.in"), "workdir": str(wd)})

    def test_unknown_handle_status_is_pending(self, tmp_path, fake_remote):
        ex = _slurm_executor(fake_remote)
        assert ex.status(JobHandle(job_id="ghost", remote_path="/")) == JobStatus.PENDING

    def test_cancel_unknown_handle_returns_false(self, tmp_path, fake_remote):
        ex = _slurm_executor(fake_remote)
        assert ex.cancel(JobHandle(job_id="ghost", remote_path="/")) is False
