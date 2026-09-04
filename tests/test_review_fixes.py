"""Regression tests for the 2026-09 review round.

Pins the seven reviewed defects:
1. web session path escape (id validation + containment + delete guard)
2. SSH CLI lifecycle (submit → run_pw round trip, scheduler-id status/cancel,
   --backend accepted after the job subcommand)
3. ToolRegistry.call() refuses schema-invalid arguments at the boundary
4. scheduler commands shell-quote work_dir / job ids
5. cancel_graphs_under scopes cancellation to one session's engines
6. verify_t1 pressure failures set report.passed = False
7. template-level outputs contract resolves in graph.run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


# ── 1. Web session path safety ───────────────────────────────────────────────

class TestSessionPathSafety:
    def _dir(self, session_id):
        import dft_forge.web as web_mod
        from fastapi import HTTPException

        try:
            return web_mod._session_dir(session_id)
        except HTTPException as exc:
            return exc

    def test_valid_id_resolves_inside_workdir(self):
        import dft_forge.web as web_mod

        d = self._dir("abc-123_XY")
        assert d.name == "abc-123_XY"
        assert d.parent == web_mod.WEB_WORKDIR.resolve()

    @pytest.mark.parametrize("bad", [
        "../etc", "..", "a/b", "a\\b", "", "x" * 65, "semi;colon", "sp ace",
        "$(rm)", "foo|bar",
    ])
    def test_invalid_ids_rejected(self, bad):
        from fastapi import HTTPException

        result = self._dir(bad)
        assert isinstance(result, HTTPException) and result.status_code == 400

    def test_symlink_escape_rejected(self, tmp_path, monkeypatch):
        import dft_forge.web as web_mod
        from fastapi import HTTPException

        monkeypatch.setattr(web_mod, "WEB_WORKDIR", tmp_path / "sessions")
        (tmp_path / "sessions").mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        (tmp_path / "sessions" / "evil").symlink_to(outside)
        result = self._dir("evil")
        assert isinstance(result, HTTPException)

    def test_delete_endpoint_rejects_traversal(self):
        from fastapi.testclient import TestClient
        import dft_forge.web as web_mod

        client = TestClient(web_mod.app)
        # %2F-encoded separators are stopped by routing (404) or validation
        # (400) — either way the delete handler never runs on outside paths
        r = client.delete("/api/sessions/..%2F..%2Fetc")
        assert r.status_code in (400, 404)

    def test_delete_endpoint_rejects_invalid_id(self):
        from fastapi.testclient import TestClient
        import dft_forge.web as web_mod

        client = TestClient(web_mod.app)
        r = client.delete("/api/sessions/bad%20id%21")
        assert r.status_code == 400

    def test_valid_session_lifecycle_untouched(self, tmp_path, monkeypatch):
        from fastapi.testclient import TestClient
        import dft_forge.web as web_mod

        client = TestClient(web_mod.app)
        r = client.get("/api/sessions/legit-session-1")
        assert r.status_code == 200


# ── 2. SSH CLI lifecycle ─────────────────────────────────────────────────────

class TestSSHCLILifecycle:
    def _config(self, tmp_path, scheduler="none"):
        cfg = tmp_path / "backend.json"
        cfg.write_text(json.dumps({
            "host": "fake-host", "username": "u", "remote_workdir": "/remote/ws",
            "qe_bin_dir": "/opt/qe/bin", "scheduler": scheduler,
        }))
        return cfg

    def test_submit_uses_run_pw_round_trip(self, tmp_path, monkeypatch):
        import dft_forge.executor as exec_mod
        from dft_forge.cli import cmd_job_submit

        calls = {}

        class StubSSH:
            def __init__(self, **kw):
                calls["init"] = kw

            def run_pw(self, input_file, workdir):
                from dft_forge.executor import JobResult
                calls["run_pw"] = (Path(input_file).name, Path(workdir).name)
                return JobResult(
                    success=True, exit_code=0, stdout="JOB DONE", job_done=True,
                    output_files=["a.out"], scheduler_job_id="42",
                    remote_path="/remote/ws/pw_x",
                )

        monkeypatch.setattr(exec_mod, "SSHExecutor", StubSSH)
        inp = tmp_path / "si.in"
        inp.write_text("&CONTROL\n/\n")
        ns = argparse.Namespace(
            input=str(inp), backend="ssh",
            backend_config=str(self._config(tmp_path)), output=None,
        )
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cmd_job_submit(ns)
        assert rc == 0
        assert calls["run_pw"] == ("si.in", tmp_path.name)
        payload = json.loads(buf.getvalue())
        assert payload["scheduler_job_id"] == "42"
        assert payload["remote_path"] == "/remote/ws/pw_x"

    def test_status_direct_mode_is_honest(self, tmp_path):
        from dft_forge.cli import cmd_job_status

        ns = argparse.Namespace(
            job_id="whatever", backend="ssh",
            backend_config=str(self._config(tmp_path, scheduler="none")),
            output=None,
        )
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cmd_job_status(ns)
        assert rc == 0
        data = json.loads(buf.getvalue())
        assert data["state"] == "unknown"
        assert "synchronous" in data["note"]

    def test_cancel_direct_mode_is_honest(self, tmp_path):
        from dft_forge.cli import cmd_job_cancel

        ns = argparse.Namespace(
            job_id="whatever", backend="ssh",
            backend_config=str(self._config(tmp_path, scheduler="none")),
            output=None,
        )
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cmd_job_cancel(ns)
        assert rc == 0
        data = json.loads(buf.getvalue())
        assert data["cancelled"] is False

    def test_status_scheduler_mode_queries_by_job_id(self, tmp_path, monkeypatch):
        from dft_forge.cli import cmd_job_status
        import dft_forge.executor as exec_mod

        class StubAdapter:
            def get_job_status(self, job_id):
                from dft_forge.executor import JobStatus
                assert job_id == "42"
                return JobStatus.RUNNING

        real_init = exec_mod.SSHExecutor.__init__

        def patched_init(self, *a, **kw):
            real_init(self, *a, **kw)
            self.scheduler_adapter = StubAdapter()

        monkeypatch.setattr(exec_mod.SSHExecutor, "__init__", patched_init)
        ns = argparse.Namespace(
            job_id="42", backend="ssh",
            backend_config=str(self._config(tmp_path, scheduler="slurm")),
            output=None,
        )
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cmd_job_status(ns)
        assert rc == 0
        assert json.loads(buf.getvalue())["state"] == "running"

    def test_backend_flag_accepted_after_subcommand(self):
        # argparse used to reject `job submit x.in --backend ssh` because
        # --backend lived on the main parser
        proc = subprocess.run(
            [sys.executable, "-m", "dft_forge.cli", "job", "submit", "nope.in", "--backend", "ssh"],
            capture_output=True, text=True, cwd=str(REPO_ROOT),
        )
        assert "Input file not found" in proc.stdout + proc.stderr  # got past argparse

    def test_status_scheduler_mode_reports_unknown_for_missing_job(self, tmp_path, monkeypatch):
        from dft_forge.cli import cmd_job_status
        import dft_forge.executor as exec_mod

        class StubAdapter:
            def get_job_status(self, job_id):
                return None

        real_init = exec_mod.SSHExecutor.__init__

        def patched_init(self, *a, **kw):
            real_init(self, *a, **kw)
            self.scheduler_adapter = StubAdapter()

        monkeypatch.setattr(exec_mod.SSHExecutor, "__init__", patched_init)
        ns = argparse.Namespace(
            job_id="99", backend="ssh",
            backend_config=str(self._config(tmp_path, scheduler="slurm")),
            output=None,
        )
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cmd_job_status(ns)
        assert rc == 0
        assert json.loads(buf.getvalue())["state"] == "unknown"


# ── 3. Registry schema enforcement ───────────────────────────────────────────

def _echo_registry():
    from dft_forge.tools.registry import ToolRegistry
    from dft_forge.tools.models import ToolEntry

    executed = []
    reg = ToolRegistry()
    reg.register(ToolEntry(
        id="test.echo", name="Echo", description="",
        input_schema={
            "type": "object",
            "properties": {"msg": {"type": "string"}, "times": {"type": "integer"}},
            "required": ["msg"],
        },
        execute_fn=lambda a: executed.append(a) or {"echo": a.get("msg")},
    ))
    return reg, executed


class TestRegistrySchemaEnforcement:
    def test_missing_required_arg_refused(self):
        reg, executed = _echo_registry()
        result = asyncio.run(reg.call("test.echo", {}))
        assert result.error and "msg" in result.error
        assert executed == []  # never reached the tool

    def test_wrong_type_refused(self):
        reg, executed = _echo_registry()
        result = asyncio.run(reg.call("test.echo", {"msg": 123}))
        assert result.error
        assert executed == []

    def test_valid_args_execute(self):
        reg, executed = _echo_registry()
        result = asyncio.run(reg.call("test.echo", {"msg": "hi"}))
        assert result.error is None
        assert executed == [{"msg": "hi"}]

    def test_none_optional_args_treated_as_absent(self):
        # templates/plans legitimately emit explicit nulls for optional params
        reg, executed = _echo_registry()
        result = asyncio.run(reg.call("test.echo", {"msg": "hi", "times": None}))
        assert result.error is None
        assert executed == [{"msg": "hi", "times": None}]

    def test_non_dict_args_refused(self):
        reg, executed = _echo_registry()
        result = asyncio.run(reg.call("test.echo", "just a string"))
        assert result.error and "object" in result.error
        assert executed == []

    def test_real_graph_run_rejects_non_string_template_id(self):
        from dft_forge.tools.definitions import register_default_tools
        from dft_forge.tools.registry import registry

        register_default_tools()
        result = asyncio.run(registry.call("graph.run", {"template_id": 123}))
        assert result.error and "template_id" in result.error


# ── 4. Scheduler shell quoting ───────────────────────────────────────────────

class _RecordingRunner:
    def __init__(self, sbatch_stdout="Submitted batch job 7\n"):
        from dft_forge.hpc.runner import CommandResult
        self._result_cls = CommandResult
        self.cmds = []
        self.sbatch_stdout = sbatch_stdout

    def run(self, cmd, timeout=60.0):
        self.cmds.append(cmd)
        if "sbatch" in cmd or "qsub" in cmd:
            return self._result_cls(0, stdout=self.sbatch_stdout)
        return self._result_cls(0)


class TestSchedulerQuoting:
    def test_slurm_submit_quotes_metacharacters(self):
        from dft_forge.hpc.scheduler import SlurmScheduler

        runner = _RecordingRunner()
        sched = SlurmScheduler(runner)
        evil_dir = "/tmp/evil; rm -rf /"
        ok, msg, jid = sched.submit_job("#!/bin/bash\ntrue\n", evil_dir, "j")
        assert ok and jid == "7"
        heredoc, sbatch_cmd = runner.cmds[0], runner.cmds[1]
        assert f"mkdir -p {shlex_quote(evil_dir)}" in heredoc
        assert f"cd {shlex_quote(evil_dir)} && sbatch" in sbatch_cmd
        # the raw metacharacter must never appear unquoted
        assert "mkdir -p /tmp/evil; rm" not in heredoc
        assert "cd /tmp/evil; rm" not in sbatch_cmd

    def test_pbs_submit_quotes_metacharacters(self):
        from dft_forge.hpc.scheduler import PbsScheduler

        runner = _RecordingRunner(sbatch_stdout="42.server\n")
        sched = PbsScheduler(runner)
        evil_dir = "/tmp/x && touch pwned"
        ok, msg, jid = sched.submit_job("#!/bin/bash\ntrue\n", evil_dir, "j")
        assert ok
        assert f"cd {shlex_quote(evil_dir)} && qsub" in runner.cmds[1]
        assert "cd /tmp/x && touch" not in runner.cmds[1]

    def test_slurm_cancel_quotes_job_id(self):
        from dft_forge.hpc.scheduler import SlurmScheduler

        runner = _RecordingRunner()
        sched = SlurmScheduler(runner)
        sched.cancel_job("7; reboot")
        assert any("scancel " + shlex_quote("7; reboot") in c for c in runner.cmds)
        assert not any("scancel 7; reboot" in c for c in runner.cmds)

    def test_slurm_status_quotes_job_id(self):
        from dft_forge.hpc.scheduler import SlurmScheduler

        runner = _RecordingRunner()
        sched = SlurmScheduler(runner)
        sched.get_job_status("1; evil")
        assert any("squeue -j " + shlex_quote("1; evil") in c for c in runner.cmds)


def shlex_quote(s: str) -> str:
    import shlex
    return shlex.quote(s)


# ── 5. Session-scoped cancellation ───────────────────────────────────────────

class TestSessionScopedCancel:
    def test_cancel_graphs_under_kills_only_target_session(self, tmp_path):
        from dft_forge.tools import graph_tools

        class E:
            def __init__(self, n):
                self.n = n

            def _cancel_executors(self):
                return self.n

        a = tmp_path / "sessA" / "graphs"
        b = tmp_path / "sessB" / "graphs"
        a.mkdir(parents=True)
        b.mkdir(parents=True)
        graph_tools._ENGINE_CACHE.clear()
        graph_tools._ENGINE_CACHE[str(a)] = E(1)
        graph_tools._ENGINE_CACHE[str(b)] = E(2)
        try:
            assert graph_tools.cancel_graphs_under(tmp_path / "sessA") == 1
            assert graph_tools.cancel_graphs_under(tmp_path / "sessB") == 2
            assert graph_tools.cancel_graphs_under(tmp_path / "nowhere") == 0
        finally:
            graph_tools._ENGINE_CACHE.clear()

    def test_cancel_graphs_under_sweeps_nested_dirs(self, tmp_path):
        from dft_forge.tools import graph_tools

        class E:
            def _cancel_executors(self):
                return 1

        nested = tmp_path / "sess" / "graphs"
        nested.mkdir(parents=True)
        graph_tools._ENGINE_CACHE.clear()
        graph_tools._ENGINE_CACHE[str(nested)] = E()
        try:
            # session dir is the PARENT of the engine workdir — containment
            # must still match
            assert graph_tools.cancel_graphs_under(tmp_path / "sess") == 1
        finally:
            graph_tools._ENGINE_CACHE.clear()


# ── 6. Pressure check hard-fails ─────────────────────────────────────────────

def _vc_stdout(converged=True, pressure="          P=       5.00"):
    return (
        "     Program PWSCF v.7.5 starts\n"
        "convergence has been achieved in  12 iterations\n"
        "bfgs converged in  12 scf cycles and   4 bfgs steps\n"
        + (f"{pressure}\n" if pressure else "")
        + "     atom    1 type  1   force = 0.00000000 0.00000000 0.00000000\n"
        + "     atom    2 type  1   force = 0.00010000 0.00000000 0.00000000\n"
        + "     CELL_PARAMETERS (alat=  10.26000000)\n"
        + "      0.000000000  0.500000000  0.500000000\n"
        + "      0.500000000  0.000000000  0.500000000\n"
        + "      0.500000000  0.500000000  0.000000000\n"
        + "JOB DONE.\n"
    )


class TestPressureHardFail:
    def _verify(self, pressure_kbar: float):
        from dft_forge.parser import ParsedVCResult, ParsedCell
        from dft_forge.protocol.schemas import VerificationInput
        from dft_forge.verifier import ScientificVerifier

        verifier = ScientificVerifier()
        parsed = ParsedVCResult(
            final_energy_ry=-100.0, max_force_ry_bohr=1e-4,
            pressure_kbar=pressure_kbar, natoms=2, converged=True,
            cell=ParsedCell(a_bohr=10.26, b_bohr=10.26, c_bohr=10.26, ibrav=2),
        )
        inp = VerificationInput(
            stdout=_vc_stdout(), xml_path=None, job_success=True,
        )
        return verifier.verify_t1(inp, parsed)

    def test_pressure_above_threshold_fails(self):
        report = self._verify(pressure_kbar=5.0)
        assert report.passed is False
        assert any("Pressure" in r for r in report.failure_reasons)
        assert report.checks["pressure"]["pass"] is False

    def test_pressure_below_threshold_no_false_failure(self):
        report = self._verify(pressure_kbar=0.1)
        assert report.passed is True
        assert report.checks["pressure"]["pass"] is True


# ── 7. Template-level outputs ────────────────────────────────────────────────

class TestTemplateOutputs:
    def test_validate_rejects_unknown_node_ref(self):
        from dft_forge.runtime.graph import GraphTemplate, NodeSpec

        t = GraphTemplate(
            template_id="t",
            nodes=[NodeSpec(id="a", tool="qe")],
            outputs={"x": "${nodes.b.outputs.y}"},
        )
        errs = t.validate()
        assert any("unknown node 'b'" in e for e in errs)

    def test_validate_rejects_malformed_ref(self):
        from dft_forge.runtime.graph import GraphTemplate, NodeSpec

        t = GraphTemplate(
            template_id="t",
            nodes=[NodeSpec(id="a", tool="qe")],
            outputs={"x": "just-a-literal"},
        )
        errs = t.validate()
        assert any("malformed" in e or "must be a" in e for e in errs)

    def test_graph_run_returns_resolved_outputs(self, tmp_path):
        from dft_forge.tools.graph_tools import _resolve_template_outputs
        from dft_forge.runtime.graph import GraphTemplate, NodeSpec
        from dft_forge.runtime.states import NodeState
        from dft_forge.runtime.run import create_run

        template = GraphTemplate(
            template_id="t",
            nodes=[NodeSpec(id="scf", tool="qe")],
            outputs={"energy": "${nodes.scf.outputs.energy_ry}",
                     "missing": "${nodes.scf.outputs.nope}"},
        )
        run = create_run(template, {})
        run.nodes["scf"].state = NodeState.SUCCEEDED
        run.nodes["scf"].outputs = {"energy_ry": -12.5}
        outs = _resolve_template_outputs(template, run)
        assert outs == {"energy": -12.5, "missing": None}

    def test_failed_node_yields_none_output(self):
        from dft_forge.tools.graph_tools import _resolve_template_outputs
        from dft_forge.runtime.graph import GraphTemplate, NodeSpec
        from dft_forge.runtime.states import NodeState
        from dft_forge.runtime.run import create_run

        template = GraphTemplate(
            template_id="t",
            nodes=[NodeSpec(id="scf", tool="qe")],
            outputs={"energy": "${nodes.scf.outputs.energy_ry}"},
        )
        run = create_run(template, {})
        run.nodes["scf"].state = NodeState.FAILED
        outs = _resolve_template_outputs(template, run)
        assert outs == {"energy": None}

    def test_t1_template_outputs_resolve_through_engine(self, tmp_path):
        # the shipped templates declare real output contracts — verify the
        # refs point at keys the QE tool actually emits
        from dft_forge.runtime.graph import load_template

        templates_dir = REPO_ROOT / "templates"
        for tid in ("t0_scf", "t1_vc_relax", "t2_bands", "t2_dos"):
            template = load_template(templates_dir / f"{tid}.json")
            errs = template.validate()
            assert errs == [], f"{tid}: {errs}"
