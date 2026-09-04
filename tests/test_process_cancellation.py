"""Task: subprocess-level cancellation (Popen tracking, kill, scheduler wiring).

Review gap: "暂停/取消不杀子进程" — _cancel only re-labeled nodes while pw.x
kept burning CPU. These tests pin the fixed end-to-end semantics.
"""

from __future__ import annotations

import os
import stat
import threading
import time
from pathlib import Path

import pytest

from dft_forge.executor import LocalExecutor
from dft_forge.runtime.graph import GraphTemplate, NodeSpec
from dft_forge.runtime.run import NodeRun, ToolError, create_run
from dft_forge.runtime.scheduler import GraphScheduler
from dft_forge.runtime.states import NodeState, RunState
from dft_forge.runtime.store import SQLiteStateStore


SLEEP_BIN = """#!/bin/sh
sleep {duration}
echo done
"""


def _fake_qe_bin(tmp_path: Path, duration: int = 30) -> Path:
    """A fake pw.x that mimics a minutes-long DFT run."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("pw.x", "bands.x", "dos.x", "projwfc.x"):
        p = bin_dir / name
        p.write_text(SLEEP_BIN.format(duration=duration))
        p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return bin_dir


def _input_file(tmp_path: Path, name: str = "si_vcrelax") -> Path:
    workdir = tmp_path / "work"
    workdir.mkdir(exist_ok=True)
    f = workdir / f"{name}.in"
    f.write_text("&CONTROL\n/\n")
    return f


class TestLocalExecutorCancellation:
    def test_cancel_all_kills_running_process(self, tmp_path: Path):
        bin_dir = _fake_qe_bin(tmp_path, duration=30)
        ex = LocalExecutor(qe_bin_dir=bin_dir, max_walltime_sec=300)
        inp = _input_file(tmp_path)

        result_box = {}

        def run_in_thread():
            result_box["r"] = ex._run_tool("pw.x", inp, inp.parent)

        t = threading.Thread(target=run_in_thread)
        t.start()
        # wait for the process to be registered
        for _ in range(100):
            with ex._proc_lock:
                if ex._running:
                    break
            time.sleep(0.05)
        with ex._proc_lock:
            assert len(ex._running) == 1, "process should be tracked while running"

        killed = ex.cancel_all()
        t.join(timeout=10)

        assert killed >= 1
        assert not t.is_alive(), "killed process should unblock the tool thread"
        r = result_box["r"]
        assert r.success is False
        assert "cancelled" in r.error_message

    def test_walltime_timeout_kills_process(self, tmp_path: Path):
        bin_dir = _fake_qe_bin(tmp_path, duration=30)
        ex = LocalExecutor(qe_bin_dir=bin_dir, max_walltime_sec=1)
        inp = _input_file(tmp_path)

        t0 = time.time()
        r = ex._run_tool("pw.x", inp, inp.parent)
        elapsed = time.time() - t0

        assert r.success is False
        assert "timed out" in r.error_message
        assert elapsed < 10, "timeout must kill the process, not wait for sleep to finish"

    def test_cancel_all_with_nothing_running(self, tmp_path: Path):
        bin_dir = _fake_qe_bin(tmp_path)
        ex = LocalExecutor(qe_bin_dir=bin_dir)
        assert ex.cancel_all() == 0


class _BlockingTool:
    """Node tool that blocks until released (mimics an in-flight pw.x node)."""

    def __init__(self, name="block"):
        self.name = name
        self.release = threading.Event()

    def execute(self, node: NodeRun, ctx) -> dict:
        self.release.wait(timeout=30)
        return {"out": "done"}

    def cancel_all(self) -> int:
        self.release.set()
        return 1


class TestSchedulerCancelKillsSubprocesses:
    def test_cancel_invokes_on_cancel_hook(self, tmp_path: Path):
        store = SQLiteStateStore(tmp_path / "s.db")
        template = GraphTemplate(template_id="t", nodes=[NodeSpec(id="a", tool="block")])
        tool = _BlockingTool()
        run = create_run(template, {})
        store.save_run(run)

        cancel_event = threading.Event()
        # auto-fire the cancel shortly after the node starts running
        threading.Timer(0.3, cancel_event.set).start()

        # engine wiring: on_cancel releases/kills the blocked tool
        sched = GraphScheduler(
            store, {"block": tool}, cancel_event=cancel_event, on_cancel=tool.cancel_all
        )
        t0 = time.time()
        result = sched.run(run, template, tmp_path)
        elapsed = time.time() - t0

        assert result.state == RunState.PAUSED
        assert result.nodes["a"].state == NodeState.CANCELLED
        assert elapsed < 10, "cancel must not wait for the blocked node to finish naturally"

    def test_cancelled_node_not_resurrected_by_late_result(self, tmp_path: Path):
        """A thread that finishes after cancellation must not flip the node
        back to SUCCEEDED (would corrupt the paused run's state)."""
        store = SQLiteStateStore(tmp_path / "s.db")
        template = GraphTemplate(template_id="t", nodes=[NodeSpec(id="a", tool="block")])
        tool = _BlockingTool()
        run = create_run(template, {})
        store.save_run(run)

        cancel_event = threading.Event()
        # cancel first, then release the blocked tool a moment later — its
        # success result arrives AFTER the node was marked CANCELLED
        def cancel_then_release():
            cancel_event.set()
            time.sleep(0.2)
            tool.release.set()

        threading.Timer(0.3, cancel_then_release).start()

        sched = GraphScheduler(store, {"block": tool}, cancel_event=cancel_event)
        result = sched.run(run, template, tmp_path)
        assert result.nodes["a"].state == NodeState.CANCELLED


class TestEngineCancelHookCollection:
    def test_engine_collects_cancel_all_from_tools(self, tmp_path: Path):
        from dft_forge.runtime.engine import GraphEngine

        engine = GraphEngine(tmp_path)
        engine.tools["block"] = _BlockingTool()
        engine.tools["plain"] = type("PlainTool", (), {"name": "p", "execute": lambda self, n, c: {}})()

        # tool exposes cancel_all directly
        assert engine._cancel_executors() == 1
        assert tool_release_is_set(engine)

    def test_engine_collects_cancel_all_from_tool_executor(self, tmp_path: Path):
        from dft_forge.runtime.engine import GraphEngine

        tool = _BlockingTool()
        holder = type("Holder", (), {"name": "qe", "executor": tool, "execute": lambda self, n, c: {}})()
        engine = GraphEngine(tmp_path)
        engine.tools["qe"] = holder
        assert engine._cancel_executors() == 1
        assert tool.release.is_set()


def tool_release_is_set(engine) -> bool:
    return any(
        getattr(t, "release", None) is not None and t.release.is_set()
        for t in engine.tools.values()
    )


class TestNodeTimeoutSpec:
    """NodeSpec.timeout_seconds must actually bound the subprocess."""

    def test_node_timeout_overrides_executor_default(self, tmp_path: Path):
        # executor default 300s, node spec says 1s → the 30s sleeper must die at 1s
        bin_dir = _fake_qe_bin(tmp_path, duration=30)
        ex = LocalExecutor(qe_bin_dir=bin_dir, max_walltime_sec=300)
        inp = _input_file(tmp_path)

        t0 = time.time()
        r = ex._run_tool("pw.x", inp, inp.parent, timeout=1)
        elapsed = time.time() - t0

        assert r.success is False
        assert "timed out after 1s" in r.error_message
        assert elapsed < 10

    def test_scheduler_passes_spec_timeout_into_ctx(self, tmp_path: Path):
        from dft_forge.runtime.run import ExecutionContext

        template = GraphTemplate(
            template_id="t",
            nodes=[NodeSpec(id="a", tool="ta", timeout_seconds=42.0)],
        )
        store = SQLiteStateStore(tmp_path / "s.db")

        captured = {}

        class CaptureTool:
            name = "cap"

            def execute(self, node, ctx: ExecutionContext):
                captured["timeout"] = ctx.timeout_seconds
                return {}

        run = create_run(template, {})
        store.save_run(run)
        sched = GraphScheduler(store, {"ta": CaptureTool()})
        result = sched.run(run, template, tmp_path)

        assert result.state == RunState.SUCCEEDED
        assert captured["timeout"] == 42.0

    def test_params_resolved_once_across_requeues(self, tmp_path: Path):
        """The params_resolved flag: a node whose params resolve to {} must not
        be re-resolved on every dispatch (repair/retry loop)."""
        from dft_forge.runtime.run import resolve_params as _rp

        template = GraphTemplate(template_id="t", nodes=[NodeSpec(id="a", tool="ta")])
        store = SQLiteStateStore(tmp_path / "s.db")

        calls = {"n": 0}
        orig = GraphScheduler._prepare_dispatch

        class CountTool:
            name = "ct"

            def execute(self, node, ctx):
                return {}

        run = create_run(template, {})
        store.save_run(run)

        import dft_forge.runtime.scheduler as sched_mod

        orig_resolve = sched_mod.resolve_params

        def counting_resolve(params, run):
            calls["n"] += 1
            return orig_resolve(params, run)

        sched_mod.resolve_params = counting_resolve
        try:
            sched = GraphScheduler(store, {"ta": CountTool()})
            sched.run(run, template, tmp_path)
        finally:
            sched_mod.resolve_params = orig_resolve

        assert calls["n"] == 1, "params should resolve exactly once per run"


class TestWebCancelEndpoint:
    def test_cancel_without_active_run(self):
        from fastapi.testclient import TestClient
        from dft_forge.web import app

        client = TestClient(app)
        r = client.post("/api/sessions/nope/cancel")
        assert r.status_code == 200
        assert r.json()["cancelled"] is False

    def test_cancel_with_active_run_sets_event(self):
        import asyncio

        from fastapi.testclient import TestClient
        import dft_forge.web as web_mod

        client = TestClient(web_mod.app)
        ev = threading.Event()
        task_holder = {}

        async def fake_worker():
            await asyncio.sleep(30)

        # simulate a registered active run for an existing asyncio loop
        loop = asyncio.new_event_loop()
        try:
            task = loop.create_task(fake_worker())
            web_mod.WEB_ACTIVE_RUNS["sess1"] = {"task": task, "cancel": ev, "ts": time.time()}
            r = client.post("/api/sessions/sess1/cancel")
            data = r.json()
            assert data["cancelled"] is True
            assert ev.is_set()
            task.cancel()
        finally:
            loop.close()
            web_mod.WEB_ACTIVE_RUNS.pop("sess1", None)

    def test_running_endpoint(self):
        from fastapi.testclient import TestClient
        import dft_forge.web as web_mod

        client = TestClient(web_mod.app)
        web_mod.WEB_ACTIVE_RUNS["sess2"] = {"cancel": threading.Event(), "ts": 0}
        try:
            assert client.get("/api/sessions/sess2/running").json()["running"] is True
            assert client.get("/api/sessions/other/running").json()["running"] is False
        finally:
            web_mod.WEB_ACTIVE_RUNS.pop("sess2", None)


class TestGraphToolsCancel:
    def test_step_prep_injects_cancel_event(self, tmp_path: Path):
        from dft_forge.agent_loop.step_prep import prepare_step_args

        ev = threading.Event()
        args = {"template_id": "t1_vc_relax", "inputs": {"material": "Si"}}
        ctx = {"cancel_event": ev}
        err = prepare_step_args(
            "graph.run", args,
            session_dir=tmp_path, ctx=ctx,
            built_structure=None, scf_energies=[],
            node_log=[], emit=lambda e: None, idx=0,
        )
        assert err is None
        assert args["_cancel_event"] is ev

    def test_cancel_running_graphs_sweeps_cache(self, tmp_path: Path):
        from dft_forge.tools import graph_tools

        engine = type("E", (), {"tools": {}, "_cancel_executors": lambda self: 2})()
        graph_tools._ENGINE_CACHE.clear()
        graph_tools._ENGINE_CACHE["w1"] = engine
        try:
            assert graph_tools.cancel_running_graphs() == 2
        finally:
            graph_tools._ENGINE_CACHE.clear()
