from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

from dft_forge.runtime.graph import GraphTemplate, NodeSpec
from dft_forge.runtime.run import create_run
from dft_forge.runtime.scheduler import GraphScheduler
from dft_forge.runtime.states import NodeState, RunState
from dft_forge.runtime.store import SQLiteStateStore


def _run(tmp_path: Path, tool_name: str = "tool"):
    store = SQLiteStateStore(tmp_path / "state.db")
    template = GraphTemplate(template_id="cancel", nodes=[NodeSpec(id="a", tool=tool_name)])
    run = create_run(template, {})
    store.save_run(run)
    return store, template, run


def test_async_cancel_without_hook_waits_for_natural_worker_exit(tmp_path):
    cancel = threading.Event()
    exited = threading.Event()

    class Tool:
        name = "tool"

        def execute(self, node, ctx):
            cancel.set()
            time.sleep(0.15)
            exited.set()
            return {"ok": True}

    store, template, run = _run(tmp_path)
    started = time.monotonic()
    scheduler = GraphScheduler(
        store, {"tool": Tool()}, cancel_event=cancel, use_asyncio=True, poll_interval=0.01
    )

    async def live_loop():
        result = await scheduler.run_async(run, template, tmp_path)
        # Assert before asyncio.run shuts down its default executor: that
        # shutdown would otherwise hide a prematurely returned scheduler.
        assert exited.is_set()
        return result

    result = asyncio.run(live_loop())

    assert time.monotonic() - started >= 0.12
    assert result.state == RunState.PAUSED
    assert result.nodes["a"].state == NodeState.CANCELLED


def test_async_cancel_hook_is_called_once_and_returns_quickly(tmp_path):
    cancel = threading.Event()
    release = threading.Event()
    calls = []

    class Tool:
        name = "tool"

        def execute(self, node, ctx):
            release.wait(2)
            return {"ok": True}

    def on_cancel():
        calls.append(1)
        release.set()

    store, template, run = _run(tmp_path)
    timer = threading.Timer(0.03, cancel.set)
    timer.start()
    started = time.monotonic()
    result = GraphScheduler(
        store,
        {"tool": Tool()},
        cancel_event=cancel,
        use_asyncio=True,
        poll_interval=0.01,
        on_cancel=on_cancel,
    ).run(run, template, tmp_path)

    assert time.monotonic() - started < 0.8
    assert calls == [1]
    assert result.state == RunState.PAUSED
    assert result.nodes["a"].state == NodeState.CANCELLED


def test_async_cancel_before_scheduler_commit_discards_completed_result(tmp_path):
    cancel = threading.Event()

    class Tool:
        name = "tool"

        def execute(self, node, ctx):
            cancel.set()
            return {"must_not_commit": True}

    store, template, run = _run(tmp_path)
    result = GraphScheduler(
        store, {"tool": Tool()}, cancel_event=cancel, use_asyncio=True, poll_interval=0.01
    ).run(run, template, tmp_path)

    assert result.state == RunState.PAUSED
    assert result.nodes["a"].state == NodeState.CANCELLED
    assert result.nodes["a"].outputs == {}


def test_async_cancel_after_last_successful_commit_keeps_success(tmp_path):
    cancel = threading.Event()

    class Tool:
        name = "tool"

        def execute(self, node, ctx):
            return {"ok": True}

    def on_event(event):
        if event.get("state") == "succeeded":
            cancel.set()

    store, template, run = _run(tmp_path)
    result = GraphScheduler(
        store,
        {"tool": Tool()},
        cancel_event=cancel,
        use_asyncio=True,
        poll_interval=0.01,
    ).run(run, template, tmp_path, on_event=on_event)

    assert result.state == RunState.SUCCEEDED
    assert result.nodes["a"].state == NodeState.SUCCEEDED


def test_async_cancel_during_tool_exception_consumes_task_result(tmp_path):
    cancel = threading.Event()

    class Tool:
        name = "tool"

        def execute(self, node, ctx):
            cancel.set()
            raise RuntimeError("worker failed")

    store, template, run = _run(tmp_path)
    result = GraphScheduler(
        store, {"tool": Tool()}, cancel_event=cancel, use_asyncio=True, poll_interval=0.01
    ).run(run, template, tmp_path)

    assert result.state == RunState.PAUSED
    assert result.nodes["a"].state == NodeState.CANCELLED
