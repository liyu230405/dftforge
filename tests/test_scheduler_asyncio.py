"""Task: scheduler asyncio dual-mode (same state machine, two runtimes)."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any, Dict

import pytest

from dft_forge.runtime.engine import GraphEngine
from dft_forge.runtime.graph import GraphTemplate, NodeSpec
from dft_forge.runtime.run import ExecutionContext, NodeRun, ToolError, create_run
from dft_forge.runtime.scheduler import GraphScheduler
from dft_forge.runtime.states import NodeState, RunState
from dft_forge.runtime.store import SQLiteStateStore


class _FakeTool:
    """Records execution order; optionally fails or blocks."""

    def __init__(self, name: str, fail: bool = False, block_event: threading.Event = None):
        self.name = name
        self.fail = fail
        self.block_event = block_event
        self.calls = 0
        self.thread_ids = []

    def execute(self, node: NodeRun, ctx: ExecutionContext) -> Dict[str, Any]:
        self.calls += 1
        self.thread_ids.append(threading.get_ident())
        if self.block_event is not None:
            self.block_event.wait(timeout=5)
        if self.fail:
            raise ToolError(f"{self.name} boom", category="execution")
        return {"out": self.name, "workdir": str(ctx.base_dir / node.node_id)}


def _template() -> GraphTemplate:
    return GraphTemplate(
        template_id="t",
        nodes=[
            NodeSpec(id="a", tool="ta"),
            NodeSpec(id="b", tool="tb", depends_on=["a"]),
            NodeSpec(id="c", tool="tc", depends_on=["a"]),
        ],
    )


def _store_run(store: SQLiteStateStore, template: GraphTemplate) -> Any:
    run = create_run(template, {})
    store.save_run(run)
    return run


class TestAsyncioModeParity:
    def test_all_nodes_succeed(self, tmp_path: Path):
        store = SQLiteStateStore(tmp_path / "s.db")
        template = _template()
        tools = {f"t{k}": _FakeTool(k) for k in "abc"}
        run = _store_run(store, template)

        sched = GraphScheduler(store, tools, use_asyncio=True)
        result = sched.run(run, template, tmp_path)

        assert result.state == RunState.SUCCEEDED
        assert all(n.state == NodeState.SUCCEEDED for n in result.nodes.values())
        assert all(t.calls == 1 for t in tools.values())

    def test_failure_propagates_to_blocked(self, tmp_path: Path):
        store = SQLiteStateStore(tmp_path / "s.db")
        template = _template()
        tools = {
            "ta": _FakeTool("a", fail=True),
            "tb": _FakeTool("b"),
            "tc": _FakeTool("c"),
        }
        run = _store_run(store, template)

        sched = GraphScheduler(store, tools, use_asyncio=True)
        result = sched.run(run, template, tmp_path)

        assert result.state == RunState.FAILED
        assert result.nodes["a"].state == NodeState.FAILED
        assert result.nodes["b"].state == NodeState.BLOCKED
        assert result.nodes["c"].state == NodeState.BLOCKED
        assert tools["tb"].calls == 0  # dependents never dispatched

    def test_tools_run_off_event_loop_thread(self, tmp_path: Path):
        """Blocking tools must run in worker threads, never on the loop."""
        store = SQLiteStateStore(tmp_path / "s.db")
        template = _template()
        tool = _FakeTool("a")
        tools = {"ta": tool, "tb": _FakeTool("b"), "tc": _FakeTool("c")}
        run = _store_run(store, template)

        loop_thread = threading.get_ident()
        sched = GraphScheduler(store, tools, use_asyncio=True)
        sched.run(run, template, tmp_path)

        # asyncio.run creates a fresh loop thread different from this one
        assert all(tid != loop_thread for tid in tool.thread_ids)

    def test_run_async_from_live_loop(self, tmp_path: Path):
        store = SQLiteStateStore(tmp_path / "s.db")
        template = _template()
        tools = {f"t{k}": _FakeTool(k) for k in "abc"}

        async def main():
            run = _store_run(store, template)
            sched = GraphScheduler(store, tools, use_asyncio=True)
            return await sched.run_async(run, template, tmp_path)

        result = asyncio.run(main())
        assert result.state == RunState.SUCCEEDED

    def test_run_asyncio_inside_loop_raises(self, tmp_path: Path):
        """sync .run() with use_asyncio=True inside a live loop must fail loud,
        not silently nest event loops."""
        store = SQLiteStateStore(tmp_path / "s.db")
        template = _template()
        tools = {f"t{k}": _FakeTool(k) for k in "abc"}
        run = _store_run(store, template)
        sched = GraphScheduler(store, tools, use_asyncio=True)

        async def main():
            return sched.run(run, template, tmp_path)

        with pytest.raises(RuntimeError):
            asyncio.run(main())


class TestThreadedModeUnchanged:
    def test_threaded_default_still_works(self, tmp_path: Path):
        store = SQLiteStateStore(tmp_path / "s.db")
        template = _template()
        tools = {f"t{k}": _FakeTool(k) for k in "abc"}
        run = _store_run(store, template)

        sched = GraphScheduler(store, tools)  # default: threaded
        result = sched.run(run, template, tmp_path)
        assert result.state == RunState.SUCCEEDED


class TestEngineAsyncioSwitch:
    def test_engine_use_asyncio_flag_flows_through(self, tmp_path: Path):
        engine = GraphEngine(tmp_path, use_asyncio=True)
        assert engine.use_asyncio is True
        engine2 = GraphEngine(tmp_path / "b")
        assert engine2.use_asyncio is False

    def test_engine_runs_template_in_asyncio_mode(self, tmp_path: Path):
        engine = GraphEngine(tmp_path, use_asyncio=True)
        engine.tools["ta"] = _FakeTool("a")
        engine.tools["tb"] = _FakeTool("b")
        engine.register_template(GraphTemplate(
            template_id="t",
            nodes=[NodeSpec(id="a", tool="ta"), NodeSpec(id="b", tool="tb", depends_on=["a"])],
        ))
        run = engine.run_template("t", {})
        assert run.state == RunState.SUCCEEDED
