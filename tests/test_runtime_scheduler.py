"""Tests for the scheduler loop and GraphEngine lifecycle."""

import threading
import time

import pytest

from dft_forge.runtime.engine import GraphEngine
from dft_forge.runtime.graph import GraphTemplate, NodeSpec
from dft_forge.runtime.run import GraphRun, NodeRun, ToolError, create_run
from dft_forge.runtime.scheduler import (
    GraphScheduler,
    find_ready_nodes,
    process_blocked,
)
from dft_forge.runtime.states import NodeState, RunState
from dft_forge.runtime.store import SQLiteStateStore


class OkTool:
    def __init__(self, name="fake", delay=0.0, outputs=None):
        self.name = name
        self.delay = delay
        self.outputs = outputs or {"v": 1}
        self.calls = 0
        self.lock = threading.Lock()

    def execute(self, node, ctx):
        with self.lock:
            self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        (ctx.base_dir / node.node_id).mkdir(parents=True, exist_ok=True)
        return dict(self.outputs)


class FailTool:
    def __init__(self, repairable=True, category="verification", fail_times=999):
        self.name = "fake"
        self.repairable = repairable
        self.category = category
        self.fail_times = fail_times
        self.calls = 0

    def execute(self, node, ctx):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ToolError("boom", category=self.category, repairable=self.repairable)
        return {"v": 2}


class BumpRepairer:
    name = "bump"

    def __init__(self):
        self.calls = 0

    def repair(self, node, error):
        self.calls += 1
        node.params["ecutwfc"] = float(node.params.get("ecutwfc") or 45) + 10
        return True


def make_engine(tmp_path, tool, repairer=None, max_concurrency=2, template=None):
    template = template or GraphTemplate(
        template_id="t_sched",
        nodes=[NodeSpec(id="a", tool="fake"), NodeSpec(id="b", tool="fake", depends_on=["a"])],
    )
    engine = GraphEngine(
        base_dir=tmp_path / "ws",
        store_path=tmp_path / "ws" / "state.db",
        tools={"fake": tool},
        repairers={"bump": repairer} if repairer else {},
        max_concurrency=max_concurrency,
    )
    engine.register_template(template)
    return engine


class TestSchedulerHelpers:
    def _run(self):
        run = GraphRun(run_id="r", template_id="t")
        for nid in ("a", "b", "c"):
            run.nodes[nid] = NodeRun(node_id=nid)
        return run

    def test_find_ready_no_deps(self):
        run = self._run()
        template = GraphTemplate(
            template_id="t",
            nodes=[NodeSpec(id=n, tool="x") for n in ("a", "b", "c")],
        )
        assert sorted(find_ready_nodes(run, template)) == ["a", "b", "c"]

    def test_find_ready_respects_deps(self):
        run = self._run()
        template = GraphTemplate(
            template_id="t",
            nodes=[
                NodeSpec(id="a", tool="x"),
                NodeSpec(id="b", tool="x", depends_on=["a"]),
                NodeSpec(id="c", tool="x", depends_on=["b"]),
            ],
        )
        assert find_ready_nodes(run, template) == ["a"]
        run.nodes["a"].state = NodeState.SUCCEEDED
        assert find_ready_nodes(run, template) == ["b"]

    def test_ready_state_requeued_nodes(self):
        run = self._run()
        run.nodes["a"].state = NodeState.READY
        template = GraphTemplate(template_id="t", nodes=[NodeSpec(id="a", tool="x")])
        assert find_ready_nodes(run, template) == ["a"]

    def test_process_blocked(self):
        run = self._run()
        run.nodes["a"].state = NodeState.FAILED
        template = GraphTemplate(
            template_id="t",
            nodes=[
                NodeSpec(id="a", tool="x"),
                NodeSpec(id="b", tool="x", depends_on=["a"]),
                NodeSpec(id="c", tool="x", depends_on=["b"]),
            ],
        )
        # single pass blocks b, and transitively c (b became Blocked in-order)
        assert process_blocked(run, template) == ["b", "c"]
        assert run.nodes["b"].state == NodeState.BLOCKED
        assert run.nodes["c"].state == NodeState.BLOCKED


class TestSchedulerRun:
    def test_linear_success(self, tmp_path):
        engine = make_engine(tmp_path, OkTool())
        run = engine.run_template("t_sched")
        assert run.state == RunState.SUCCEEDED
        assert run.nodes["a"].outputs == {"v": 1}
        assert run.nodes["a"].attempt == 1

    def test_diamond_parallelism(self, tmp_path):
        template = GraphTemplate(
            template_id="t_diamond",
            nodes=[
                NodeSpec(id="top", tool="fake"),
                NodeSpec(id="l", tool="fake", depends_on=["top"]),
                NodeSpec(id="r", tool="fake", depends_on=["top"]),
                NodeSpec(id="join", tool="fake", depends_on=["l", "r"]),
            ],
        )
        engine = make_engine(tmp_path, OkTool(delay=0.12), template=template)
        start = time.time()
        run = engine.run_template("t_diamond")
        assert run.state == RunState.SUCCEEDED
        elapsed = time.time() - start
        # l and r ran concurrently: serial would need >= 4 * 0.12s
        assert elapsed < 4 * 0.12 + 0.35

    def test_failure_blocks_downstream(self, tmp_path):
        engine = make_engine(tmp_path, FailTool())
        run = engine.run_template("t_sched")
        assert run.state == RunState.FAILED
        assert run.nodes["a"].state == NodeState.FAILED
        assert run.nodes["b"].state == NodeState.BLOCKED

    def test_partial_success(self, tmp_path):
        template = GraphTemplate(
            template_id="t_partial",
            nodes=[
                NodeSpec(id="ok", tool="good"),
                NodeSpec(id="bad", tool="bad"),
            ],
        )
        engine = GraphEngine(
            base_dir=tmp_path / "ws",
            store_path=tmp_path / "ws" / "state.db",
            tools={"good": OkTool(), "bad": FailTool(repairable=False)},
            max_concurrency=2,
        )
        engine.register_template(template)
        run = engine.run_template("t_partial")
        assert run.state == RunState.PARTIALLY_SUCCEEDED
        assert run.nodes["ok"].state == NodeState.SUCCEEDED

    def test_repair_path(self, tmp_path):
        tool = FailTool(fail_times=1)  # fails once, then succeeds
        repairer = BumpRepairer()
        template = GraphTemplate(
            template_id="t_repair",
            nodes=[NodeSpec(id="a", tool="fake", max_repair_attempts=2)],
        )
        engine = GraphEngine(
            base_dir=tmp_path / "ws",
            store_path=tmp_path / "ws" / "state.db",
            tools={"fake": tool},
            repairers={"bump": repairer},
        )
        engine.register_template(template)
        run = engine.run_template("t_repair")
        assert run.state == RunState.SUCCEEDED
        assert repairer.calls == 1
        assert run.nodes["a"].repair_attempts == 1
        assert run.nodes["a"].outputs == {"v": 2}
        assert run.nodes["a"].params["ecutwfc"] == 55.0

    def test_repair_exhausted_fails(self, tmp_path):
        tool = FailTool(fail_times=999)
        repairer = BumpRepairer()
        template = GraphTemplate(
            template_id="t_repair2",
            nodes=[NodeSpec(id="a", tool="fake", max_repair_attempts=1)],
        )
        engine = GraphEngine(
            base_dir=tmp_path / "ws",
            store_path=tmp_path / "ws" / "state.db",
            tools={"fake": tool},
            repairers={"bump": repairer},
        )
        engine.register_template(template)
        run = engine.run_template("t_repair2")
        assert run.state == RunState.FAILED
        assert run.nodes["a"].state == NodeState.FAILED
        assert run.nodes["a"].error and "boom" in run.nodes["a"].error

    def test_plain_retry(self, tmp_path):
        tool = FailTool(fail_times=1)
        template = GraphTemplate(
            template_id="t_retry",
            nodes=[NodeSpec(id="a", tool="fake", max_attempts=2, max_repair_attempts=0)],
        )
        engine = GraphEngine(
            base_dir=tmp_path / "ws",
            store_path=tmp_path / "ws" / "state.db",
            tools={"fake": tool},
        )
        engine.register_template(template)
        run = engine.run_template("t_retry")
        assert run.state == RunState.SUCCEEDED
        assert run.nodes["a"].attempt == 2

    def test_concurrency_limit(self, tmp_path):
        template = GraphTemplate(
            template_id="t_par",
            nodes=[NodeSpec(id=f"n{i}", tool="fake") for i in range(4)],
        )
        tool = OkTool(delay=0.1)
        engine = make_engine(tmp_path, tool, max_concurrency=2, template=template)
        start = time.time()
        run = engine.run_template("t_par")
        assert run.state == RunState.SUCCEEDED
        assert time.time() - start >= 0.2  # 4 tasks / 2 slots = 2 waves


class TestEngineLifecycle:
    def test_status_and_list(self, tmp_path):
        engine = make_engine(tmp_path, OkTool())
        run = engine.run_template("t_sched")
        status = engine.status(run.run_id)
        assert status["state"] == "succeeded"
        assert set(status["nodes"]) == {"a", "b"}
        assert any(r["run_id"] == run.run_id for r in engine.list_runs())

    def test_retry_failed_nodes(self, tmp_path):
        # a fails permanently first time; retry with a fixed tool succeeds
        class FlakyFactory:
            def __init__(self):
                self.tool = FailTool(fail_times=1)

        template = GraphTemplate(
            template_id="t_rt",
            nodes=[NodeSpec(id="a", tool="fake", max_repair_attempts=0, max_attempts=1)],
        )
        engine = GraphEngine(
            base_dir=tmp_path / "ws",
            store_path=tmp_path / "ws" / "state.db",
            tools={"fake": FlakyFactory().tool},
        )
        engine.register_template(template)
        run = engine.run_template("t_rt")
        assert run.state == RunState.FAILED

        fixed = OkTool()
        engine.tools["fake"] = fixed
        retried = engine.retry(run.run_id)
        assert retried.state == RunState.SUCCEEDED

    def test_terminal_run_start_is_noop(self, tmp_path):
        engine = make_engine(tmp_path, OkTool())
        run = engine.run_template("t_sched")
        again = engine.start(run.run_id)
        assert again.state == RunState.SUCCEEDED
        assert engine.tools["fake"].calls == 2  # a and b each ran once total

    def test_unknown_template_raises(self, tmp_path):
        engine = make_engine(tmp_path, OkTool())
        with pytest.raises(ValueError, match="unknown template"):
            engine.get_template("ghost")

    def test_invalid_template_rejected_at_registration(self, tmp_path):
        engine = make_engine(tmp_path, OkTool())
        bad = GraphTemplate(
            template_id="bad",
            nodes=[NodeSpec(id="a", tool="fake", depends_on=["a"])],
        )
        with pytest.raises(ValueError, match="invalid template"):
            engine.register_template(bad)

    def test_resume_after_interrupted_run(self, tmp_path):
        template = GraphTemplate(
            template_id="t_res",
            nodes=[
                NodeSpec(id="a", tool="fake"),
                NodeSpec(id="b", tool="fake", depends_on=["a"]),
            ],
        )
        engine = GraphEngine(
            base_dir=tmp_path / "ws",
            store_path=tmp_path / "ws" / "state.db",
            tools={"fake": OkTool()},
        )
        engine.register_template(template)
        run = engine.create("t_res")
        # simulate a crash: a succeeded and persisted, b stuck in Running
        run.nodes["a"].state = NodeState.SUCCEEDED
        run.nodes["a"].outputs = {"v": 1}
        run.nodes["b"].state = NodeState.RUNNING
        engine.store.save_run(run)
        engine.store.save_node_run(run, run.nodes["a"])
        engine.store.save_node_run(run, run.nodes["b"])

        resumed = engine.resume(run.run_id)
        assert resumed.state == RunState.SUCCEEDED
        assert resumed.nodes["a"].attempt == 0  # untouched
