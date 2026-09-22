"""Regression tests for graph-run recovery and lifecycle persistence."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from dft_forge.runtime.engine import GraphEngine
from dft_forge.runtime.graph import GraphTemplate, NodeSpec
from dft_forge.runtime.run import ToolError
from dft_forge.runtime.states import NodeState, RunState


class RecordingTool:
    def __init__(self, name: str = "fake", fail: bool = False):
        self.name = name
        self.fail = fail
        self.calls = []

    def execute(self, node, ctx):
        self.calls.append((node.node_id, dict(node.params)))
        if self.fail:
            raise ToolError("failed", category="test")
        return {"node": node.node_id}


def _engine(tmp_path: Path, template: GraphTemplate, tool: RecordingTool) -> GraphEngine:
    engine = GraphEngine(tmp_path / "workspace", tools={tool.name: tool})
    engine.register_template(template)
    return engine


def test_resume_resets_cancelled_nodes(tmp_path):
    template = GraphTemplate(template_id="recover", nodes=[NodeSpec(id="a", tool="fake")])
    tool = RecordingTool()
    engine = _engine(tmp_path, template, tool)
    run = engine.create("recover")
    run.state = RunState.PAUSED
    run.nodes["a"].state = NodeState.CANCELLED
    engine.store.save_node_run(run, run.nodes["a"])
    engine.store.save_run(run)

    resumed = engine.resume(run.run_id)

    assert resumed.state == RunState.SUCCEEDED
    assert resumed.nodes["a"].state == NodeState.SUCCEEDED
    assert [node_id for node_id, _ in tool.calls] == ["a"]


def test_repaired_params_survive_restart_and_resume(tmp_path):
    template = GraphTemplate(
        template_id="repair_resume",
        nodes=[NodeSpec(id="a", tool="fake", params={"cutoff": 30})],
    )
    tool = RecordingTool()
    engine = _engine(tmp_path, template, tool)
    run = engine.create("repair_resume")
    node = run.nodes["a"]
    node.state = NodeState.READY
    node.params = {"cutoff": 60}
    node.params_resolved = True
    run.state = RunState.RUNNING
    engine.store.save_node_run(run, node)
    engine.store.save_run(run)

    resumed = engine.resume(run.run_id)

    assert resumed.state == RunState.SUCCEEDED
    assert tool.calls == [("a", {"cutoff": 60})]


def test_template_snapshot_wins_over_changed_registry(tmp_path):
    original = GraphTemplate(template_id="snapshot", nodes=[NodeSpec(id="a", tool="fake")])
    tool = RecordingTool()
    engine = _engine(tmp_path, original, tool)
    run = engine.create("snapshot")
    changed = GraphTemplate(template_id="snapshot", nodes=[NodeSpec(id="b", tool="fake")])
    engine._templates["snapshot"] = changed

    finished = engine.start(run.run_id)

    assert finished.state == RunState.SUCCEEDED
    assert set(finished.nodes) == {"a"}
    assert [node_id for node_id, _ in tool.calls] == ["a"]


def test_template_snapshot_survives_fresh_engine_restart(tmp_path):
    template = GraphTemplate(template_id="fresh_restart", nodes=[NodeSpec(id="a", tool="fake")])
    first_tool = RecordingTool()
    first = _engine(tmp_path, template, first_tool)
    run = first.create("fresh_restart")
    run.state = RunState.PAUSED
    run.nodes["a"].state = NodeState.CANCELLED
    first.store.save_node_run(run, run.nodes["a"])
    first.store.save_run(run)

    second_tool = RecordingTool()
    second = GraphEngine(tmp_path / "workspace", tools={"fake": second_tool})

    resumed = second.resume(run.run_id)

    assert resumed.state == RunState.SUCCEEDED
    assert [node_id for node_id, _ in second_tool.calls] == ["a"]


def test_template_snapshot_is_immutable(tmp_path):
    original = GraphTemplate(
        template_id="immutable",
        nodes=[NodeSpec(id="a", tool="fake", params={"value": 1})],
    )
    replacement = GraphTemplate(
        template_id="immutable",
        nodes=[NodeSpec(id="b", tool="fake", params={"value": 2})],
    )
    engine = _engine(tmp_path, original, RecordingTool())
    run = engine.create("immutable")

    engine.store.save_run(run, replacement)

    snapshot = engine.store.template_for(run.run_id)
    assert snapshot is not None
    assert {node.id for node in snapshot.nodes} == {"a"}


def test_corrupt_snapshot_does_not_fallback_to_current_template(tmp_path):
    template = GraphTemplate(template_id="corrupt_snapshot", nodes=[NodeSpec(id="a", tool="fake")])
    engine = _engine(tmp_path, template, RecordingTool())
    run = engine.create("corrupt_snapshot")
    with sqlite3.connect(engine.store.path) as conn:
        conn.execute(
            "UPDATE graph_run_templates SET template_json = ? WHERE run_id = ?",
            ("{not-json", run.run_id),
        )

    with pytest.raises(ValueError, match="invalid template snapshot"):
        engine.status(run.run_id)


def test_snapshot_identity_and_tool_contract_are_validated(tmp_path):
    template = GraphTemplate(template_id="snapshot_contract", nodes=[NodeSpec(id="a", tool="fake")])
    engine = _engine(tmp_path, template, RecordingTool())
    run = engine.create("snapshot_contract")
    payload = template.to_dict()
    payload["template_id"] = "other"
    with sqlite3.connect(engine.store.path) as conn:
        conn.execute(
            "UPDATE graph_run_templates SET template_json = ? WHERE run_id = ?",
            (json.dumps(payload), run.run_id),
        )
    with pytest.raises(ValueError, match="template snapshot mismatch"):
        engine.status(run.run_id)

    payload["template_id"] = "snapshot_contract"
    payload["nodes"][0]["tool"] = "missing"
    with sqlite3.connect(engine.store.path) as conn:
        conn.execute(
            "UPDATE graph_run_templates SET template_json = ? WHERE run_id = ?",
            (json.dumps(payload), run.run_id),
        )
    with pytest.raises(ValueError, match="invalid template snapshot"):
        engine.status(run.run_id)


def test_paused_blocked_nodes_are_rechecked_on_resume(tmp_path):
    template = GraphTemplate(
        template_id="paused_blocked",
        nodes=[NodeSpec(id="a", tool="fake"), NodeSpec(id="b", tool="fake", depends_on=["a"])],
    )
    tool = RecordingTool()
    engine = _engine(tmp_path, template, tool)
    run = engine.create("paused_blocked")
    run.state = RunState.PAUSED
    run.nodes["a"].state = NodeState.CANCELLED
    run.nodes["b"].state = NodeState.BLOCKED
    engine.store.save_node_run(run, run.nodes["a"])
    engine.store.save_node_run(run, run.nodes["b"])
    engine.store.save_run(run)

    resumed = engine.resume(run.run_id)

    assert resumed.state == RunState.SUCCEEDED
    assert [node_id for node_id, _ in tool.calls] == ["a", "b"]


def test_blocked_nodes_are_persisted(tmp_path):
    template = GraphTemplate(
        template_id="blocked_persist",
        nodes=[NodeSpec(id="a", tool="fake"), NodeSpec(id="b", tool="fake", depends_on=["a"])],
    )
    tool = RecordingTool(fail=True)
    engine = _engine(tmp_path, template, tool)

    run = engine.run_template("blocked_persist")
    loaded = engine.store.load_run(run.run_id, template)

    assert loaded is not None
    assert loaded.nodes["a"].state == NodeState.FAILED
    assert loaded.nodes["b"].state == NodeState.BLOCKED


def test_retry_validates_before_mutating(tmp_path):
    template = GraphTemplate(template_id="retry_validate", nodes=[NodeSpec(id="a", tool="fake")])
    tool = RecordingTool(fail=True)
    engine = _engine(tmp_path, template, tool)
    run = engine.run_template("retry_validate")

    with pytest.raises(ValueError, match="unknown retry"):
        engine.retry(run.run_id, ["a", "missing"])

    unchanged = engine.store.load_run(run.run_id, template)
    assert unchanged is not None
    assert unchanged.state == RunState.FAILED
    assert unchanged.nodes["a"].state == NodeState.FAILED


def test_retry_requeues_blocked_descendants(tmp_path):
    template = GraphTemplate(
        template_id="retry_chain",
        nodes=[
            NodeSpec(id="a", tool="fake"),
            NodeSpec(id="b", tool="fake", depends_on=["a"]),
            NodeSpec(id="c", tool="fake", depends_on=["b"]),
        ],
    )
    failing = RecordingTool(fail=True)
    engine = _engine(tmp_path, template, failing)
    run = engine.run_template("retry_chain")
    engine.tools["fake"] = RecordingTool()

    retried = engine.retry(run.run_id, ["a"])

    assert retried.state == RunState.SUCCEEDED
    assert all(node.state == NodeState.SUCCEEDED for node in retried.nodes.values())


def test_retry_clears_stale_outputs_and_job_id(tmp_path):
    template = GraphTemplate(template_id="retry_clear", nodes=[NodeSpec(id="a", tool="fake")])
    tool = RecordingTool()
    engine = _engine(tmp_path, template, tool)
    run = engine.create("retry_clear")
    node = run.nodes["a"]
    node.state = NodeState.FAILED
    node.outputs = {"stale": True}
    node.job_id = "old-job"
    run.state = RunState.FAILED
    engine.store.save_node_run(run, node)
    engine.store.save_run(run)

    retried = engine.retry(run.run_id)

    assert retried.state == RunState.SUCCEEDED
    assert retried.nodes["a"].outputs == {"node": "a"}
    assert retried.nodes["a"].job_id is None


@pytest.mark.parametrize("dead_state", [NodeState.CANCELLED, NodeState.SKIPPED])
def test_retry_rejects_blocked_node_with_unretryable_dependency(tmp_path, dead_state):
    template = GraphTemplate(
        template_id=f"retry_dead_{dead_state.value}",
        nodes=[
            NodeSpec(id="dead", tool="fake"),
            NodeSpec(id="blocked", tool="fake", depends_on=["dead"]),
            NodeSpec(id="ok", tool="fake"),
        ],
    )
    engine = _engine(tmp_path, template, RecordingTool())
    run = engine.create(template.template_id)
    run.state = RunState.PARTIALLY_SUCCEEDED
    run.nodes["dead"].state = dead_state
    run.nodes["blocked"].state = NodeState.BLOCKED
    run.nodes["ok"].state = NodeState.SUCCEEDED
    for node in run.nodes.values():
        engine.store.save_node_run(run, node)
    engine.store.save_run(run)

    with pytest.raises(ValueError, match="failed dependency remains outside retry targets"):
        engine.retry(run.run_id, ["blocked"])
