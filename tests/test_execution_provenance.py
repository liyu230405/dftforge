from __future__ import annotations

from pathlib import Path

import pytest

from dft_forge.runtime.engine import GraphEngine
from dft_forge.runtime.graph import GraphTemplate, NodeSpec
from dft_forge.runtime.run import ToolError
from dft_forge.runtime.states import NodeState, RunState
from dft_forge.tools import graph_tools


class _Tool:
    name = "fake"

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls = 0

    def execute(self, node, ctx):
        self.calls += 1
        if self.fail:
            raise ToolError("failed", category="test")
        return {"value": self.calls}


def _template(name: str = "provenance") -> GraphTemplate:
    return GraphTemplate(template_id=name, nodes=[NodeSpec(id="a", tool="fake")])


def _engine(tmp_path: Path, identity: str = "identity-a", walltime: int = 60, tool=None):
    engine = GraphEngine(
        tmp_path / "workspace",
        tools={"fake": tool or _Tool()},
        execution_identity=identity,
        execution_profile={
            "backend": "fake",
            "simulated": True,
            "walltime_limit_sec": walltime,
        },
    )
    engine.register_template(_template())
    return engine


def test_new_run_persists_origin_and_one_start_attempt(tmp_path):
    engine = _engine(tmp_path)
    run = engine.run_template("provenance")

    execution = run.metadata["execution"]
    assert execution["origin"]["identity"] == "identity-a"
    assert execution["origin"]["profile"]["walltime_limit_sec"] == 60
    assert [a["operation"] for a in execution["attempts"]] == ["start"]
    assert engine.status(run.run_id)["metadata"] == run.metadata


def test_retry_records_retry_without_nested_start(tmp_path):
    failing = _Tool(fail=True)
    engine = _engine(tmp_path, tool=failing)
    failed = engine.run_template("provenance")
    engine.tools["fake"] = _Tool()

    retried = engine.retry(failed.run_id)

    assert retried.state == RunState.SUCCEEDED
    assert [a["operation"] for a in retried.metadata["execution"]["attempts"]] == [
        "start",
        "retry",
    ]


def test_resume_records_current_profile_but_keeps_origin(tmp_path):
    first = _engine(tmp_path, walltime=60)
    run = first.create("provenance")
    run.state = RunState.PAUSED
    run.nodes["a"].state = NodeState.CANCELLED
    first.store.save_node_run(run, run.nodes["a"])
    first.store.save_run(run)

    second = _engine(tmp_path, walltime=120)
    resumed = second.resume(run.run_id)
    execution = resumed.metadata["execution"]

    assert execution["origin"]["profile"]["walltime_limit_sec"] == 60
    assert execution["attempts"][-1]["operation"] == "resume"
    assert execution["attempts"][-1]["profile"]["walltime_limit_sec"] == 120


def test_identity_mismatch_fails_before_recovery_mutation(tmp_path):
    first = _engine(tmp_path)
    run = first.create("provenance")
    run.state = RunState.PAUSED
    run.nodes["a"].state = NodeState.RUNNING
    first.store.save_node_run(run, run.nodes["a"])
    first.store.save_run(run)

    second = _engine(tmp_path, identity="identity-b")
    with pytest.raises(ValueError, match="execution identity mismatch"):
        second.resume(run.run_id)

    unchanged = second.store.load_run(run.run_id, _template())
    assert unchanged.state == RunState.PAUSED
    assert unchanged.nodes["a"].state == NodeState.RUNNING


def test_recorded_identity_rejects_identity_less_engine_before_resume(tmp_path):
    source = _engine(tmp_path)
    run = source.create("provenance")
    run.state = RunState.PAUSED
    run.nodes["a"].state = NodeState.RUNNING
    source.store.save_node_run(run, run.nodes["a"])
    source.store.save_run(run)

    identity_less = GraphEngine(tmp_path / "workspace", tools={"fake": _Tool()})
    identity_less.register_template(_template())
    with pytest.raises(ValueError, match="execution identity mismatch"):
        identity_less.resume(run.run_id)

    unchanged = identity_less.store.load_run(run.run_id, _template())
    assert unchanged.state == RunState.PAUSED
    assert unchanged.nodes["a"].state == NodeState.RUNNING


def test_legacy_recovery_marks_origin_unknown(tmp_path):
    legacy = GraphEngine(tmp_path / "workspace", tools={"fake": _Tool()})
    legacy.register_template(_template())
    run = legacy.create("provenance")
    run.state = RunState.PAUSED
    run.nodes["a"].state = NodeState.CANCELLED
    legacy.store.save_node_run(run, run.nodes["a"])
    legacy.store.save_run(run)

    current = _engine(tmp_path)
    recovered = current.resume(run.run_id)
    execution = recovered.metadata["execution"]
    assert execution["origin_unknown"] is True
    assert execution["origin"]["identity"] == "identity-a"


def test_executor_selection_is_fail_closed_and_walltime_is_validated(tmp_path, monkeypatch):
    monkeypatch.setenv("DFT_FORGE_EXECUTOR", "fake")
    monkeypatch.setenv("DFT_FORGE_WALLTIME", "30")
    fake = graph_tools._pick_executor()
    assert fake.__class__.__name__ == "FakeExecutor"

    monkeypatch.setenv("DFT_FORGE_EXECUTOR", "local")
    monkeypatch.delenv("DFT_FORGE_QE_BIN", raising=False)
    monkeypatch.setattr(graph_tools.shutil, "which", lambda _: None)
    with pytest.raises(ValueError, match="本地 QE"):
        graph_tools._pick_executor()

    monkeypatch.setenv("DFT_FORGE_WALLTIME", "29")
    with pytest.raises(ValueError, match="DFT_FORGE_WALLTIME"):
        graph_tools._pick_executor()


def test_read_only_graph_paths_do_not_require_qe(tmp_path, monkeypatch):
    monkeypatch.setenv("DFT_FORGE_EXECUTOR", "local")
    monkeypatch.delenv("DFT_FORGE_QE_BIN", raising=False)
    monkeypatch.setattr(graph_tools.shutil, "which", lambda _: None)
    graph_tools._ENGINE_CACHE.clear()
    try:
        result = graph_tools._graph_templates({"workdir": str(tmp_path / "graphs")})
        assert "t1_vc_relax" in result["templates"]
    finally:
        graph_tools._ENGINE_CACHE.clear()
