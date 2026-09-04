"""Tests for the graph runtime kernel: states, templates, DAG validation, store."""

import json

import pytest

from dft_forge.runtime.graph import GraphTemplate, NodeSpec, load_template
from dft_forge.runtime.run import ToolError, create_run, resolve_params
from dft_forge.runtime.states import (
    NodeState,
    RunState,
    TERMINAL_NODE_STATES,
    can_transition,
    determine_run_status,
)
from dft_forge.runtime.store import SQLiteStateStore

TEMPLATES = None  # filled in fixture-free helpers below


def linear_template() -> GraphTemplate:
    return GraphTemplate(
        template_id="t_test",
        nodes=[
            NodeSpec(id="a", tool="fake", params={"x": "${inputs.alpha}"}),
            NodeSpec(id="b", tool="fake", params={"y": "${nodes.a.outputs.v}"}, depends_on=["a"]),
        ],
        inputs={"alpha": 1},
    )


class TestStates:
    def test_legal_transitions(self):
        assert can_transition(NodeState.PENDING, NodeState.READY)
        assert can_transition(NodeState.RUNNING, NodeState.REPAIRING)
        assert can_transition(NodeState.REPAIRING, NodeState.READY)
        assert can_transition(NodeState.RUNNING, NodeState.READY)  # plain retry
        assert can_transition(NodeState.FAILED, NodeState.PENDING)  # explicit retry

    def test_illegal_transitions(self):
        assert not can_transition(NodeState.SUCCEEDED, NodeState.RUNNING)
        assert not can_transition(NodeState.PENDING, NodeState.SUCCEEDED)
        assert not can_transition(NodeState.BLOCKED, NodeState.RUNNING)

    def test_terminal_states(self):
        assert NodeState.SUCCEEDED in TERMINAL_NODE_STATES
        assert NodeState.RUNNING not in TERMINAL_NODE_STATES

    def test_determine_run_status_priority(self):
        s = [NodeState.SUCCEEDED, NodeState.SUCCEEDED]
        assert determine_run_status(s) == RunState.SUCCEEDED
        assert determine_run_status([NodeState.SUCCEEDED, NodeState.FAILED]) == RunState.PARTIALLY_SUCCEEDED
        assert determine_run_status([NodeState.FAILED, NodeState.BLOCKED]) == RunState.FAILED
        assert determine_run_status([NodeState.CANCELLED, NodeState.FAILED]) == RunState.CANCELLED
        assert determine_run_status([]) == RunState.FAILED


class TestValidation:
    def test_valid_template(self):
        assert linear_template().validate() == []

    def test_empty_template_id(self):
        t = GraphTemplate(template_id=" ")
        assert any("template_id" in e for e in t.validate())

    def test_duplicate_ids(self):
        t = GraphTemplate(
            template_id="x",
            nodes=[NodeSpec(id="a", tool="t"), NodeSpec(id="a", tool="t")],
        )
        assert any("duplicate" in e for e in t.validate())

    def test_unknown_dep(self):
        t = GraphTemplate(
            template_id="x",
            nodes=[NodeSpec(id="a", tool="t", depends_on=["ghost"])],
        )
        assert any("unknown node" in e for e in t.validate())

    def test_self_dependency(self):
        t = GraphTemplate(template_id="x", nodes=[NodeSpec(id="a", tool="t", depends_on=["a"])])
        assert any("itself" in e for e in t.validate())

    def test_cycle_detection(self):
        t = GraphTemplate(
            template_id="x",
            nodes=[
                NodeSpec(id="a", tool="t", depends_on=["c"]),
                NodeSpec(id="b", tool="t", depends_on=["a"]),
                NodeSpec(id="c", tool="t", depends_on=["b"]),
            ],
        )
        errors = t.validate()
        assert any("cycle" in e for e in errors)

    def test_unknown_tool_flagged_with_registry(self):
        t = linear_template()
        assert t.validate(known_tools={"other"}) != []

    def test_roundtrip_serialization(self, tmp_path):
        t = linear_template()
        path = tmp_path / "t.json"
        path.write_text(json.dumps(t.to_dict()))
        loaded = load_template(path)
        assert loaded.template_id == "t_test"
        assert [n.id for n in loaded.nodes] == ["a", "b"]
        assert loaded.validate() == []


class TestCreateRunAndBindings:
    def test_create_run_merges_inputs(self):
        run = create_run(linear_template(), {"alpha": 42})
        assert run.inputs["alpha"] == 42
        assert set(run.nodes) == {"a", "b"}
        assert all(n.state == NodeState.PENDING for n in run.nodes.values())

    def test_unknown_input_rejected(self):
        with pytest.raises(ValueError, match="unknown template inputs"):
            create_run(linear_template(), {"beta": 1})

    def test_resolve_params_inputs(self):
        run = create_run(linear_template(), {"alpha": 7})
        assert resolve_params(linear_template().nodes[0].params, run) == {"x": 7}

    def test_resolve_params_node_outputs(self):
        run = create_run(linear_template())
        run.nodes["a"].state = NodeState.SUCCEEDED
        run.nodes["a"].outputs = {"v": 99}
        assert resolve_params(linear_template().nodes[1].params, run) == {"y": 99}

    def test_resolve_params_upstream_not_succeeded(self):
        run = create_run(linear_template())
        with pytest.raises(ValueError, match="not succeeded"):
            resolve_params(linear_template().nodes[1].params, run)

    def test_resolve_params_missing_output_key_raises(self):
        run = create_run(linear_template())
        run.nodes["a"].state = NodeState.SUCCEEDED
        run.nodes["a"].outputs = {"other": 1}
        with pytest.raises(ValueError, match="not produced by node"):
            resolve_params(linear_template().nodes[1].params, run)

    def test_resolve_params_undeclared_input_raises(self):
        run = create_run(linear_template())
        with pytest.raises(ValueError, match="undeclared template input"):
            resolve_params({"x": "${inputs.nope}"}, run)


class TestStore:
    def _populated(self, tmp_path):
        template = linear_template()
        store = SQLiteStateStore(tmp_path / "state.db")
        run = create_run(template, {"alpha": 5})
        run.nodes["a"].state = NodeState.SUCCEEDED
        run.nodes["a"].outputs = {"v": 1}
        run.nodes["b"].state = NodeState.RUNNING
        store.save_run(run)
        store.save_node_run(run, run.nodes["a"])
        store.save_node_run(run, run.nodes["b"])
        return store, template, run

    def test_save_and_load(self, tmp_path):
        store, template, run = self._populated(tmp_path)
        store2 = SQLiteStateStore(tmp_path / "state.db")
        loaded = store2.load_run(run.run_id, template)
        assert loaded.nodes["a"].state == NodeState.SUCCEEDED
        assert loaded.nodes["a"].outputs == {"v": 1}
        assert loaded.nodes["b"].state == NodeState.RUNNING
        assert loaded.inputs == {"alpha": 5}

    def test_resume_resets_inflight(self, tmp_path):
        store, template, run = self._populated(tmp_path)
        resumed = store.resume_run(run.run_id, template)
        assert resumed.nodes["a"].state == NodeState.SUCCEEDED  # idempotent skip
        assert resumed.nodes["b"].state == NodeState.PENDING  # re-dispatched

    def test_resume_terminal_run_untouched(self, tmp_path):
        store, template, run = self._populated(tmp_path)
        from dft_forge.runtime.states import RunState

        run.state = RunState.SUCCEEDED
        store.save_run(run)
        resumed = store.resume_run(run.run_id, template)
        assert resumed.state == RunState.SUCCEEDED

    def test_list_runs(self, tmp_path):
        store, template, run = self._populated(tmp_path)
        listing = store.list_runs()
        assert any(entry["run_id"] == run.run_id for entry in listing)

    def test_list_runs_numeric_ordering(self, tmp_path):
        # updated_at is stored as a float string; TEXT ordering would put
        # "9.5" after "10.5" — force both digit widths and check the order
        store, template, run = self._populated(tmp_path)
        import sqlite3

        with sqlite3.connect(tmp_path / "state.db") as conn:
            conn.execute("UPDATE graph_runs SET updated_at = '10.5' WHERE id = ?", (run.run_id,))
            conn.execute(
                "INSERT INTO graph_runs (id, template_id, status, data, created_at, updated_at) "
                "VALUES ('run_other', 't_test', 'created', '{}', '1.0', '9.5')"
            )
        listing = store.list_runs()
        assert [e["run_id"] for e in listing][:2] == [run.run_id, "run_other"]

    def test_template_id_for(self, tmp_path):
        store, template, run = self._populated(tmp_path)
        assert store.template_id_for(run.run_id) == "t_test"
        assert store.template_id_for("missing") is None

    def test_load_missing_returns_none(self, tmp_path):
        store = SQLiteStateStore(tmp_path / "state.db")
        assert store.load_run("nope", linear_template()) is None


class TestToolError:
    def test_fields(self):
        err = ToolError("boom", category="scf_not_converged", repairable=True)
        assert err.category == "scf_not_converged"
        assert err.repairable
        assert str(err) == "boom"
