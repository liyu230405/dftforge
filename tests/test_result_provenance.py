from __future__ import annotations

from dft_forge.agent_loop.provenance import provenance_warning, result_provenance


def _execution(backend="local", simulated=False, run_id="run_1", unknown=False):
    return {
        "run_id": run_id,
        "metadata": {
            "execution": {
                "origin_unknown": unknown,
                "origin": {"profile": {"backend": backend, "simulated": simulated}},
            }
        },
    }


def test_graph_execution_uses_persisted_profile_only():
    result = result_provenance("graph.run", _execution(backend="fake", simulated=True))
    assert result == {
        "source_kind": "execution",
        "execution_class": "simulated",
        "backend": "fake",
        "contains_simulated": True,
        "run_id": "run_1",
    }


def test_unknown_execution_does_not_read_environment():
    result = result_provenance("graph.status", _execution(backend="local", unknown=True))
    assert result["execution_class"] == "unknown"
    assert result["backend"] is None


def test_unknown_origin_still_warns_when_attempts_show_simulation():
    data = _execution(backend="fake", simulated=True, unknown=True)
    data["metadata"]["execution"]["attempts"] = [{"profile": {"backend": "fake", "simulated": True}}]
    result = result_provenance("graph.run", data)
    assert result["execution_class"] == "unknown"
    assert result["contains_simulated"] is True
    assert provenance_warning(result) == "执行来源不完整且包含模拟尝试（非科研结论）"


def test_run_id_macro_is_execution_without_metadata():
    result = result_provenance("workflow.convergence", {"run_id": "run_2"})
    assert result["source_kind"] == "execution"
    assert result["run_id"] == "run_2"


def test_non_graph_tool_cannot_promote_user_metadata_to_execution():
    result = result_provenance(
        "thermo.formation",
        {"run_id": "fake", "metadata": {"execution": {"origin": {"profile": {"backend": "local"}}}}},
    )
    assert result["source_kind"] == "user_input"
    assert result["execution_class"] == "unknown"


def test_explicit_and_parsed_results_are_not_claimed_as_compute():
    assert result_provenance("result.parse", None)["source_kind"] == "parsed"
    assert result_provenance("thermo.formation", {"energy": -1})["source_kind"] == "user_input"
    assert result_provenance("analysis.compare", {"differences": []})["execution_class"] == "unknown"


def test_derived_inputs_combine_simulated_and_compute():
    result = result_provenance(
        "thermo.formation",
        None,
        [
            result_provenance("graph.run", _execution(backend="fake", simulated=True, run_id="run_sim")),
            result_provenance("graph.run", _execution(backend="local", run_id="run_compute")),
        ],
    )
    # The side-channel entries are already summaries in production.
    assert result["source_kind"] == "derived"
    assert result["execution_class"] == "mixed"
    assert result["contains_simulated"] is True
    assert result["backend"] is None
    assert result["run_id"] is None


def test_all_compute_and_same_backend_stays_compute():
    result = result_provenance(
        "analysis.compare",
        None,
        [
            {"source_kind": "execution", "execution_class": "compute", "backend": "ssh", "run_id": "r"},
            {"source_kind": "execution", "execution_class": "compute", "backend": "ssh", "run_id": "r"},
        ],
    )
    assert result["execution_class"] == "compute"
    assert result["backend"] == "ssh"
    assert result["run_id"] == "r"


def test_explicit_unknown_input_is_not_dropped_from_combination():
    fake = {"source_kind": "execution", "execution_class": "simulated", "backend": "fake"}
    compute = {"source_kind": "execution", "execution_class": "compute", "backend": "local"}
    assert result_provenance("analysis.compare", None, [fake, None])["execution_class"] == "mixed"
    assert result_provenance("analysis.compare", None, [compute, None])["execution_class"] == "unknown"


def test_malformed_profile_values_are_unknown_not_type_errors():
    data = _execution(backend={"bad": 1}, simulated=None)
    result = result_provenance("graph.run", data)
    assert result["execution_class"] == "unknown"
    assert result["backend"] is None


def test_warning_only_for_simulated_or_mixed_simulated_results():
    assert provenance_warning(None) == ""
    assert provenance_warning({"execution_class": "compute", "contains_simulated": False}) == ""
    assert provenance_warning({"execution_class": "mixed", "contains_simulated": True}) == "模拟输出（非科研结论）"
    assert provenance_warning({"execution_class": "unknown", "contains_simulated": True}) == "执行来源不完整且包含模拟尝试（非科研结论）"
