"""Small, conservative provenance summaries for agent-loop results."""

from __future__ import annotations

from typing import Any, Iterable, Optional

_SOURCE_KINDS = {"execution", "derived", "parsed", "user_input", "unknown"}
_EXECUTION_CLASSES = {"simulated", "compute", "unknown", "mixed"}
_BACKENDS = {"fake", "local", "ssh"}
_GRAPH_TOOLS = {"graph.run", "graph.status"}
_GRAPH_BACKED_WORKFLOWS = {
    "workflow.convergence",
    "workflow.neutral_vacancy",
    "workflow.charged_vacancy",
    "workflow.adsorption_scan",
    "workflow.magnetic_compare",
    "workflow.phonon_dfpt",
}
_EXPLICIT_TOOLS = {
    "thermo.eads",
    "thermo.formation",
    "thermo.chemical_potential_scan",
    "chemical_potential_scan",
    "analysis.compare",
}


def _empty() -> dict[str, Any]:
    return {
        "source_kind": "unknown",
        "execution_class": "unknown",
        "backend": None,
        "contains_simulated": False,
        "run_id": None,
    }


def _run_id(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and value else None


def _execution_from_data(data: Any) -> dict[str, Any]:
    result = _empty()
    if not isinstance(data, dict):
        return result
    metadata = data.get("metadata")
    execution = metadata.get("execution") if isinstance(metadata, dict) else None
    if not isinstance(execution, dict):
        return result
    result["run_id"] = _run_id(data.get("run_id"))
    if execution.get("origin_unknown") is True:
        attempts = execution.get("attempts")
        if isinstance(attempts, list):
            result["contains_simulated"] = any(
                isinstance(item, dict)
                and isinstance(item.get("profile"), dict)
                and (item["profile"].get("simulated") is True or item["profile"].get("backend") == "fake")
                for item in attempts
            )
        return result
    origin = execution.get("origin")
    profile = origin.get("profile") if isinstance(origin, dict) else None
    if not isinstance(profile, dict):
        return result
    raw_backend = profile.get("backend")
    backend = raw_backend if isinstance(raw_backend, str) and raw_backend in _BACKENDS else None
    simulated_value = profile.get("simulated")
    simulated = raw_backend == "fake" or simulated_value is True
    if simulated:
        result["execution_class"] = "simulated"
        result["contains_simulated"] = True
    elif backend in {"local", "ssh"} and simulated_value is False:
        result["execution_class"] = "compute"
    result["backend"] = backend
    return result


def _normalise_input(value: Any) -> Optional[dict[str, Any]]:
    if not isinstance(value, dict):
        return None
    source = value.get("source_kind")
    execution_class = value.get("execution_class")
    if not isinstance(source, str) or source not in _SOURCE_KINDS:
        source = "unknown"
    if not isinstance(execution_class, str) or execution_class not in _EXECUTION_CLASSES:
        execution_class = "unknown"
    raw_backend = value.get("backend")
    backend = raw_backend if isinstance(raw_backend, str) and raw_backend in _BACKENDS else None
    return {
        "source_kind": source,
        "execution_class": execution_class,
        "backend": backend,
        "contains_simulated": bool(value.get("contains_simulated")) or execution_class == "simulated",
        "run_id": _run_id(value.get("run_id")),
    }


def _flatten(values: Any) -> list[dict[str, Any]]:
    if values is None:
        return [_empty()]
    if isinstance(values, (list, tuple)):
        if not values:
            return [_empty()]
        result: list[dict[str, Any]] = []
        for value in values:
            result.extend(_flatten(value))
        return result
    item = _normalise_input(values)
    return [item] if item is not None else [_empty()]


def _combine(items: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = list(items)
    if not values:
        return _empty()
    has_simulated = any(item["contains_simulated"] or item["execution_class"] == "simulated" for item in values)
    if has_simulated:
        execution_class = "simulated" if all(item["execution_class"] == "simulated" for item in values) else "mixed"
    elif all(item["execution_class"] == "compute" for item in values):
        execution_class = "compute"
    else:
        execution_class = "unknown"
    backends = {item["backend"] for item in values}
    backend = next(iter(backends)) if len(backends) == 1 else None
    run_ids = {item["run_id"] for item in values if item["run_id"] is not None}
    return {
        "source_kind": "derived",
        "execution_class": execution_class,
        "backend": backend,
        "contains_simulated": has_simulated,
        "run_id": next(iter(run_ids)) if len(run_ids) == 1 else None,
    }


def result_provenance(tool_id: str, data: Any, input_provenance: Any = None) -> dict[str, Any]:
    """Return a conservative, JSON-safe provenance summary.

    ``input_provenance`` is a private runtime side channel. User-facing tool
    arguments and environment variables are intentionally never consulted.
    """
    tool_id = str(tool_id or "")
    if tool_id == "result.parse":
        result = _empty()
        result["source_kind"] = "parsed"
        return result

    if tool_id in _GRAPH_TOOLS or (
        tool_id in _GRAPH_BACKED_WORKFLOWS and isinstance(data, dict) and data.get("run_id")
    ):
        result = _execution_from_data(data)
        result["source_kind"] = "execution"
        if isinstance(data, dict):
            result["run_id"] = _run_id(data.get("run_id"))
        return result

    if input_provenance is not None:
        return _combine(_flatten(input_provenance))

    if tool_id in _EXPLICIT_TOOLS:
        result = _empty()
        result["source_kind"] = "user_input"
        return result

    return _empty()


def provenance_warning(provenance: Optional[dict[str, Any]]) -> str:
    if not isinstance(provenance, dict):
        return ""
    if provenance.get("execution_class") == "unknown" and provenance.get("contains_simulated") is True:
        return "执行来源不完整且包含模拟尝试（非科研结论）"
    if provenance.get("contains_simulated") is True or provenance.get("execution_class") == "simulated":
        return "模拟输出（非科研结论）"
    return ""
