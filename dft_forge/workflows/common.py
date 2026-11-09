"""Shared helpers for scientific workflow macros."""

from __future__ import annotations

import hashlib
import json
import math
import numbers
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from dft_forge.runtime.graph import GraphTemplate


class WorkflowError(ValueError):
    """Invalid scientific workflow request or unusable workflow evidence."""


_MAX_CONVERGENCE_JOBS = 256


def _finite_real(value: Any) -> bool:
    """Return true only for finite, non-boolean real-valued evidence."""
    return (
        isinstance(value, numbers.Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _matching_atom_count(value: Any, expected: int) -> bool:
    """Atom counts are discrete evidence; ``True == 1`` must not pass."""
    return (
        isinstance(value, numbers.Integral)
        and not isinstance(value, bool)
        and int(value) == int(expected)
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        try:
            return _jsonable(value.tolist())
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _workflow_id(prefix: str, args: Dict[str, Any]) -> str:
    payload = json.dumps(args, sort_keys=True, default=str, ensure_ascii=False)
    return f"{prefix}_{hashlib.sha1(payload.encode()).hexdigest()[:10]}"


def _normalise_kpoint(value: Any) -> List[int]:
    def strict_int(item: Any, label: str) -> int:
        if isinstance(item, bool) or not isinstance(item, numbers.Integral):
            raise WorkflowError(f"{label} must contain strict integer values")
        return int(item)

    if isinstance(value, bool):
        raise WorkflowError("k-point grid must contain strict integer values")
    if isinstance(value, numbers.Integral):
        if value < 1:
            raise WorkflowError("k-point grid values must be positive")
        value = int(value)
        return [value, value, value, 1, 1, 1]
    if isinstance(value, (list, tuple)) and len(value) in (3, 6):
        vals = [strict_int(x, "k-point grid") for x in value]
        if any(x < 1 for x in vals[:3]):
            raise WorkflowError("k-point grid values must be positive")
        if len(vals) == 6 and any(x not in (0, 1) for x in vals[3:]):
            raise WorkflowError("k-point grid offsets must be 0 or 1")
        return vals if len(vals) == 6 else vals + [1, 1, 1]
    raise WorkflowError(f"invalid k-point grid: {value!r}")


def _positive_grid(values: Any, name: str, default: Sequence[Any]) -> List[Any]:
    raw = default if values in (None, "") else values
    if not isinstance(raw, (list, tuple)) or not raw:
        raise WorkflowError(f"{name} must be a non-empty array")
    return list(raw)


def _profile_common_method(output: Dict[str, Any]) -> Dict[str, Any]:
    """Small method signature used before combining independently run nodes."""
    profile = output.get("method_profile") if isinstance(output, dict) else None
    profile = profile if isinstance(profile, dict) else {}
    effective = profile.get("effective") if isinstance(profile.get("effective"), dict) else {}
    return {
        "pseudos": dict(profile.get("pseudos") or {}),
        "input_dft": profile.get("input_dft"),
        "ecutwfc": effective.get("ecutwfc"),
        "ecutrho": effective.get("ecutrho"),
        "kpoints": list(effective.get("kpoints") or []),
    }


def _require_method_compatibility(
    left: Dict[str, Any], right: Dict[str, Any], *,
    compare_kpoints: bool = True, common_elements: Optional[Iterable[str]] = None,
) -> None:
    """Reject thermodynamic reductions with incompatible electronic methods."""
    a, b = _profile_common_method(left), _profile_common_method(right)
    for label, signature in (("left", a), ("right", b)):
        if not (_finite_real(signature.get("ecutwfc")) and float(signature["ecutwfc"]) > 0):
            raise WorkflowError(f"{label} method profile lacks finite positive ecutwfc evidence")
        if not (_finite_real(signature.get("ecutrho")) and float(signature["ecutrho"]) > 0):
            raise WorkflowError(f"{label} method profile lacks finite positive ecutrho evidence")
        kpoints = signature.get("kpoints")
        if (not isinstance(kpoints, list) or len(kpoints) not in (3, 6)
                or any(isinstance(value, bool) or not isinstance(value, numbers.Integral)
                       for value in kpoints)):
            raise WorkflowError(f"{label} method profile lacks valid kpoints evidence")
    for key in ("ecutwfc", "ecutrho", "input_dft"):
        if a.get(key) != b.get(key):
            raise WorkflowError(f"method mismatch for {key}: {a.get(key)!r} != {b.get(key)!r}")
    if compare_kpoints and a.get("kpoints") != b.get("kpoints"):
        raise WorkflowError(f"method mismatch for kpoints: {a.get('kpoints')!r} != {b.get('kpoints')!r}")
    elements = set(common_elements or (set(a["pseudos"]) & set(b["pseudos"])))
    if not elements:
        raise WorkflowError("method comparison has no shared-element pseudopotential evidence")
    for element in sorted(elements):
        if not a["pseudos"].get(element) or not b["pseudos"].get(element):
            raise WorkflowError(
                f"method profile lacks pseudopotential evidence for shared element {element}"
            )
        if a["pseudos"].get(element) != b["pseudos"].get(element):
            raise WorkflowError(f"pseudopotential mismatch for shared element {element}")


def _graph_payload(engine: Any, template: GraphTemplate, run: Any) -> Dict[str, Any]:
    from dft_forge.tools.graph_tools import _resolve_template_outputs

    nodes = {
        nid: {
            "state": node.state.value,
            "error": node.error,
            "outputs": _jsonable(node.outputs) if node.state.value == "succeeded" else None,
        }
        for nid, node in run.nodes.items()
    }
    return {
        "run_id": run.run_id,
        "state": run.state.value,
        "nodes": nodes,
        "outputs": _jsonable(_resolve_template_outputs(template, run)),
        "metadata": getattr(run, "metadata", {}),
    }


def _run_graph(
    template: GraphTemplate, workdir: Path, inputs: Optional[Dict[str, Any]] = None,
    *, cancel_event=None, on_event=None, method_contract=None,
) -> Dict[str, Any]:
    from dft_forge.tools.graph_tools import _get_engine

    engine = _get_engine(str(workdir))
    engine.register_template(template)
    if method_contract:
        run = engine.run_template(
            template.template_id,
            inputs or {},
            cancel_event=cancel_event,
            on_event=on_event,
            run_metadata={"method_contract": method_contract},
        )
    else:
        run = engine.run_template(
            template.template_id, inputs or {}, cancel_event=cancel_event, on_event=on_event,
        )
    # Resolve through the package namespace so tests can monkeypatch
    # ``dft_forge.workflows._graph_payload``.
    import dft_forge.workflows as _pkg

    return _pkg._graph_payload(engine, template, run)


def _runtime_controls(args: Dict[str, Any]) -> Dict[str, Any]:
    """Runtime-only handles are injected by the caller, never stored in a DAG."""
    controls = {
        name: args.pop(f"_{name}")
        for name in ("cancel_event", "on_event")
        if f"_{name}" in args
    }
    if "_method_contract" in args:
        controls["method_contract"] = args.pop("_method_contract")
    return controls


def _base_params(args: Dict[str, Any]) -> Dict[str, Any]:
    return {
        k: args[k] for k in ("material", "structure", "ecutwfc", "ecutrho", "kpoints", "conv_thr")
        if args.get(k) not in (None, "")
    }


def _workflow_workdir(args: Dict[str, Any], name: str) -> Path:
    """Resolve a workspace for direct registry calls as well as chat calls."""
    raw = args.get("workdir")
    return Path(str(raw)) if raw not in (None, "") else Path.cwd() / "work" / "workflows" / name


def _write_structure(result: Any, path: Path) -> str:
    from ase.io import write

    path.parent.mkdir(parents=True, exist_ok=True)
    write(str(path), result.atoms, format="cif")
    return str(path)


def _charge_value(mapping: Any, charge: int, label: str, *, default: Any = None) -> Any:
    if not isinstance(mapping, dict):
        if default is not None:
            return default
        raise WorkflowError(f"{label} must be an object keyed by charge")
    for key in (str(charge), charge, f"{charge:+d}"):
        if key in mapping:
            return mapping[key]
    if default is not None:
        return default
    raise WorkflowError(f"{label} is missing charge {charge:+d}")


def _charge_node_id(charge: int) -> str:
    return "q_0" if charge == 0 else f"q_{'p' if charge > 0 else 'm'}{abs(charge)}"
