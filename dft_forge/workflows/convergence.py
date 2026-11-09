"""Parameter-convergence sweep runner."""

from __future__ import annotations

from typing import Any, Dict

from dft_forge.runtime.graph import GraphTemplate, NodeSpec

from .common import (
    WorkflowError,
    _base_params,
    _runtime_controls,
    _workflow_id,
    _workflow_workdir,
)
from .energetics import convergence_grid, select_converged_point


def _run_graph(*args, **kwargs):
    """Resolve via the package namespace so tests can monkeypatch it."""
    import dft_forge.workflows as _pkg

    return _pkg._run_graph(*args, **kwargs)


def run_convergence(args: Dict[str, Any]) -> Dict[str, Any]:
    material = str(args.get("material") or "")
    structure = args.get("structure")
    if not material and not structure:
        raise WorkflowError("convergence workflow requires material or structure")
    points = convergence_grid(
        ecutwfc_values=args.get("ecutwfc_values"),
        kpoints_values=args.get("kpoints_values"),
        smearing_values=args.get("smearing_values"),
        degauss_values=args.get("degauss_values"),
        max_jobs=64 if args.get("max_jobs") is None else args.get("max_jobs"),
    )
    nodes = []
    for i, point in enumerate(points):
        params = {"calc": "scf", "prefix": f"conv_{i + 1}", **_base_params(args), **point}
        nodes.append(NodeSpec(id=f"point_{i + 1}", tool="qe", params=params, max_repair_attempts=1))
    template = GraphTemplate(
        template_id=_workflow_id("convergence", {"points": points, "material": material, "structure": structure}),
        description="parameter convergence sweep",
        nodes=nodes,
    )
    payload = _run_graph(template, _workflow_workdir(args, "convergence") / "graphs", **_runtime_controls(args))
    rows = []
    for i, point in enumerate(points):
        node = payload["nodes"][f"point_{i + 1}"]
        out = node.get("outputs") or {}
        rows.append({**point, "state": node.get("state"), "error": node.get("error"), **out})
    payload.update({"workflow_type": "convergence", "points": rows})
    if payload["state"] == "succeeded":
        try:
            payload["selection"] = select_converged_point(
                rows, threshold_mev_per_atom=float(args.get("threshold_mev_per_atom", 1.0))
            )
            selected = payload["selection"]
            payload["error_curve"] = {
                "kind": "convergence",
                "threshold_mev_per_atom": selected["threshold_mev_per_atom"],
                "points": [
                    {
                        "ecutwfc": point.get("ecutwfc"),
                        "kpoints": point.get("kpoints"),
                        "smearing": point.get("smearing"),
                        "degauss": point.get("degauss"),
                        "cost_proxy": point.get("cost_proxy"),
                        "delta_mev_per_atom": point.get("delta_mev_per_atom"),
                        "state": point.get("state"),
                    }
                    for point in selected["points"]
                ],
            }
        except WorkflowError as exc:
            payload["status"] = "partial"
            payload["blocking_reasons"] = [str(exc)]
    return payload
