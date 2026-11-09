"""Magnetic-state comparison reducer and runner."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from dft_forge.parser import RY_TO_EV
from dft_forge.runtime.graph import GraphTemplate, NodeSpec

from .common import (
    WorkflowError,
    _base_params,
    _runtime_controls,
    _workflow_id,
    _workflow_workdir,
)


def _run_graph(*args, **kwargs):
    """Resolve via the package namespace so tests can monkeypatch it."""
    import dft_forge.workflows as _pkg
    return _pkg._run_graph(*args, **kwargs)


def compare_magnetic_results(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Compare magnetic states and flag a requested state that collapsed."""
    usable = []
    warnings: List[str] = []
    for item in results:
        if not isinstance(item.get("energy_ry"), (int, float)):
            continue
        nat = float(item.get("natoms") or 1)
        row = dict(item)
        row["energy_ev_per_atom"] = float(item["energy_ry"]) * RY_TO_EV / nat
        order = str(item.get("requested_order") or item.get("order") or "").upper()
        mags = [float(x) for x in (item.get("atom_magnetizations_muB") or []) if isinstance(x, (int, float))]
        total = item.get("total_magnetization_muB")
        absolute = item.get("absolute_magnetization_muB")
        total = float(total) if isinstance(total, (int, float)) else None
        absolute = float(absolute) if isinstance(absolute, (int, float)) else None
        row["realized_order"] = order
        if order == "FM":
            magnitude = max(abs(x) for x in mags) if mags else abs(total) if total is not None else None
            if magnitude is not None and magnitude < 0.1:
                row["state_status"] = "collapsed"
                warnings.append("FM 初始磁序收敛为近似零磁矩，不能当作稳定 FM 态")
            elif magnitude is None:
                row["state_status"] = "unverified"
                warnings.append("FM 缺少总磁矩或逐原子磁矩，无法确认磁性是否保留")
            else:
                row["state_status"] = "available"
        elif order == "AFM":
            if mags:
                if any(x > 0.1 for x in mags) and any(x < -0.1 for x in mags):
                    row["state_status"] = "available"
                else:
                    row["state_status"] = "collapsed"
                    warnings.append("AFM 初始磁序没有保留相反子晶格磁矩")
            elif total is not None and absolute is not None and abs(total) < 0.1 and absolute > 0.1:
                row["state_status"] = "partially_verified"
                warnings.append("AFM 仅由近零总磁矩和非零绝对磁矩支持；需逐原子磁矩确认子晶格")
            else:
                row["state_status"] = "unverified"
                warnings.append("AFM 缺少逐原子磁矩，无法确认相反子晶格")
        else:
            row["state_status"] = "available"
        usable.append(row)
    if not usable:
        raise WorkflowError("no magnetic state produced a total energy")
    ground = min(usable, key=lambda x: x["energy_ev_per_atom"])
    return {
        "states": usable,
        "ground_state": ground.get("requested_order"),
        "ground_state_status": ground.get("state_status"),
        "warnings": sorted(set(warnings)),
    }


def run_magnetic_compare(args: Dict[str, Any]) -> Dict[str, Any]:
    material = str(args.get("material") or "Fe")
    orders = [str(x).upper() for x in (args.get("orders") or ["NM", "FM", "AFM"])]
    if not orders or any(x not in {"NM", "FM", "AFM"} for x in orders):
        raise WorkflowError("orders must contain only NM, FM, or AFM")
    if len(set(orders)) != len(orders):
        raise WorkflowError("orders must not contain duplicates")
    base = {"material": material, **_base_params(args)}
    nodes = []
    for order in orders:
        nodes.append(NodeSpec(
            id=order.lower(), tool="qe",
            params={
                "calc": "scf", "prefix": f"mag_{order.lower()}",
                "magnetic_order": order.lower(),
                "magnetic_element": args.get("magnetic_element"),
                "initial_moment_muB": float(args.get("initial_moment_muB", 2.2)),
                "supercell": str(args.get("supercell", "1x1x1")),
                **base,
            },
        ))
    template = GraphTemplate(
        template_id=_workflow_id("magnetic_compare", {"material": material, "orders": orders, "supercell": args.get("supercell")}),
        description="collinear NM/FM/AFM magnetic comparison",
        nodes=nodes,
    )
    payload = _run_graph(template, _workflow_workdir(args, "magnetic_compare") / "graphs", **_runtime_controls(args))
    rows = []
    for order in orders:
        out = payload["nodes"][order.lower()].get("outputs") or {}
        rows.append({"requested_order": order, **out})
    payload["workflow_type"] = "magnetic_compare"
    payload["comparison"] = compare_magnetic_results(rows) if any(r.get("energy_ry") is not None for r in rows) else None
    return payload
