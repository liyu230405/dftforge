"""Phonon stability reducer and DFPT runner."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

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


def phonon_stability(frequencies_thz: Sequence[float], *, imaginary_threshold_thz: float = 0.1) -> Dict[str, Any]:
    freqs = [float(x) for x in frequencies_thz]
    if not freqs:
        raise WorkflowError("phonon output contains no frequencies")
    threshold = abs(float(imaginary_threshold_thz))
    imaginary = [x for x in freqs if x < -threshold]
    soft = [x for x in freqs if abs(x) <= threshold]
    return {
        "n_modes": len(freqs),
        "min_frequency_thz": min(freqs),
        "imaginary_count": len(imaginary),
        "soft_count": len(soft),
        "dynamically_stable_at_threshold": not imaginary,
        "imaginary_threshold_thz": threshold,
    }


def run_phonon_dfpt(args: Dict[str, Any]) -> Dict[str, Any]:
    material = str(args.get("material") or "MgO")
    q_grid = [int(x) for x in (args.get("q_grid") or [2, 2, 2])]
    if len(q_grid) != 3 or any(x < 1 for x in q_grid):
        raise WorkflowError("q_grid must contain three positive integers")
    # The user structure is only the input to pre-relaxation.  Subsequent
    # nodes must consume the relaxed artifact, not silently fall back to the
    # original geometry.
    base = {k: v for k, v in {"material": material, **_base_params(args)}.items() if k != "structure"}
    postprocess_base = {"material": material}
    do_relax = bool(args.get("relax", True))
    nodes: List[NodeSpec] = []
    if do_relax:
        relax_params = {"calc": "vc-relax", "prefix": "phonon_relax", **base}
        if args.get("structure"):
            relax_params["structure"] = args["structure"]
        nodes.append(NodeSpec(id="relax", tool="qe", params=relax_params))
        common_structure = "${nodes.relax.outputs.final_structure_cif}"
        scf_dep = ["relax"]
    else:
        common_structure = args.get("structure")
        scf_dep = []
    scf_params = {"calc": "scf", "prefix": "phonon", **base}
    if common_structure:
        scf_params["structure"] = common_structure
    nodes.extend([
        NodeSpec(id="scf", tool="qe", params=scf_params, depends_on=scf_dep),
        NodeSpec(
            id="ph", tool="qe",
            params={
                "calc": "ph", "prefix": "phonon", "q_grid": q_grid,
                "tr2_ph": float(args.get("tr2_ph", 1.0e-14)), **base,
            },
            depends_on=["scf"],
        ),
        NodeSpec(
            id="q2r", tool="qe",
            params={
                "calc": "q2r", "prefix": "phonon", "dynamics_prefix": "phonon.dyn", **postprocess_base,
            },
            depends_on=["ph"],
        ),
        NodeSpec(
            id="matdyn", tool="qe",
            params={
                "calc": "matdyn", "prefix": "phonon", "force_constants_prefix": "phonon.fc",
                "q_grid": q_grid, **postprocess_base,
            },
            depends_on=["q2r"],
        ),
    ])
    template = GraphTemplate(
        template_id=_workflow_id("phonon_dfpt", {"material": material, "q_grid": q_grid, "structure": args.get("structure")}),
        description="relax → SCF → ph.x → q2r.x → matdyn.x phonon workflow",
        nodes=nodes,
    )
    payload = _run_graph(template, _workflow_workdir(args, "phonon_dfpt") / "graphs", **_runtime_controls(args))
    payload["workflow_type"] = "phonon_dfpt"
    out = payload["nodes"].get("matdyn", {}).get("outputs") or {}
    if out.get("frequencies_thz"):
        payload["stability"] = phonon_stability(out["frequencies_thz"], imaginary_threshold_thz=float(args.get("imaginary_threshold_thz", 0.1)))
    return payload
