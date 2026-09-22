"""Deterministic scientific workflow macros.

The chat planner selects one of these high-level capabilities.  The macro
then builds a typed graph and performs all scientific bookkeeping in Python;
the LLM never expands a composite calculation into an ad-hoc executable DAG.

This module intentionally keeps the first implementation conservative:
missing QE programs, missing outputs, incomplete competitor sets, and
ambiguous chemical potentials fail closed or return an explicit partial
status.  It does not implement charged-defect or finite-size corrections.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from dft_forge.parser import RY_TO_EV
from dft_forge.runtime.graph import GraphTemplate, NodeSpec


class WorkflowError(ValueError):
    """Invalid scientific workflow request or unusable workflow evidence."""


_MAX_CONVERGENCE_JOBS = 256


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


def convergence_grid(
    *,
    ecutwfc_values: Any = None,
    kpoints_values: Any = None,
    smearing_values: Any = None,
    degauss_values: Any = None,
    max_jobs: int = 64,
) -> List[Dict[str, Any]]:
    """Build a reproducible Cartesian parameter grid with a hard job cap."""
    def finite_positive(values: Any, name: str, default: Sequence[Any]) -> List[float]:
        converted: List[float] = []
        for raw in _positive_grid(values, name, default):
            if isinstance(raw, bool):
                raise WorkflowError(f"{name} values must be finite and positive")
            try:
                value = float(raw)
            except (TypeError, ValueError) as exc:
                raise WorkflowError(f"{name} values must be finite and positive") from exc
            if not math.isfinite(value) or value <= 0:
                raise WorkflowError(f"{name} values must be finite and positive")
            converted.append(value)
        return converted

    ecut = finite_positive(ecutwfc_values, "ecutwfc_values", [30, 40, 50, 60])
    kpts = [_normalise_kpoint(x) for x in _positive_grid(kpoints_values, "kpoints_values", [4, 6, 8])]
    smearing = [str(x) for x in _positive_grid(smearing_values, "smearing_values", ["mv"])]
    if any(not value.strip() for value in smearing):
        raise WorkflowError("smearing values must be non-empty")
    degauss = finite_positive(degauss_values, "degauss_values", [0.01])
    if isinstance(max_jobs, bool) or not isinstance(max_jobs, numbers.Integral):
        raise WorkflowError("max_jobs must be a positive integer")
    max_jobs = int(max_jobs)
    if max_jobs < 1 or max_jobs > _MAX_CONVERGENCE_JOBS:
        raise WorkflowError(
            f"max_jobs must be between 1 and {_MAX_CONVERGENCE_JOBS}"
        )
    n_points = len(ecut) * len(kpts) * len(smearing) * len(degauss)
    if n_points > max_jobs:
        raise WorkflowError(
            f"convergence grid has {n_points} jobs, above max_jobs={max_jobs}; "
            "reduce the grid explicitly"
        )
    points = [
        {"ecutwfc": e, "kpoints": k, "smearing": s, "degauss": d}
        for e in ecut for k in kpts for s in smearing for d in degauss
    ]
    return points


def select_converged_point(
    results: Sequence[Dict[str, Any]],
    *,
    threshold_mev_per_atom: float = 1.0,
) -> Dict[str, Any]:
    """Select the least-cost point converged to the densest/highest reference.

    Every successful point is compared with the highest cutoff + densest
    k-grid point for the same smearing/degauss pair.  A point is eligible only
    when that reference exists and its per-atom energy difference is within the
    requested tolerance.
    """
    if threshold_mev_per_atom <= 0:
        raise WorkflowError("threshold_mev_per_atom must be positive")
    good = [
        r for r in results
        if r.get("state") in (None, "succeeded", "success")
        and isinstance(r.get("energy_ry"), (int, float))
        and isinstance(r.get("natoms"), (int, float))
        and r.get("natoms", 0) > 0
    ]
    if not good:
        raise WorkflowError("no successful SCF points with energy and natoms evidence")

    def key(r: Dict[str, Any]) -> Tuple[float, int, int]:
        kp = r.get("kpoints") or [1, 1, 1]
        return float(r.get("ecutwfc") or 0), sum(int(x) for x in kp[:3]), len(kp)

    enriched = []
    groups: Dict[Tuple[str, float], List[Dict[str, Any]]] = {}
    for r in good:
        group = (str(r.get("smearing", "")), float(r.get("degauss", 0.0) or 0.0))
        groups.setdefault(group, []).append(r)
    references = {group: max(items, key=key) for group, items in groups.items()}
    for r in good:
        group = (str(r.get("smearing", "")), float(r.get("degauss", 0.0) or 0.0))
        reference = references[group]
        ref_per_atom = float(reference["energy_ry"]) / float(reference["natoms"])
        per_atom = float(r["energy_ry"]) / float(r["natoms"])
        delta = abs(per_atom - ref_per_atom) * RY_TO_EV * 1000.0
        kp = r.get("kpoints") or [1, 1, 1]
        cost = float(r.get("ecutwfc") or 0) * math.prod(int(x) for x in kp[:3])
        enriched.append({**r, "delta_mev_per_atom": delta, "cost_proxy": cost})
    eligible = [r for r in enriched if r["delta_mev_per_atom"] <= threshold_mev_per_atom]
    if not eligible:
        raise WorkflowError(
            f"no point converged to {threshold_mev_per_atom:g} meV/atom against the reference"
        )
    chosen = min(eligible, key=lambda r: (r["cost_proxy"], float(r.get("ecutwfc") or 0)))
    return {
        "selected": chosen,
        "reference": max(references.values(), key=key),
        "references_by_condition": [
            {"smearing": group[0], "degauss": group[1], "point": ref}
            for group, ref in references.items()
        ],
        "threshold_mev_per_atom": float(threshold_mev_per_atom),
        "points": enriched,
    }


def vacancy_formation_energy(
    *,
    defect_energy_ry: float,
    pristine_energy_ry: float,
    chemical_potential_ry: float,
    removed_atoms: int = 1,
) -> Dict[str, float]:
    """Neutral vacancy formation energy using a per-atom chemical potential.

    For a vacancy, ``E_f = E_def - E_bulk + n_removed * mu_removed``.
    The caller must provide ``mu_removed`` explicitly (for example half the
    total O2 energy); this function never guesses a reference state.
    """
    if removed_atoms < 1:
        raise WorkflowError("removed_atoms must be >= 1")
    ry = float(defect_energy_ry) - float(pristine_energy_ry) + removed_atoms * float(chemical_potential_ry)
    return {"formation_energy_ry": ry, "formation_energy_ev": ry * RY_TO_EV}


def _composition(formula: str) -> Dict[str, float]:
    try:
        from pymatgen.core import Composition

        data = Composition(str(formula)).as_dict()
    except Exception as exc:
        raise WorkflowError(f"invalid chemical formula {formula!r}: {exc}") from exc
    if not data or len(data) not in (2, 3) or any(float(v) <= 0 for v in data.values()):
        raise WorkflowError("chemical-potential scan currently supports binary/ternary formulas")
    return {str(k): float(v) for k, v in data.items()}


def _clip_polygon(poly: List[Tuple[float, float]], a: float, b: float, c: float) -> List[Tuple[float, float]]:
    """Clip polygon by a*x+b*y <= c."""
    if not poly:
        return []
    out: List[Tuple[float, float]] = []
    for p, q in zip(poly, poly[1:] + poly[:1]):
        fp = a * p[0] + b * p[1] - c
        fq = a * q[0] + b * q[1] - c
        pin, qin = fp <= 1e-10, fq <= 1e-10
        if pin:
            out.append(p)
        if pin != qin:
            t = fp / (fp - fq)
            out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return out


def chemical_potential_scan(
    *,
    formula: str,
    formation_energy_ev_per_formula: float,
    competitors: Optional[Sequence[Dict[str, Any]]] = None,
    n_points: int = 21,
) -> Dict[str, Any]:
    """Return the elemental-reference chemical-potential stability domain.

    The parent compound is imposed as an equality and all elemental chemical
    potentials satisfy Δμ_i <= 0.  Optional competitors add half-plane
    constraints.  Without an exhaustive competitor set the result is marked
    partial rather than being called a convex-hull stability region.
    """
    comp = _composition(formula)
    elems = list(comp)
    hf = float(formation_energy_ev_per_formula)
    if not math.isfinite(hf) or hf > 1e-10:
        raise WorkflowError("formation energy must be finite and normally non-positive")
    competitors = list(competitors or [])
    n_points = max(2, min(int(n_points), 10000))
    clean_competitors: List[Tuple[Dict[str, Any], Dict[str, float], float]] = []
    for item in competitors:
        if not isinstance(item, dict) or not item.get("formula"):
            raise WorkflowError("each competitor must provide a formula and formation_energy_ev_per_formula")
        try:
            competitor_comp = _composition(str(item["formula"]))
            competitor_hf = float(item["formation_energy_ev_per_formula"])
        except (KeyError, TypeError, ValueError) as exc:
            raise WorkflowError(
                "each competitor must provide a formula and finite formation_energy_ev_per_formula"
            ) from exc
        if not math.isfinite(competitor_hf):
            raise WorkflowError("competitor formation energies must be finite")
        extra = set(competitor_comp) - set(comp)
        if extra:
            raise WorkflowError(
                f"competitor {item['formula']} contains elements outside parent {formula}: {sorted(extra)}"
            )
        clean_competitors.append((item, competitor_comp, competitor_hf))

    if len(elems) == 2:
        a, b = elems
        na, nb = comp[a], comp[b]
        # Δμ_a is the scan coordinate; Δμ_b follows the parent equality.
        lo, hi = hf / na, 0.0
        bounds = [(1.0, 0.0, -lo), (-1.0, 0.0, -hi)]  # x >= lo, x <= hi
        for item, c, h in clean_competitors:
            ca, cb = c.get(a, 0.0), c.get(b, 0.0)
            coeff = ca - cb * na / nb
            rhs = h - cb * hf / nb
            if abs(coeff) <= 1e-12:
                if rhs < -1e-10:
                    raise WorkflowError(
                        f"competitor {item.get('formula')} makes the chemical-potential interval empty"
                    )
            else:
                bounds.append((coeff, 0.0, rhs))
        # Solve 1-D inequalities explicitly and retain a clear reason for
        # infeasible competitor sets.
        for coeff, _, rhs in bounds[2:]:
            if coeff > 0:
                hi = min(hi, rhs / coeff)
            elif coeff < 0:
                lo = max(lo, rhs / coeff)
        if lo > hi + 1e-10:
            raise WorkflowError("chemical-potential constraints have an empty feasible interval")
        points = []
        for i in range(n_points):
            x = lo + (hi - lo) * i / (n_points - 1)
            y = (hf - na * x) / nb
            points.append({f"delta_mu_{a}_ev": x, f"delta_mu_{b}_ev": y})
        return {
            "formula": formula,
            "elements": elems,
            "formation_energy_ev_per_formula": hf,
            "domain_dimension": 1,
            "interval": {a: [lo, hi]},
            "points": points,
            "constraints": {"elemental_bounds": True, "competitors_supplied": bool(competitors)},
            "status": "complete" if competitors else "partial",
            "warning": "未提供完整竞争相集合，不能据此声称位于 convex hull" if not competitors else None,
        }

    # Three components: eliminate the third chemical potential and clip a
    # conservative square by elemental and competitor half-planes.
    a, b, c = elems
    na, nb, nc = comp[a], comp[b], comp[c]
    scale = max(1.0, abs(hf) * 2.0 + 1.0)
    poly = [(-scale, -scale), (scale, -scale), (scale, scale), (-scale, scale)]
    # Δμ_a <= 0, Δμ_b <= 0, Δμ_c <= 0 where
    # Δμ_c = (hf - na*x - nb*y) / nc.
    poly = _clip_polygon(poly, 1.0, 0.0, 0.0)
    poly = _clip_polygon(poly, 0.0, 1.0, 0.0)
    poly = _clip_polygon(poly, -na / nc, -nb / nc, -hf / nc)
    for item, cc, h in clean_competitors:
        ca, cb, ccc = cc.get(a, 0.0), cc.get(b, 0.0), cc.get(c, 0.0)
        poly = _clip_polygon(
            poly,
            ca - ccc * na / nc,
            cb - ccc * nb / nc,
            h - ccc * hf / nc,
        )
    if not poly:
        raise WorkflowError("chemical-potential constraints have an empty feasible polygon")
    vertices = [
        {f"delta_mu_{a}_ev": x, f"delta_mu_{b}_ev": y,
         f"delta_mu_{c}_ev": (hf - na * x - nb * y) / nc}
        for x, y in poly
    ]
    return {
        "formula": formula,
        "elements": elems,
        "formation_energy_ev_per_formula": hf,
        "domain_dimension": 2,
        "vertices": vertices,
        "constraints": {"elemental_bounds": True, "competitors_supplied": bool(competitors)},
        "status": "complete" if competitors else "partial",
        "warning": "未提供完整竞争相集合，不能据此声称位于 convex hull" if not competitors else None,
    }


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
    return _graph_payload(engine, template, run)


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


def _write_structure(result: Any, path: Path) -> str:
    from ase.io import write

    path.parent.mkdir(parents=True, exist_ok=True)
    write(str(path), result.atoms, format="cif")
    return str(path)


def run_neutral_vacancy(args: Dict[str, Any]) -> Dict[str, Any]:
    if int(args.get("charge", 0) or 0) != 0:
        raise WorkflowError("neutral_vacancy only accepts charge=0; charged defects need a charged-DFT/correction workflow")
    source = str(args.get("source") or "")
    element = str(args.get("element") or "")
    if not source or not element:
        raise WorkflowError("neutral_vacancy requires source and element")
    workdir = _workflow_workdir(args, "neutral_vacancy")
    from dft_forge.structure_builder import supercell_3d, vacancy_3d

    pristine = supercell_3d(source, supercell=str(args.get("supercell", "2x2x2")))
    defect = vacancy_3d(source, element, supercell=str(args.get("supercell", "2x2x2")), index=args.get("index"))
    pristine_path = _write_structure(pristine, workdir / "pristine.cif")
    defect_path = _write_structure(defect, workdir / "vacancy.cif")
    base = _base_params(args)
    base.pop("structure", None)
    base["material"] = source
    base.setdefault("conv_thr", 1e-8 if args.get("conv_thr") in (None, "") else args["conv_thr"])
    nodes = [
        NodeSpec(
            id="pristine_scf", tool="qe",
            params={"calc": "scf", "prefix": "pristine", "structure": pristine_path, **base},
        ),
    ]
    if bool(args.get("relax", True)):
        nodes.append(NodeSpec(
            id="defect_relax", tool="qe",
            params={"calc": "relax", "prefix": "defect", "structure": defect_path, **base},
        ))
        defect_structure = "${nodes.defect_relax.outputs.final_structure_cif}"
        defect_dep = ["defect_relax"]
    else:
        defect_structure = defect_path
        defect_dep = []
    nodes.append(NodeSpec(
        id="defect_scf", tool="qe",
        params={"calc": "scf", "prefix": "defect_scf", "structure": defect_structure, **base},
        depends_on=defect_dep,
    ))
    template = GraphTemplate(
        template_id=_workflow_id("neutral_vacancy", {"source": source, "element": element, "supercell": args.get("supercell"), "relax": bool(args.get("relax", True))}),
        description="neutral bulk vacancy formation workflow",
        nodes=nodes,
    )
    payload = _run_graph(template, workdir / "graphs", **_runtime_controls(args))
    p_out = payload["nodes"]["pristine_scf"].get("outputs") or {}
    d_out = payload["nodes"]["defect_scf"].get("outputs") or {}
    payload["workflow_type"] = "neutral_vacancy"
    payload["structure"] = {
        "pristine": _jsonable(pristine.to_dict()),
        "defect": _jsonable(defect.to_dict()),
    }
    mu_ry = args.get("chemical_potential_ry")
    if mu_ry is None and args.get("chemical_potential_ev") is not None:
        mu_ry = float(args["chemical_potential_ev"]) / RY_TO_EV
    if mu_ry is None:
        payload["status"] = "partial"
        payload["blocking_reasons"] = ["chemical_potential_ry/ev is required; refuse to guess the removed-atom reference"]
    elif isinstance(p_out.get("energy_ry"), (int, float)) and isinstance(d_out.get("energy_ry"), (int, float)):
        payload["formation"] = vacancy_formation_energy(
            defect_energy_ry=d_out["energy_ry"],
            pristine_energy_ry=p_out["energy_ry"],
            chemical_potential_ry=float(mu_ry),
        )
    return payload


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


__all__ = [
    "WorkflowError", "convergence_grid", "select_converged_point",
    "vacancy_formation_energy", "chemical_potential_scan",
    "compare_magnetic_results", "phonon_stability", "run_convergence",
    "run_neutral_vacancy", "run_magnetic_compare", "run_phonon_dfpt",
]
