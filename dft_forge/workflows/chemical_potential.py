"""Chemical-potential stability-domain scan (binary/ternary)."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .common import WorkflowError


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
