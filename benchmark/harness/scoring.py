"""Scoring curves (SCORING.md §2.2).

Borrowed from silicolab bench `_graded_score`: `tolerance` is the ZERO line,
full marks at err <= 0.3*tol, linear in between. ~3.3x stricter than the usual
"inside tol = full marks".
"""
from __future__ import annotations

FULL_MARK_FRACTION = 0.3


def graded_score(err: float, tol: float) -> float:
    """err/tol -> [0, 1] on the tightened curve."""
    if tol <= 0:
        return 1.0 if err == 0 else 0.0
    if err < 0:
        return 0.0
    full = FULL_MARK_FRACTION * tol
    if err <= full:
        return 1.0
    if err >= tol:
        return 0.0
    return (tol - err) / (tol - full)


def upper_bound_score(value: float, limit: float) -> float:
    """value <= limit -> 1, then decays; 0 once value >= 2*limit."""
    if value <= limit:
        return 1.0
    if limit <= 0:
        return 0.0
    over = (value - limit) / limit
    return max(0.0, 1.0 - over)


def lower_bound_score(value: float, limit: float) -> float:
    if value >= limit:
        return 1.0
    if limit <= 0:
        return 0.0
    under = (limit - value) / limit
    return max(0.0, 1.0 - under)


def aggregate(check_scores: list[tuple[float, float]]) -> float:
    """[(score, weight)] -> weighted mean; empty -> 0.0."""
    total_w = sum(w for _, w in check_scores)
    if total_w <= 0:
        return 0.0
    return sum(s * w for s, w in check_scores) / total_w
