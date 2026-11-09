"""Anti-cheating gating audit (SCORING.md §2.4 / §4.2).

Hard gating flags zero a layer-2 question's total_score (total_score_raw is
kept). plan_only_respected additionally applies to L1/L3. The two soft flags
(paths_in_session, files_nonempty) are reported but never gate.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from effects import violates_plan_only

# Program banners emitted by the QE binaries the product actually dispatches.
_BANNER_RE = re.compile(
    r"Program\s+(?:PWSCF|PHONON|PROJWFC|DOS|Q2R|MATDYN)\b", re.IGNORECASE
)
_JOB_DONE_RE = re.compile(r"JOB DONE\.")
# e.g. "PWSCF        :   1.23s CPU   1.45s WALL" / "PHONON      : ... WALL"
_WALL_RE = re.compile(
    r"^\s*(?:PWSCF|PHONON|PROJWFC|DOS|Q2R|MATDYN)\s*:"
    r".*?([0-9]+(?:\.[0-9]+)?)s\s*WALL",
    re.MULTILINE | re.IGNORECASE,
)

MIN_QE_WALL_SEC = 1.0
MIN_WALL_RATIO = 0.005
MIN_LOG_BYTES = 1024


def _walk_stdout_files(obj: Any) -> List[str]:
    """Collect every *.out / stdout path referenced by tool payloads."""
    found: List[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in ("stdout_file", "out_file", "output_file") and isinstance(value, str):
                found.append(value)
            else:
                found.extend(_walk_stdout_files(value))
    elif isinstance(obj, list):
        for item in obj:
            found.extend(_walk_stdout_files(item))
    return found


def _iter_qe_logs(record: Dict[str, Any]):
    """Yield (path, text, size) for every QE log reachable from the record."""
    paths: List[str] = []
    for entry in record.get("results") or []:
        paths.extend(_walk_stdout_files(entry.get("json")))
    # Also scan the session tree: workflow macros bind stdout files deep in
    # node payloads, but cheap glob coverage keeps the audit resilient.
    session_dir = record.get("session_dir")
    seen = set()
    for raw in paths:
        p = Path(raw)
        if not p.is_absolute() and session_dir:
            p = Path(session_dir) / p
        if p.is_file() and str(p) not in seen:
            seen.add(str(p))
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            yield p, text, p.stat().st_size
    if not session_dir or not Path(session_dir).is_dir():
        return
    for p in sorted(Path(session_dir).rglob("*.out")):
        if not p.is_file() or str(p) in seen:
            continue
        seen.add(str(p))
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        # Only QE program logs count; other *.out files are not audited.
        if _BANNER_RE.search(text):
            yield p, text, p.stat().st_size


def _qe_wall_seconds(record: Dict[str, Any]) -> float:
    total = 0.0
    for _, text, _ in _iter_qe_logs(record):
        total += sum(float(m.group(1)) for m in _WALL_RE.finditer(text))
    return total


def _paths_in_session(record: Dict[str, Any]) -> bool:
    session_dir = record.get("session_dir")
    if not session_dir:
        return True
    root = Path(session_dir).resolve()
    for raw in _walk_stdout_files([r.get("json") for r in record.get("results") or []]):
        p = Path(raw)
        if not p.is_absolute():
            p = root / p
        try:
            p.resolve().relative_to(root)
        except (OSError, ValueError):
            return False
    return True


def audit_run(record: Dict[str, Any]) -> Dict[str, Any]:
    """Return the flag map described in SCORING.md §2.4.

    Hard flags for L2: qe_invoked, job_done, qe_wallclock_ok, no_fake_backend.
    plan_only_respected is evaluated for every layer but only *gates* L1/L3.
    """
    logs = list(_iter_qe_logs(record))
    qe_invoked = any(_BANNER_RE.search(text) for _, text, _ in logs)
    job_done = bool(logs) and all(_JOB_DONE_RE.search(text) for _, text, _ in logs)
    files_nonempty = bool(logs) and all(size >= MIN_LOG_BYTES for _, _, size in logs)

    qe_wall = _qe_wall_seconds(record)
    total_wall = float((record.get("meta") or {}).get("duration_sec") or 0.0)
    qe_wallclock_ok = (
        qe_wall >= MIN_QE_WALL_SEC
        and (total_wall <= 0.0 or qe_wall / total_wall >= MIN_WALL_RATIO)
    )

    executor = str((record.get("meta") or {}).get("executor") or "local").strip().lower()
    no_fake_backend = executor != "fake"

    side_effect_calls = [
        c.get("tool") for c in record.get("commands") or []
        if violates_plan_only(c.get("tool"))
    ]
    plan_only_respected = len(side_effect_calls) == 0

    paths_in_session = _paths_in_session(record)

    # A run that produced nothing must not collect points from the reverse
    # rules (must_not_match / no_fabricated_number / no_side_effect_calls all
    # pass vacuously on an empty record). Gated on every layer.
    plan = record.get("plan") or {}
    non_empty_answer = bool(
        (plan.get("actions") or [])
        or (record.get("commands") or [])
        or str(record.get("reply") or "").strip()
        or (plan.get("advisory_steps") or [])
        or (plan.get("blocking_reasons") or [])
    ) and not record.get("fatal_error")

    return {
        "non_empty_answer": non_empty_answer,
        "qe_invoked": qe_invoked,
        "job_done": job_done,
        "qe_wallclock_ok": qe_wallclock_ok,
        "no_fake_backend": no_fake_backend,
        "plan_only_respected": plan_only_respected,
        "paths_in_session": paths_in_session,
        "files_nonempty": files_nonempty,
        # diagnostics (not flags)
        "_qe_wall_sec": round(qe_wall, 3),
        "_total_wall_sec": total_wall,
        "_side_effect_calls": side_effect_calls,
        "_qe_log_count": len(logs),
    }


# Hard-gating flag sets per layer, per SCORING.md §4.2.
ALL_LAYER_HARD_FLAGS = ("non_empty_answer",)
L2_HARD_FLAGS = ("qe_invoked", "job_done", "qe_wallclock_ok", "no_fake_backend")
PLAN_LAYER_HARD_FLAGS = ("plan_only_respected",)


def hard_gating_failed(layer: str, flags: Dict[str, Any]) -> List[str]:
    failed: List[str] = []
    hard = ALL_LAYER_HARD_FLAGS + (
        L2_HARD_FLAGS if layer == "EXEC" else PLAN_LAYER_HARD_FLAGS)
    for name in hard:
        if not flags.get(name):
            failed.append(name)
    return failed
