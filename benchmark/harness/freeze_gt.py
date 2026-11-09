"""Freeze L2 ground truth from two agreeing live rounds (SCORING.md §7).

    python3 harness/freeze_gt.py runs/<round-A> runs/<round-B> [--apply]

Default is a dry run. With --apply it rewrites the EXEC question JSONs:
  - both rounds agree within the check's own tolerance -> write `value`
  - they disagree                                       -> mark gt_unstable and
                                                           drop the numeric check
                                                           to a link assertion
Only questions whose gating passed in BOTH rounds are eligible: a number from a
run that did not really invoke QE must never become a baseline.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

HARNESS = Path(__file__).resolve().parent
ROOT = HARNESS.parent
import sys                                               # noqa: E402
sys.path.insert(0, str(HARNESS))

from extract import extract                              # noqa: E402
from trace import load_record                            # noqa: E402

GRADED = ("graded_relative", "graded_absolute")


def round_records(path: Path) -> Dict[str, Dict[str, Any]]:
    """question_id -> the first repeat's record (L2 runs once by design)."""
    out: Dict[str, Dict[str, Any]] = {}
    for rec_path in sorted(path.glob("*/rep*/record.json")):
        rec = load_record(rec_path)
        out.setdefault(rec["question_id"], rec)
    return out


def gating_ok(path: Path, qid: str) -> bool:
    for score_path in sorted(path.glob(f"{qid}/rep*/score.json")):
        data = json.loads(score_path.read_text(encoding="utf-8"))
        return not data.get("gating_failed")
    return False


def agree(a: float, b: float, rule: str, tol: float) -> bool:
    if rule == "graded_relative":
        ref = max(abs(a), abs(b))
        return ref > 0 and abs(a - b) / ref <= tol
    return abs(a - b) <= tol


def plan(round_a: Path, round_b: Path) -> List[Dict[str, Any]]:
    recs_a, recs_b = round_records(round_a), round_records(round_b)
    actions: List[Dict[str, Any]] = []
    for qpath in sorted((ROOT / "questions").glob("EXEC-*.json")):
        q = json.loads(qpath.read_text(encoding="utf-8"))
        qid = q["id"]
        graded = [c for c in q["checks"] if c["rule"] in GRADED]
        if not graded:
            continue
        if qid not in recs_a or qid not in recs_b:
            actions.append({"qid": qid, "status": "missing",
                            "note": "两轮中至少一轮没有该题记录"})
            continue
        if not (gating_ok(round_a, qid) and gating_ok(round_b, qid)):
            actions.append({"qid": qid, "status": "gating_failed",
                            "note": "有一轮 gating 未通过，不能作为基线"})
            continue
        for c in graded:
            field, tol = c["args"]["field"], float(c["args"]["tolerance"])
            va, vb = extract(recs_a[qid], field), extract(recs_b[qid], field)
            if va is None or vb is None:
                actions.append({"qid": qid, "check": c["id"], "field": field,
                                "status": "not_extracted", "a": va, "b": vb})
                continue
            if agree(float(va), float(vb), c["rule"], tol):
                actions.append({"qid": qid, "check": c["id"], "field": field,
                                "status": "freeze", "a": va, "b": vb,
                                "value": round((float(va) + float(vb)) / 2, 6),
                                "old_value": c["args"].get("value")})
            else:
                actions.append({"qid": qid, "check": c["id"], "field": field,
                                "status": "unstable", "a": va, "b": vb, "tol": tol})
    return actions


def apply(actions: List[Dict[str, Any]], meta: Dict[str, Any]) -> None:
    by_q: Dict[str, List[Dict[str, Any]]] = {}
    for a in actions:
        by_q.setdefault(a["qid"], []).append(a)
    for qid, items in by_q.items():
        qpath = ROOT / "questions" / f"{qid}.json"
        q = json.loads(qpath.read_text(encoding="utf-8"))
        changed = False
        for a in items:
            if a["status"] == "freeze":
                for c in q["checks"]:
                    if c["id"] == a["check"]:
                        c["args"]["value"] = a["value"]
                        changed = True
            elif a["status"] == "unstable":
                q["gt_unstable"] = True
                for c in q["checks"]:
                    if c["id"] == a["check"]:
                        c["kind"], c["rule"] = "numeric", "finite"
                        c["args"] = {"field": a["field"],
                                     "unit": a.get("unit", "")}
                        c["notes"] = (f"两轮数值不一致 ({a['a']} vs {a['b']})，"
                                      f"降级为链路断言")
                        changed = True
        if changed:
            q["gt_frozen_at"] = meta["frozen_at"]
            q["gt_qe_version"] = meta["qe_version"]
            q["gt_llm_model"] = meta["llm_model"]
            qpath.write_text(json.dumps(q, ensure_ascii=False, indent=1) + "\n",
                             encoding="utf-8")
            print(f"  已更新 {qpath.name}")


def main() -> int:
    ap = argparse.ArgumentParser(description="用两轮一致的实测值冻结 L2 基线")
    ap.add_argument("round_a")
    ap.add_argument("round_b")
    ap.add_argument("--apply", action="store_true", help="真正写回题目 JSON")
    args = ap.parse_args()
    a, b = Path(args.round_a), Path(args.round_b)
    actions = plan(a, b)
    counts: Dict[str, int] = {}
    for x in actions:
        counts[x["status"]] = counts.get(x["status"], 0) + 1
        line = f"{x['qid']:8s} {x.get('check','-'):4s} {x['status']:14s}"
        if "field" in x:
            line += f" {x['field']} a={x.get('a')} b={x.get('b')}"
        if x["status"] == "freeze":
            line += f" -> {x['value']} (原 {x.get('old_value')})"
        print(line)
    print(f"\n汇总: {counts}")
    if not args.apply:
        print("这是试运行。确认无误后加 --apply 写回。")
        return 0
    report = json.loads((b / "report.json").read_text(encoding="utf-8"))
    meta = {"frozen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "qe_version": (report.get("meta") or {}).get("qe_version", "unknown"),
            "llm_model": (report.get("meta") or {}).get("llm_model", "unknown")}
    apply(actions, meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
