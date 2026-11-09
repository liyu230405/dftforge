"""Cross-round regression diff (SCORING.md §5).

    python3 harness/compare.py runs/<old> runs/<new>

Ordering is deliberate: red lines first, then per-check regressions, then
layer averages last. An average that rose while a boundary question flipped
from a correct refusal to a plausible-looking wrong flow is still a regression.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

EPS = 1e-6


def load(path: Path) -> Dict[str, Any]:
    p = Path(path)
    if p.is_dir():
        p = p / "report.json"
    return json.loads(p.read_text(encoding="utf-8"))


def by_id(report: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {q["question_id"]: q for q in report.get("per_question") or []}


def check_map(q: Dict[str, Any]) -> Dict[str, float]:
    return {c["id"]: c["score"] for c in q.get("checks") or []}


def diff(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    o, n = by_id(old), by_id(new)
    shared = sorted(set(o) & set(n))

    moved: List[Dict[str, Any]] = []
    new_failures: List[Dict[str, Any]] = []
    for qid in shared:
        delta = n[qid]["total_score"] - o[qid]["total_score"]
        if abs(delta) > EPS:
            moved.append({"question_id": qid, "title": n[qid].get("title", ""),
                          "old": o[qid]["total_score"], "new": n[qid]["total_score"],
                          "delta": round(delta, 4)})
        oc, nc = check_map(o[qid]), check_map(n[qid])
        for cid, score in nc.items():
            before = oc.get(cid)
            if before is None or score >= before - EPS:
                continue
            detail = next((c["detail"] for c in n[qid]["checks"] if c["id"] == cid), "")
            new_failures.append({"question_id": qid, "check": cid,
                                 "rule": next((c["rule"] for c in n[qid]["checks"]
                                               if c["id"] == cid), ""),
                                 "old": before, "new": score, "detail": detail})
        og, ng = set(o[qid].get("gating_failed") or []), set(n[qid].get("gating_failed") or [])
        for flag in sorted(ng - og):
            new_failures.append({"question_id": qid, "check": f"gating:{flag}",
                                 "rule": "audit", "old": 1.0, "new": 0.0,
                                 "detail": f"新增 gating 失败: {flag}"})

    moved.sort(key=lambda m: m["delta"])
    red = {}
    for key in ("plan_only_violations", "audit_failures", "false_capability_claims"):
        red[key] = {"old": (old.get("red_lines") or {}).get(key),
                    "new": (new.get("red_lines") or {}).get(key)}
    layers = {}
    for key in ("L1", "L2", "L3"):
        layers[key] = {"old": (old.get("layer_scores") or {}).get(key),
                       "new": (new.get("layer_scores") or {}).get(key)}

    comparable = (
        (old.get("meta") or {}).get("llm_model") == (new.get("meta") or {}).get("llm_model")
        and (old.get("meta") or {}).get("qe_version") == (new.get("meta") or {}).get("qe_version")
        and (old.get("meta") or {}).get("executor") == (new.get("meta") or {}).get("executor")
    )
    return {
        "comparable": comparable,
        "red_lines": red,
        "layer_scores": layers,
        "regressions": [m for m in moved if m["delta"] < 0],
        "improvements": [m for m in moved if m["delta"] > 0],
        "new_check_failures": new_failures,
        "only_in_old": sorted(set(o) - set(n)),
        "only_in_new": sorted(set(n) - set(o)),
    }


def render(d: Dict[str, Any], old_meta: Dict[str, Any], new_meta: Dict[str, Any]) -> str:
    out: List[str] = []
    out.append("==== 跨轮对比 ====")
    out.append(f"旧: {old_meta.get('round_id')} sha={old_meta.get('git_sha')} "
               f"model={old_meta.get('llm_model')} qe={old_meta.get('qe_version')}")
    out.append(f"新: {new_meta.get('round_id')} sha={new_meta.get('git_sha')} "
               f"model={new_meta.get('llm_model')} qe={new_meta.get('qe_version')}")
    if not d["comparable"]:
        out.append("!! 模型/QE 版本/执行器发生变化，两轮不可直接比较，需重跑基线")

    out.append("\n-- 红线指标（任一不为 0 即判本轮退化）--")
    breached = False
    for key, v in d["red_lines"].items():
        arrow = "" if v["old"] == v["new"] else f"  ({v['old']} → {v['new']})"
        flag = "" if not v["new"] else "  !!"
        breached = breached or bool(v["new"])
        out.append(f"  {key}: {v['new']}{arrow}{flag}")

    out.append("\n-- 分数下降的题目 --")
    if d["regressions"]:
        for m in d["regressions"]:
            out.append(f"  {m['question_id']} {m['delta']:+.3f} "
                       f"({m['old']:.3f} → {m['new']:.3f})  {m['title']}")
    else:
        out.append("  无")

    out.append("\n-- 新增失败的 check --")
    if d["new_check_failures"]:
        for f in d["new_check_failures"]:
            out.append(f"  {f['question_id']}/{f['check']} [{f['rule']}] "
                       f"{f['old']} → {f['new']}")
            out.append(f"      {f['detail'][:140]}")
    else:
        out.append("  无")

    out.append("\n-- 分数上升的题目 --")
    if d["improvements"]:
        for m in d["improvements"]:
            out.append(f"  {m['question_id']} {m['delta']:+.3f}  {m['title']}")
    else:
        out.append("  无")

    out.append("\n-- 层级平均分（仅供参考，排在红线与逐题差异之后）--")
    for key, v in d["layer_scores"].items():
        if v["old"] is None or v["new"] is None:
            out.append(f"  {key}: {v['old']} → {v['new']}")
        else:
            out.append(f"  {key}: {v['old']:.3f} → {v['new']:.3f} "
                       f"({v['new'] - v['old']:+.3f})")

    if d["only_in_old"] or d["only_in_new"]:
        out.append(f"\n题集变化: 旧独有={d['only_in_old']} 新独有={d['only_in_new']}")

    verdict = "退化" if (breached or d["regressions"]) else \
              ("改善" if d["improvements"] else "持平")
    out.append(f"\n结论: {verdict}")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="对比两轮 benchmark 结果")
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--json", action="store_true", help="输出 JSON 而非文本")
    args = ap.parse_args()
    old, new = load(Path(args.old)), load(Path(args.new))
    d = diff(old, new)
    if args.json:
        print(json.dumps(d, ensure_ascii=False, indent=2))
    else:
        print(render(d, old.get("meta") or {}, new.get("meta") or {}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
