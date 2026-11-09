"""Drive one round of the benchmark against a live DFT-Forge AgentLoop.

    python3 harness/run.py --layer PLAN --repeat 3
    python3 harness/run.py --only EXEC-01,EXEC-02
    python3 harness/run.py --score-only runs/2026-09-27T…

Each question gets its own session dir under runs/<round>/<QID>/rep<N>/session.
Raw run records are written before scoring, so a round can always be re-scored
offline after a scorer fix without re-spending QE time.

The API key is read from DFT_FORGE_LLM_API_KEY if already exported, otherwise
prompted for with getpass and kept in memory only. It is never written to disk:
every record passes through a redaction pass first.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import getpass
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

HARNESS = Path(__file__).resolve().parent
ROOT = HARNESS.parent
sys.path.insert(0, str(HARNESS))

import checks                     # noqa: E402
import trace as tracemod          # noqa: E402
from effects import verify_against_registry   # noqa: E402

REPO = Path(os.environ.get("DFT_FORGE_REPO", str(ROOT.parent)))
BENCH_VERSION = "1.0"

DEFAULT_ENV = {
    "DFT_FORGE_LLM_PROVIDER": "openai",
    "DFT_FORGE_LLM_MODEL": "deepseek-flash",
    "DFT_FORGE_LLM_BASE_URL": "https://api.deepseek.com/v1",
    "DFT_FORGE_EXECUTOR": "local",
    "DFT_FORGE_QE_BIN": "/opt/homebrew/bin",
    "DFT_FORGE_WALLTIME": "600",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


# ---------------------------------------------------------------- utilities


def redact(value: Any) -> Any:
    raw = json.dumps(value, ensure_ascii=False, default=str)
    key = os.environ.get("DFT_FORGE_LLM_API_KEY", "")
    if key:
        raw = raw.replace(key, "[REDACTED]")
    return json.loads(raw)


def load_questions(only: List[str] | None, layer: str | None) -> List[Dict[str, Any]]:
    qs = []
    for path in sorted((ROOT / "questions").glob("*.json")):
        if path.name == "index.json":
            continue
        q = json.loads(path.read_text(encoding="utf-8"))
        if layer and q["layer"] != layer:
            continue
        if only and q["id"] not in only:
            continue
        qs.append(q)
    order = {"PLAN": 0, "EXEC": 1, "EDGE": 2}
    qs.sort(key=lambda q: (order.get(q["layer"], 9), q["id"]))
    return qs


def git_sha() -> str:
    try:
        out = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def qe_version() -> str:
    pw = Path(os.environ.get("DFT_FORGE_QE_BIN", "/opt/homebrew/bin")) / "pw.x"
    if not pw.exists():
        return "absent"
    # pw.x has no --version: invoking it starts a run and litters a CRASH file.
    # The version string is compiled into the binary, so read it out instead.
    try:
        blob = pw.read_bytes()
        m = re.search(rb"Program PWSCF v\.([0-9][0-9.A-Za-z]*)", blob)
        if m:
            return m.group(1).decode("ascii", "replace").rstrip(".")
    except OSError:
        pass
    # Self-built QE carries the version only in its install path (qe-7.5/bin).
    m = re.search(r"qe[-_]?(\d+\.\d+(?:\.\d+)?)", str(pw.resolve()), re.IGNORECASE)
    return m.group(1) if m else "unknown"


# ---------------------------------------------------------------- one attempt


async def run_one(question: Dict[str, Any], session: Path) -> Dict[str, Any]:
    """Run one question once; return the run record (never raises)."""
    from dft_forge.agent_loop import AgentLoop
    from dft_forge.agent_loop.method_contract import MethodContractError
    from dft_forge.llm import OpenAICompatProvider

    session.mkdir(parents=True, exist_ok=True)
    agent = AgentLoop()
    provider = OpenAICompatProvider(timeout=90)
    api_trace: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []
    chat = provider.chat

    def traced(system, user):
        t = time.monotonic()
        try:
            value = chat(system, user)
            api_trace.append({"seconds": round(time.monotonic() - t, 2)})
            return value
        except Exception as exc:                                  # noqa: BLE001
            api_trace.append({"error": str(exc)})
            raise

    provider.chat = traced
    agent.llm_planner._provider, agent.llm_planner._checked = provider, True

    run_config = dict(question.get("run_config") or {})
    run_config.setdefault("mode", "research")
    run_config.setdefault("approved", True)
    ctx = {"run_config": run_config, "cancel_event": threading.Event()}

    started = time.time()
    confirmed = False
    try:
        try:
            result = await agent.run(question["prompt"], session, ctx=ctx,
                                     on_event=events.append)
        except MethodContractError as exc:
            if exc.code != "method_confirmation_required":
                raise
            # The contract must be presented *before* anything runs; if QE
            # output already exists here, the product broke its own gate.
            if any(session.rglob("*.out")):
                return tracemod.build_error_record(
                    question=question, session_dir=session, started_at=started,
                    error="calculation occurred before method confirmation",
                    meta_extra={"executor": os.environ.get("DFT_FORGE_EXECUTOR", "local")})
            confirmed = True
            ctx["requested_method_token"] = exc.method_token
            ctx["_confirmed_method_plan"] = copy.deepcopy(exc.pending_plan)
            result = await agent.run(question["prompt"], session, ctx=ctx,
                                     on_event=events.append)
    except Exception as exc:                                      # noqa: BLE001
        return tracemod.build_error_record(
            question=question, session_dir=session, started_at=started,
            error=f"{type(exc).__name__}: {exc}",
            meta_extra={"executor": os.environ.get("DFT_FORGE_EXECUTOR", "local")})

    (result.get("ctx") or {}).pop("cancel_event", None)
    record = tracemod.build_record(
        question=question, result=redact(result), events=redact(events),
        session_dir=session, started_at=started,
        meta_extra={
            "executor": os.environ.get("DFT_FORGE_EXECUTOR", "local"),
            "llm_model": getattr(provider, "model", None),
            "method_confirmed": confirmed,
            "api_calls": len(api_trace),
        })
    return record


# ---------------------------------------------------------------- round


def majority(scores: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate repeats: median score; flaky when the repeats disagree."""
    vals = sorted(s["total_score"] for s in scores)
    mid = vals[len(vals) // 2] if len(vals) % 2 else (vals[len(vals) // 2 - 1]
                                                     + vals[len(vals) // 2]) / 2
    best = min(scores, key=lambda s: abs(s["total_score"] - mid))
    out = dict(best)
    out["repeats"] = [round(v, 4) for v in vals]
    out["flaky"] = (max(vals) - min(vals)) > 1e-6
    out["total_score"] = round(mid, 4)
    return out


async def run_round(args) -> int:
    questions = load_questions(args.only, args.layer)
    if not questions:
        print("没有匹配的题目")
        return 1
    round_id = args.round or time.strftime("%Y-%m-%dT%H%M%SZ", time.gmtime())
    dest = ROOT / "runs" / f"{round_id}_{git_sha()}"
    dest.mkdir(parents=True, exist_ok=True)

    drift = verify_against_registry()
    per_question: List[Dict[str, Any]] = []
    for q in questions:
        repeat = args.repeat if q["layer"] != "EXEC" else max(1, args.repeat_exec)
        reps = []
        for i in range(repeat):
            session = dest / q["id"] / f"rep{i + 1}" / "session"
            print(f"START {q['id']} rep{i + 1}/{repeat}", flush=True)
            record = await run_one(q, session)
            tracemod.save_record(record, session.parent / "record.json")
            scored = checks.score_question(record, q)
            (session.parent / "score.json").write_text(
                json.dumps(scored, ensure_ascii=False, indent=2), encoding="utf-8")
            reps.append(scored)
            print(f"  -> {scored['total_score']:.3f} "
                  f"(raw {scored['total_score_raw']:.3f}) "
                  f"gate={scored['gating_failed'] or 'ok'} "
                  f"{record['meta']['duration_sec']}s", flush=True)
        per_question.append(majority(reps) if len(reps) > 1 else reps[0])

    report = build_report(per_question, round_id, drift)
    (dest / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print_report(report, dest)
    return 0


def build_report(per_question, round_id, drift) -> Dict[str, Any]:
    layers = {"L1": "PLAN", "L2": "EXEC", "L3": "EDGE"}
    layer_scores = {}
    for key, layer in layers.items():
        rows = [p for p in per_question if p["layer"] == layer]
        layer_scores[key] = round(sum(r["total_score"] for r in rows) / len(rows), 4) \
            if rows else None

    plan_only_violations = sum(
        1 for p in per_question if p["layer"] in ("PLAN", "EDGE")
        and not p["audit_flags"].get("plan_only_respected"))
    audit_failures = sum(
        1 for p in per_question if p["layer"] == "EXEC" and p["gating_failed"])
    # A false capability claim: an EDGE question whose support_status check
    # failed while the answer still described a workable flow.
    false_capability = 0
    for p in per_question:
        if p["layer"] != "EDGE":
            continue
        support = next((c for c in p["checks"] if c["rule"] == "support_status_in"), None)
        fabricated = next((c for c in p["checks"]
                           if c["rule"] == "no_fabricated_number"), None)
        if (support and support["score"] == 0.0) or (fabricated and fabricated["score"] == 0.0):
            false_capability += 1

    return {
        "meta": {
            "bench_version": BENCH_VERSION,
            "round_id": round_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_sha": git_sha(),
            "llm_model": os.environ.get("DFT_FORGE_LLM_MODEL", "unknown"),
            "executor": os.environ.get("DFT_FORGE_EXECUTOR", "local"),
            "qe_version": qe_version(),
            "side_effect_table_drift": drift,
        },
        "layer_scores": layer_scores,
        "red_lines": {
            "plan_only_violations": plan_only_violations,
            "audit_failures": audit_failures,
            "false_capability_claims": false_capability,
        },
        "per_question": per_question,
    }


def print_report(report: Dict[str, Any], dest: Path) -> None:
    ls, rl = report["layer_scores"], report["red_lines"]
    print("\n==== 本轮结果 ====")
    for key, label in (("L1", "规划"), ("L2", "执行"), ("L3", "边界")):
        v = ls.get(key)
        print(f"  {key} {label}: {'—' if v is None else f'{v:.3f}'}")
    print("  红线指标（必须为 0）:")
    for name, value in rl.items():
        mark = "OK" if value == 0 else "!! 退化"
        print(f"    {name}: {value}  {mark}")
    flaky = [p["question_id"] for p in report["per_question"] if p.get("flaky")]
    if flaky:
        print(f"  不稳定题目: {', '.join(flaky)}")
    manual = [p["question_id"] for p in report["per_question"]
              if p.get("needs_manual_review")]
    if manual:
        print(f"  建议人工复核: {', '.join(manual)}")
    if report["meta"]["side_effect_table_drift"]:
        print("  !! 工具副作用表与产品不一致:")
        for d in report["meta"]["side_effect_table_drift"]:
            print(f"     {d}")
    print(f"\n报告: {dest / 'report.json'}")


def score_only(path: Path) -> int:
    """Re-score an existing round from its saved records (no LLM, no QE)."""
    questions = {q["id"]: q for q in load_questions(None, None)}
    per_question = []
    for record_path in sorted(path.glob("*/rep*/record.json")):
        record = tracemod.load_record(record_path)
        q = questions.get(record["question_id"])
        if not q:
            continue
        scored = checks.score_question(record, q)
        (record_path.parent / "score.json").write_text(
            json.dumps(scored, ensure_ascii=False, indent=2), encoding="utf-8")
        per_question.append(scored)
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for s in per_question:
        grouped.setdefault(s["question_id"], []).append(s)
    merged = [majority(v) if len(v) > 1 else v[0] for v in grouped.values()]
    merged.sort(key=lambda s: s["question_id"])
    report = build_report(merged, path.name, [])
    (path / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print_report(report, path)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="DFT-Forge agent benchmark runner")
    ap.add_argument("--layer", choices=["PLAN", "EXEC", "EDGE"])
    ap.add_argument("--only", help="逗号分隔的题号，如 EXEC-01,EXEC-02")
    ap.add_argument("--repeat", type=int, default=3, help="L1/L3 重复次数（默认 3）")
    ap.add_argument("--repeat-exec", type=int, default=1, help="L2 重复次数（默认 1）")
    ap.add_argument("--round", help="轮次名，默认用 UTC 时间戳")
    ap.add_argument("--score-only", help="只对已有轮次目录重新判分")
    args = ap.parse_args()
    args.only = [x.strip() for x in args.only.split(",")] if args.only else None

    if args.score_only:
        return score_only(Path(args.score_only))

    if not REPO.exists():
        print(f"找不到 dft-forge 仓库: {REPO}")
        return 1
    sys.path.insert(0, str(REPO))
    for key, value in DEFAULT_ENV.items():
        os.environ.setdefault(key, value)
    if not os.environ.get("DFT_FORGE_LLM_API_KEY"):
        os.environ["DFT_FORGE_LLM_API_KEY"] = getpass.getpass("API key (仅存内存): ")
    return asyncio.run(run_round(args))


if __name__ == "__main__":
    raise SystemExit(main())
