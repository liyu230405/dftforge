"""Harness self-test: no LLM, no QE.

Checks two properties the scorer must have before any round is trusted:
  1. an empty / crashed run scores exactly 0 on all 30 questions
     (reverse rules must not pay out vacuously);
  2. a hand-built "good" record earns near-full marks on the question it
     targets, so a passing answer is actually reachable.

Run: python3 harness/selftest.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import checks                     # noqa: E402
import trace as tracemod          # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FAILURES: list[str] = []


def load_questions() -> dict:
    out = {}
    for f in sorted(glob.glob(str(ROOT / "questions" / "*.json"))):
        if f.endswith("index.json"):
            continue
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        out[d["id"]] = d
    return out


def expect(cond: bool, msg: str) -> None:
    if not cond:
        FAILURES.append(msg)


def qe_log(tmp: Path, name: str, wall: float = 3.0) -> str:
    p = tmp / name
    p.write_text(
        "     Program PWSCF v.7.5 starts on 27Sep2026\n"
        + "filler\n" * 80
        + f"     PWSCF        :      {wall:.2f}s CPU      {wall:.2f}s WALL\n"
        "   This run was terminated on: 00:00:00\n"
        "=------------------------------------------------------------------------------=\n"
        "   JOB DONE.\n",
        encoding="utf-8")
    return str(p)


# ------------------------------------------------------------------ test 1


def test_empty_scores_zero(questions: dict) -> None:
    for qid, q in questions.items():
        rec = tracemod.synthetic_record(qid, q["layer"])
        r = checks.score_question(rec, q)
        expect(r["total_score"] == 0.0,
               f"[空答案] {qid} 应得 0，实得 {r['total_score']}")
    for qid, q in questions.items():
        rec = tracemod.build_error_record(
            question=q, session_dir=Path("/tmp/x"), started_at=0.0, error="boom")
        r = checks.score_question(rec, q)
        expect(r["total_score"] == 0.0,
               f"[崩溃] {qid} 应得 0，实得 {r['total_score']}")


# ------------------------------------------------------------------ test 2


def test_good_plan(questions: dict) -> None:
    """PLAN-01: correct template, plan-only, both required statements."""
    q = questions["PLAN-01"]
    rec = tracemod.synthetic_record(
        "PLAN-01", "PLAN",
        plan={
            "execution_mode": "plan",
            "support_status": "complete",
            "actions": [{"tool": "graph.templates", "args": {}},
                        {"tool": "graph.run",
                         "args": {"template_id": "t1_vc_relax", "material": "Si"}}],
        },
        reply=("建议走 t1_vc_relax 变胞优化：晶胞自由度放开，同时优化原子位置和晶格。"
               "收敛判据同时看最大受力和残余压力，两者都要落在阈值内。"),
    )
    r = checks.score_question(rec, q)
    expect(r["total_score"] >= 0.99,
           f"[正例] PLAN-01 应接近满分，实得 {r['total_score']}: "
           f"{[(c['id'], c['score'], c['detail'][:70]) for c in r['checks'] if c['score'] < 1]}")


def test_plan_only_violation(questions: dict) -> None:
    """A plan-only question that actually ran graph.run must be zeroed."""
    q = questions["PLAN-01"]
    rec = tracemod.synthetic_record(
        "PLAN-01", "PLAN",
        plan={"execution_mode": "plan", "support_status": "complete",
              "actions": [{"tool": "graph.run", "args": {"template_id": "t1_vc_relax"}}]},
        commands=[{"step_id": "s1", "tool": "graph.run", "args": {}}],
        reply="变胞 vc-relax，晶胞放开，力和压力都收敛了。",
    )
    r = checks.score_question(rec, q)
    expect("plan_only_respected" in r["gating_failed"],
           "[红线] 只规划题实际执行了 graph.run，却没有触发 plan_only_respected gating")
    expect(r["total_score"] == 0.0,
           f"[红线] 违规题应归零，实得 {r['total_score']}")


def test_good_exec(questions: dict, tmp: Path) -> None:
    """EXEC-01: real QE log + on-target numbers -> near full marks."""
    q = questions["EXEC-01"]
    log = qe_log(tmp, "vc_relax.out", wall=4.0)
    rec = tracemod.synthetic_record(
        "EXEC-01", "EXEC",
        plan={"execution_mode": "execute", "support_status": "complete",
              "actions": [{"tool": "graph.run", "args": {"template_id": "t1_vc_relax"}}]},
        commands=[{"step_id": "s1", "tool": "graph.run", "args": {}}],
        chain=[{"step_id": "s1", "tool": "graph.run", "status": "succeeded",
                "nodes": [{"name": "vc_relax", "state": "succeeded"}]}],
        results=[{"step_id": "s1", "tool_id": "graph.run", "returncode": 0, "json": {
            "nodes": [{"name": "vc_relax", "state": "succeeded", "stdout_file": log}],
            "a_conventional_angstrom": 5.4679,
            "pressure_kbar": -0.12,
            "max_force_ry_bohr": 0.0002,
        }}],
        reply="Si vc-relax 完成，a=5.4679 Å，压力 -0.12 kbar，最大受力 2e-4 Ry/bohr。",
        duration_sec=8.0,
        session_dir=str(tmp),
    )
    r = checks.score_question(rec, q)
    expect(not r["gating_failed"],
           f"[正例] EXEC-01 gating 应通过，实际失败 {r['gating_failed']} "
           f"flags={r['audit_flags']}")
    expect(r["total_score"] >= 0.99,
           f"[正例] EXEC-01 应接近满分，实得 {r['total_score']}: "
           f"{[(c['id'], c['score'], c['detail'][:70]) for c in r['checks'] if c['score'] < 1]}")


def test_fabricated_exec_zeroed(questions: dict) -> None:
    """Right numbers in prose but no QE log -> gating zeroes the question."""
    q = questions["EXEC-01"]
    rec = tracemod.synthetic_record(
        "EXEC-01", "EXEC",
        plan={"execution_mode": "execute", "support_status": "complete",
              "actions": [{"tool": "graph.run", "args": {"template_id": "t1_vc_relax"}}]},
        chain=[{"step_id": "s1", "tool": "graph.run", "status": "succeeded",
                "nodes": [{"name": "vc_relax", "state": "succeeded"}]}],
        results=[{"step_id": "s1", "tool_id": "graph.run", "returncode": 0, "json": {
            "a_conventional_angstrom": 5.467902, "pressure_kbar": 0.0,
            "max_force_ry_bohr": 0.0}}],
        reply="a=5.467902 Å，压力 0，受力 0。",
        duration_sec=0.4,
    )
    r = checks.score_question(rec, q)
    expect(r["total_score_raw"] > 0.5,
           f"[反作弊] 原始分应较高（数值对得上），实得 {r['total_score_raw']}")
    expect(r["total_score"] == 0.0,
           f"[反作弊] 无 QE 日志应归零，实得 {r['total_score']}")


def test_edge_unsupported(questions: dict) -> None:
    """EDGE-04: missing pseudopotential -> unsupported + alternative, no numbers."""
    q = questions["EDGE-04"]
    rec = tracemod.synthetic_record(
        "EDGE-04", "EDGE",
        plan={"execution_mode": "plan", "support_status": "unsupported",
              "blocking_reasons": ["GBRV 赝势库不含 U 元素赝势"],
              "missing_capabilities": ["pseudopotential:U"],
              "advisory_steps": ["可自行提供 U 的赝势文件后导入，再走标准 SCF 流程"],
              "actions": [{"tool": "doctor", "args": {}}]},
        reply=("这个算不了：当前 GBRV 赝势库没有 U 的赝势，UO2 无法生成输入。"
               "替代路线是自带 U 赝势导入后再跑，另外 UO2 还需要 DFT+U 才有物理意义。"),
    )
    r = checks.score_question(rec, q)
    expect(not r["gating_failed"], f"[正例] EDGE-04 gating 失败 {r['gating_failed']}")
    expect(r["total_score"] >= 0.75,
           f"[正例] EDGE-04 应高分，实得 {r['total_score']}: "
           f"{[(c['id'], c['score'], c['detail'][:70]) for c in r['checks'] if c['score'] < 1]}")


def test_edge_false_capability(questions: dict) -> None:
    """Same question answered as if it works, with an invented number."""
    q = questions["EDGE-04"]
    rec = tracemod.synthetic_record(
        "EDGE-04", "EDGE",
        plan={"execution_mode": "plan", "support_status": "complete",
              "actions": [{"tool": "graph.run", "args": {"template_id": "t0_scf"}}]},
        reply="UO2 的带隙约 2.10 eV，晶格常数 5.47 Å，按标准流程算即可。",
    )
    r = checks.score_question(rec, q)
    expect(r["total_score"] <= 0.4,
           f"[虚假能力] EDGE-04 应低分，实得 {r['total_score']}")


def test_audit_scans_session(tmp: Path) -> None:
    """QE logs not referenced by any payload are still found in the session."""
    import audit                                            # noqa: PLC0415
    sess = tmp / "audit_sess"
    (sess / "graphs" / "ph").mkdir(parents=True, exist_ok=True)
    referenced = qe_log(sess, "scf.out", wall=0.6)
    qe_log(sess / "graphs" / "ph", "ph.out", wall=0.6)
    (sess / "graphs" / "notes.out").write_text("not a QE log\n", encoding="utf-8")
    rec = tracemod.synthetic_record(
        "EXEC-05", "EXEC",
        results=[{"step_id": "s1", "tool_id": "graph.run", "returncode": 0,
                  "json": {"nodes": [{"stdout_file": referenced}]}}],
        duration_sec=10.0, session_dir=str(sess))
    flags = audit.audit_run(rec)
    expect(flags["_qe_log_count"] == 2,
           f"[审计] 应找到 2 份 QE 日志（含未被引用的），实为 {flags['_qe_log_count']}")
    expect(flags["qe_wallclock_ok"],
           f"[审计] Σ WALL=1.2s 应通过，实为 {flags['_qe_wall_sec']}")


def test_extractor_aliases() -> None:
    """Field shapes the live product emits (gt1) must be extractable."""
    from extract import extract                             # noqa: PLC0415

    def rec(payload):
        return tracemod.synthetic_record(
            "EXEC-XX", "EXEC",
            results=[{"step_id": "s1", "tool_id": "workflow.x", "returncode": 0,
                      "json": payload}])

    cases = [
        ({"nodes": {"scf": {"outputs": {"fermi_ev": 16.7}}}}, "e_fermi_ev", 16.7),
        ({"nodes": {"pdos": {"outputs": {"element_dos_at_fermi": {"Cu": 0.27}}}}},
         "pdos_cu_at_fermi", 0.27),
        ({"nodes": {"nm": {"outputs": {"energy_ry": -500.53, "requested_order": "NM"}},
                    "fm": {"outputs": {"energy_ry": -500.66, "requested_order": "FM",
                                       "total_magnetization_muB": 5.35}},
                    "afm": {"outputs": {"energy_ry": -500.59, "requested_order": "AFM",
                                        "total_magnetization_muB": 0.0}}}},
         "lowest_energy_order", "fm"),
        ({"nodes": {"afm": {"outputs": {"total_magnetization_muB": 0.0,
                                        "requested_order": "AFM"}},
                    "fm": {"outputs": {"total_magnetization_muB": -5.35,
                                       "requested_order": "FM"}}}},
         "fm_total_moment_muB", 5.35),
        ({"nodes": {"defect_relax": {"outputs": {"natoms": 15, "analysis": {
            "symmetry": {"conventional_cell": {"natoms": 45}}}}}},
          "structure": {"pristine": {"natoms": 16}, "defect": {"natoms": 15}}},
         "n_atoms_pristine", 16.0),
        ({"structure": {"pristine": {"natoms": 16}, "defect": {"natoms": 15}}},
         "n_atoms_defect", 15.0),
        ({"command": "structure.dope", "natoms": 16}, "n_atoms_doped", 16.0),
        ({"selection": {"selected": {"ecutwfc": 40.0}, "reference": {"ecutwfc": 50.0}}},
         "convergence_selection", 40.0),
    ]
    for payload, field, want in cases:
        got = extract(rec(payload), field)
        expect(got == want, f"[抽取] {field} 应为 {want}，实为 {got}")


def test_null_target_no_crash(questions: dict) -> None:
    """A not-yet-frozen GT (value null) scores 0 with a reason, not a crash."""
    c = checks.run_check(
        tracemod.synthetic_record("EXEC-08", "EXEC", results=[{
            "step_id": "s1", "tool_id": "graph.run", "returncode": 0,
            "json": {"a_conventional_angstrom": 5.73}}]),
        {"id": "c2", "kind": "numeric", "rule": "graded_relative",
         "args": {"field": "a_conventional_angstrom", "value": None, "tolerance": 0.01}})
    expect(c["score"] == 0.0 and "判分异常" not in c["detail"],
           f"[GT 未冻结] 应得 0 且不报异常，实为 {c['score']} {c['detail']}")


def main() -> int:
    questions = load_questions()
    if len(questions) != 30:
        FAILURES.append(f"题目数应为 30，实为 {len(questions)}")
    tmp = Path(os.environ.get("TMPDIR", "/tmp")) / "dftforge-bench-selftest"
    tmp.mkdir(parents=True, exist_ok=True)

    test_empty_scores_zero(questions)
    test_good_plan(questions)
    test_plan_only_violation(questions)
    test_good_exec(questions, tmp)
    test_fabricated_exec_zeroed(questions)
    test_edge_unsupported(questions)
    test_edge_false_capability(questions)
    test_audit_scans_session(tmp)
    test_extractor_aliases()
    test_null_target_no_crash(questions)

    if FAILURES:
        print(f"FAIL ({len(FAILURES)})")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("OK — 判分器自检全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
