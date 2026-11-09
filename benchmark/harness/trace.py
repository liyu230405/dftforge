"""Standardized run record: one agent attempt at one benchmark question.

The scorer is a *pure* function over this record, so runs scored months apart
(or replayed from disk) behave identically.  ``build_record`` converts a live
AgentLoop.run() result into the record; synthetic records can be built by hand
for harness self-tests.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA = "dftforge-bench/run-record-v1"

_PLAN_KEYS = (
    "execution_mode", "support_status", "over_budget", "blocking_reasons",
    "missing_capabilities", "advisory_steps", "actions", "reply", "goal",
)


def _plan_from_agent(result: Dict[str, Any], events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Prefer ctx.last_plan (the final PlanIR); fall back to plan events."""
    last_plan = ((result.get("ctx") or {}).get("last_plan")) or {}
    plan = {key: last_plan.get(key) for key in _PLAN_KEYS if key in last_plan}
    plan_events = [e for e in events if e.get("type") == "plan"]
    if plan_events:
        ev = plan_events[-1]
        for key in ("execution_mode", "support_status", "blocking_reasons",
                    "missing_capabilities"):
            plan.setdefault(key, ev.get(key))
    plan.setdefault("execution_mode", "execute")
    plan.setdefault("support_status", "complete")
    plan.setdefault("over_budget", False)
    plan.setdefault("blocking_reasons", [])
    plan.setdefault("missing_capabilities", [])
    plan.setdefault("advisory_steps", [])
    # Plan-side actions with their arguments. PlanIR.actions carries
    # tool/args; when absent (rule planner), reconstruct from the chain of
    # planned/executed steps.
    actions = plan.get("actions")
    if not isinstance(actions, list) or not actions:
        actions = []
        for cmd in result.get("commands") or []:
            actions.append({"tool": cmd.get("command"), "args": cmd.get("args") or {}})
        if not actions:
            for node in result.get("chain") or []:
                actions.append({"tool": node.get("tool"), "args": {}})
    plan["actions"] = actions
    return plan


def build_record(
    *,
    question: Dict[str, Any],
    result: Dict[str, Any],
    events: List[Dict[str, Any]],
    session_dir: Path,
    started_at: float,
    ended_at: Optional[float] = None,
    meta_extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    ended_at = ended_at if ended_at is not None else time.time()
    commands = []
    for cmd in result.get("commands") or []:
        commands.append({
            "step_id": cmd.get("step_id"),
            "tool": cmd.get("command"),
            "args": cmd.get("args") or {},
            "depends_on": cmd.get("depends_on") or [],
        })
    chain = []
    for node in result.get("chain") or []:
        chain.append({
            "step_id": node.get("step_id"),
            "tool": node.get("tool"),
            "status": node.get("status"),
            "summary": node.get("summary", ""),
            "nodes": node.get("nodes") or [],
        })
    # Graph/workflow tools report per-node state as results[].json.nodes, a
    # dict {name: {state, outputs, ...}}. Normalize it into chain[] so the
    # scorer has one place to look for node success.
    for entry in result.get("results") or []:
        nodes = ((entry.get("json") or {}).get("nodes")) or {}
        items = nodes.items() if isinstance(nodes, dict) else \
            ((n.get("name"), n) for n in nodes if isinstance(n, dict))
        normalized = [{"name": name, "state": (info or {}).get("state"),
                       "error": (info or {}).get("error")}
                      for name, info in items]
        if normalized:
            chain.append({
                "step_id": entry.get("step_id"),
                "tool": entry.get("tool_id"),
                "status": "succeeded" if all(
                    n["state"] == "succeeded" for n in normalized) else "failed",
                "summary": "",
                "nodes": normalized,
            })

    results = []
    for r in result.get("results") or []:
        results.append({
            "step_id": r.get("step_id"),
            "tool_id": r.get("tool_id"),
            "returncode": r.get("returncode"),
            "error": r.get("error"),
            "json": r.get("json"),
        })
    meta = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started_at)),
        "duration_sec": round(max(0.0, ended_at - started_at), 3),
        "planner": result.get("planner"),
    }
    if meta_extra:
        meta.update(meta_extra)
    return {
        "schema": SCHEMA,
        "question_id": question["id"],
        "layer": question["layer"],
        "meta": meta,
        "run_config": question.get("run_config") or {},
        "plan": _plan_from_agent(result, events),
        "commands": commands,
        "chain": chain,
        "results": results,
        "reply": result.get("reply", ""),
        "session_dir": str(session_dir),
        "fatal_error": None,
    }


def build_error_record(
    *,
    question: Dict[str, Any],
    session_dir: Path,
    started_at: float,
    error: str,
    meta_extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Record for a run that never produced an agent result (crash/timeout)."""
    ended = time.time()
    meta = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started_at)),
        "duration_sec": round(max(0.0, ended - started_at), 3),
        "planner": None,
    }
    if meta_extra:
        meta.update(meta_extra)
    return {
        "schema": SCHEMA,
        "question_id": question["id"],
        "layer": question["layer"],
        "meta": meta,
        "run_config": question.get("run_config") or {},
        "plan": {"execution_mode": "execute", "support_status": "complete",
                 "over_budget": False, "blocking_reasons": [],
                 "missing_capabilities": [], "advisory_steps": [], "actions": []},
        "commands": [],
        "chain": [],
        "results": [],
        "reply": "",
        "session_dir": str(session_dir),
        "fatal_error": str(error),
    }


def save_record(record: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def load_record(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def synthetic_record(
    question_id: str,
    layer: str,
    *,
    plan: Optional[Dict[str, Any]] = None,
    commands: Optional[List[Dict[str, Any]]] = None,
    chain: Optional[List[Dict[str, Any]]] = None,
    results: Optional[List[Dict[str, Any]]] = None,
    reply: str = "",
    run_config: Optional[Dict[str, Any]] = None,
    duration_sec: float = 10.0,
    session_dir: str = "/tmp/synthetic-session",
) -> Dict[str, Any]:
    """Minimal record builder for tests/experiments."""
    plan = plan or {}
    return {
        "schema": SCHEMA,
        "question_id": question_id,
        "layer": layer,
        "meta": {"timestamp": "synthetic", "duration_sec": duration_sec,
                 "planner": "synthetic", "executor": "local"},
        "run_config": run_config or {},
        "plan": {
            "execution_mode": plan.get("execution_mode", "plan" if layer != "EXEC" else "execute"),
            "support_status": plan.get("support_status", "complete"),
            "over_budget": plan.get("over_budget", False),
            "blocking_reasons": plan.get("blocking_reasons", []),
            "missing_capabilities": plan.get("missing_capabilities", []),
            "advisory_steps": plan.get("advisory_steps", []),
            "actions": plan.get("actions", []),
            "reply": plan.get("reply", ""),
        },
        "commands": commands or [],
        "chain": chain or [],
        "results": results or [],
        "reply": reply,
        "session_dir": session_dir,
        "fatal_error": None,
    }
