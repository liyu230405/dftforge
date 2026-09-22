"""Read-only run inspection, including graphs inside composite workflows."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from dft_forge.runtime.store import sqlite_connection


def _databases(root: Path):
    root = root.resolve()
    if not root.is_dir():
        return
    paths = [root / "graphs" / "graph_state.db"]
    paths.extend(sorted(root.glob("workflows/*/graphs/graph_state.db")))
    for path in paths:
        resolved = path.resolve()
        if path.is_symlink() or root not in resolved.parents or not path.is_file():
            continue
        yield path


def _connect(path: Path):
    return sqlite_connection(path, read_only=True, timeout=2)


def _payload(raw: str) -> dict:
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("Invalid stored run data")
    return value


def list_runs(root: Path) -> list[dict[str, Any]]:
    """Inspect existing databases without creating workspaces or executors."""
    root = root.resolve()
    runs = []
    for path in _databases(root):
        try:
            with _connect(path) as conn:
                has_snapshots = conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'graph_run_templates'"
                ).fetchone() is not None
                snapshot_column = (
                    "EXISTS (SELECT 1 FROM graph_run_templates t WHERE t.run_id = graph_runs.id)"
                    if has_snapshots else "0"
                )
                rows = conn.execute(
                    f"SELECT id, template_id, status, updated_at, data, {snapshot_column} FROM graph_runs"
                ).fetchall()
            for run_id, template_id, status, updated_at, raw, snapshot in rows:
                runs.append({
                    "run_id": run_id,
                    "template_id": template_id,
                    "status": status,
                    "updated_at": float(updated_at),
                    "workspace": path.parent.relative_to(root).as_posix(),
                    "metadata": _payload(raw).get("metadata", {}),
                    "has_template_snapshot": bool(snapshot),
                })
        except (sqlite3.Error, OSError, ValueError):
            continue
    return sorted(runs, key=lambda item: item["updated_at"], reverse=True)


def inspect_run(root: Path, run_id: str) -> dict[str, Any] | None:
    root = root.resolve()
    matches = [item for item in list_runs(root) if item["run_id"] == run_id]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError("Ambiguous run id in this session")
    summary = matches[0]
    path = root / summary["workspace"] / "graph_state.db"
    with _connect(path) as conn:
        # Read the run and its nodes from one SQLite snapshot.
        conn.execute("BEGIN")
        row = conn.execute(
            "SELECT status, data FROM graph_runs WHERE id = ?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        rows = conn.execute(
            "SELECT node_id, status, data FROM node_runs WHERE run_id = ? ORDER BY node_id",
            (run_id,),
        ).fetchall()
    payload = _payload(row[1])
    nodes = {}
    for node_id, state, raw in rows:
        data = _payload(raw)
        outputs = data.get("outputs") or {}
        # Curves/wavefunctions are downloaded as artifacts; the inspector
        # carries scalar results only so polling does not copy large arrays.
        nodes[node_id] = {
            "state": state,
            "attempt": data.get("attempt", 0),
            "repair_attempts": data.get("repair_attempts", 0),
            "error": data.get("error"),
            "outputs": {
                key: value for key, value in outputs.items()
                if value is None or isinstance(value, (str, int, float, bool))
            },
        }
    return {
        **summary,
        "status": row[0],
        "state": row[0],
        "inputs": payload.get("inputs", {}),
        "metadata": payload.get("metadata", {}),
        "nodes": nodes,
    }


def has_template_snapshot(root: Path, run: dict[str, Any]) -> bool:
    if "has_template_snapshot" in run:
        return bool(run["has_template_snapshot"])
    path = root / run["workspace"] / "graph_state.db"
    try:
        with _connect(path) as conn:
            return conn.execute(
                "SELECT 1 FROM graph_run_templates WHERE run_id = ?", (run["run_id"],)
            ).fetchone() is not None
    except sqlite3.Error:
        return False


def evidence_snapshot(root: Path, run_id: str) -> dict[str, Any] | None:
    """Read persisted evidence in one transaction, without resolving a tool.

    In particular, an unavailable legacy template is never replaced by the
    current registry template, and unresolved node parameters stay unresolved.
    """
    root = root.resolve()
    matches = [item for item in list_runs(root) if item["run_id"] == run_id]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError("Ambiguous run id in this session")
    workspace = matches[0]["workspace"]
    with _connect(root / workspace / "graph_state.db") as conn:
        conn.execute("BEGIN")
        row = conn.execute(
            "SELECT template_id, status, data FROM graph_runs WHERE id = ?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        nodes = conn.execute(
            "SELECT node_id, status, data FROM node_runs WHERE run_id = ? ORDER BY node_id",
            (run_id,),
        ).fetchall()
        has_table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'graph_run_templates'"
        ).fetchone()
        template = conn.execute(
            "SELECT template_json FROM graph_run_templates WHERE run_id = ?", (run_id,)
        ).fetchone() if has_table else None
    data = _payload(row[2])
    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    template_status = "unavailable_legacy"
    snapshot = None
    if template:
        try:
            snapshot = _payload(template[0])
            if snapshot.get("template_id") != row[0]:
                raise ValueError("Template id mismatch")
            template_status = "recorded"
        except (ValueError, TypeError):
            template_status, snapshot = "invalid", None
    return {
        "run": {
            "run_id": run_id, "template_id": row[0], "workspace": workspace,
            "state": row[1], "created_at": data.get("created_at"),
            "updated_at": data.get("updated_at"), "inputs": data.get("inputs", {}),
            "execution": metadata.get("execution", {}),
            "method_contract": metadata.get("method_contract"),
            "convergence_evidence": metadata.get("convergence_evidence"),
            "template_snapshot": snapshot, "template_snapshot_status": template_status,
        },
        "nodes": [{**_payload(raw), "node_id": node_id, "state": state} for node_id, state, raw in nodes],
    }
