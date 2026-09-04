"""SQLite state persistence for graph runs (CatGo-style two-table schema)."""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from contextlib import closing
from pathlib import Path
from typing import List, Optional

from dft_forge.runtime.graph import GraphTemplate
from dft_forge.runtime.run import GraphRun, NodeRun
from dft_forge.runtime.states import NodeState, RunState, TERMINAL_RUN_STATES

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS graph_runs (
    id TEXT PRIMARY KEY,
    template_id TEXT NOT NULL,
    status TEXT NOT NULL,
    data TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS node_runs (
    run_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    status TEXT NOT NULL,
    data TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (run_id, node_id)
);
"""


class SQLiteStateStore:
    """Durable state store. Writes are idempotent (INSERT OR REPLACE)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with closing(self._connect()) as conn, conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    # ── Graph runs ───────────────────────────────────────────────────────────

    def save_run(self, run: GraphRun) -> None:
        run.touch()
        data = {
            "inputs": run.inputs,
            "created_at": run.created_at,
            "updated_at": run.updated_at,
        }
        with self._lock, closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT OR REPLACE INTO graph_runs (id, template_id, status, data, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (run.run_id, run.template_id, run.state.value, json.dumps(data),
                 str(run.created_at), str(run.updated_at)),
            )

    def load_run(self, run_id: str, template: GraphTemplate) -> Optional[GraphRun]:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT id, template_id, status, data FROM graph_runs WHERE id = ?", (run_id,)
            ).fetchone()
        if row is None:
            return None
        _, _, status, data = row
        payload = json.loads(data)
        run = GraphRun(
            run_id=row[0],
            template_id=row[1],
            state=RunState(status),
            inputs=payload.get("inputs", {}),
            created_at=float(payload.get("created_at", time.time())),
            updated_at=float(payload.get("updated_at", time.time())),
        )
        with closing(self._connect()) as conn:
            node_rows = conn.execute(
                "SELECT node_id, status, data FROM node_runs WHERE run_id = ?", (run_id,)
            ).fetchall()
        by_id = {n.id: n for n in template.nodes}
        for node_id, node_status, node_data in node_rows:
            if node_id not in by_id:
                continue
            d = json.loads(node_data)
            run.nodes[node_id] = NodeRun(
                node_id=node_id,
                state=NodeState(node_status),
                attempt=d.get("attempt", 0),
                repair_attempts=d.get("repair_attempts", 0),
                params=d.get("params", {}),
                outputs=d.get("outputs", {}),
                error=d.get("error"),
                job_id=d.get("job_id"),
                workdir=d.get("workdir"),
            )
        for spec in template.nodes:
            run.nodes.setdefault(spec.id, NodeRun(node_id=spec.id))
        return run

    def list_runs(self) -> List[dict]:
        # updated_at is stored as a float string — ORDER BY TEXT would sort
        # "9.5" after "10.5"; cast to REAL for numeric ordering
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT id, template_id, status, updated_at FROM graph_runs "
                "ORDER BY CAST(updated_at AS REAL) DESC"
            ).fetchall()
        return [{"run_id": r[0], "template_id": r[1], "status": r[2], "updated_at": r[3]} for r in rows]

    def template_id_for(self, run_id: str) -> Optional[str]:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT template_id FROM graph_runs WHERE id = ?", (run_id,)
            ).fetchone()
        return row[0] if row else None

    # ── Node runs ────────────────────────────────────────────────────────────

    def save_node_run(self, run: GraphRun, node: NodeRun) -> None:
        node.updated_at = time.time()
        data = {
            "attempt": node.attempt,
            "repair_attempts": node.repair_attempts,
            "params": node.params,
            "outputs": node.outputs,
            "error": node.error,
            "job_id": node.job_id,
            "workdir": node.workdir,
            "updated_at": node.updated_at,
        }
        with self._lock, closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT OR REPLACE INTO node_runs (run_id, node_id, status, data, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (run.run_id, node.node_id, node.state.value, json.dumps(data), str(node.updated_at)),
            )

    # ── Crash recovery ───────────────────────────────────────────────────────

    def resume_run(self, run_id: str, template: GraphTemplate) -> Optional[GraphRun]:
        """Load a run and reset in-flight nodes for rescheduling.

        Succeeded nodes keep their outputs (idempotent skip). Nodes that were
        Running/Ready/Repairing when the process died go back to Pending so the
        scheduler re-dispatches them. Terminal run states are returned as-is.
        """
        run = self.load_run(run_id, template)
        if run is None:
            return None
        if run.state in TERMINAL_RUN_STATES:
            return run
        for node in run.nodes.values():
            if node.state in (NodeState.RUNNING, NodeState.READY, NodeState.REPAIRING):
                # deliberate direct assignment: crash recovery resets
                # in-flight states that the normal transition table forbids
                node.state = NodeState.PENDING
            self._warn_broken_symlinks(node)
        run.state = RunState.VALIDATED
        self.save_run(run)
        return run

    @staticmethod
    def _warn_broken_symlinks(node: NodeRun) -> None:
        """Flag dead .save links before rescheduling: a node reusing a broken
        upstream symlink fails with a confusing QE error instead of a hint."""
        wd = Path(node.workdir) if node.workdir else None
        if wd is None or not wd.is_dir():
            return
        try:
            for entry in wd.iterdir():
                if entry.is_symlink() and not entry.exists():
                    logger.warning(
                        "broken symlink in node %s: %s -> %s (upstream .save moved or "
                        "deleted; the engine relinks it on dispatch)",
                        node.node_id, entry.name, entry.readlink(),
                    )
        except OSError:
            pass
