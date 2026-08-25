"""Evidence ledger: SQLite-backed tracking of all DFT task executions.

Provides durable, queryable records of every run attempt, including
input hashes, verifier verdicts, failure classifications, and recovery actions.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class LedgerRunRecord:
    """Extended run record for the ledger."""
    task_id: str
    task_type: str
    attempt: int
    success: bool
    input_hash: str
    walltime_sec: float
    verifier_passed: bool
    failure_reasons: List[str] = field(default_factory=list)
    recovery_actions: List[str] = field(default_factory=list)
    failure_kind: str = ""
    evidence_path: str = ""
    output_files: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)


class EvidenceLedger:
    """SQLite-backed ledger for tracking DFT task executions."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                task_type TEXT NOT NULL,
                attempt INTEGER NOT NULL,
                success INTEGER NOT NULL,
                input_hash TEXT NOT NULL,
                walltime_sec REAL NOT NULL,
                verifier_passed INTEGER NOT NULL,
                failure_reasons TEXT NOT NULL,
                recovery_actions TEXT NOT NULL,
                failure_kind TEXT NOT NULL DEFAULT '',
                evidence_path TEXT NOT NULL DEFAULT '',
                output_files TEXT NOT NULL DEFAULT '',
                timestamp TEXT NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_runs_task_id ON runs(task_id)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_runs_timestamp ON runs(timestamp)
        """)
        self._conn.commit()

    def record_run(
        self,
        run_record: "RunRecord | None" = None,
        *,
        task_id: str = "",
        task_type: str = "",
        attempt: int = 1,
        success: bool = False,
        input_hash: str = "",
        walltime_sec: float = 0.0,
        verifier_passed: bool = False,
        failure_reasons: Optional[List[str]] = None,
        recovery_actions: Optional[List[str]] = None,
        failure_kind: str = "",
        evidence_path: str = "",
        output_files: Optional[List[str]] = None,
    ) -> int:
        """Record a single run attempt in the ledger.

        Returns the row id.
        """
        if isinstance(run_record, LedgerRunRecord):
            record = run_record
        elif run_record is not None:
            record = LedgerRunRecord(
                task_id=task_id,
                task_type=task_type,
                attempt=run_record.attempt,
                success=run_record.success,
                input_hash=run_record.input_hash,
                walltime_sec=run_record.walltime_sec,
                verifier_passed=run_record.verifier_passed,
                failure_reasons=list(run_record.failure_reasons),
                recovery_actions=list(run_record.recovery_actions),
                failure_kind=failure_kind,
                evidence_path=evidence_path,
                output_files=list(output_files or []),
            )
        else:
            record = LedgerRunRecord(
                task_id=task_id,
                task_type=task_type,
                attempt=attempt,
                success=success,
                input_hash=input_hash,
                walltime_sec=walltime_sec,
                verifier_passed=verifier_passed,
                failure_reasons=list(failure_reasons or []),
                recovery_actions=list(recovery_actions or []),
                failure_kind=failure_kind,
                evidence_path=evidence_path,
                output_files=list(output_files or []),
            )
        row_id = self._conn.execute(
            """
            INSERT INTO runs (
                task_id, task_type, attempt, success, input_hash, walltime_sec,
                verifier_passed, failure_reasons, recovery_actions,
                failure_kind, evidence_path, output_files, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.task_id,
                record.task_type,
                record.attempt,
                1 if record.success else 0,
                record.input_hash,
                record.walltime_sec,
                1 if record.verifier_passed else 0,
                json.dumps(record.failure_reasons),
                json.dumps(record.recovery_actions),
                record.failure_kind,
                record.evidence_path,
                json.dumps(record.output_files),
                datetime.now(timezone.utc).isoformat(),
            ),
        ).lastrowid
        self._conn.commit()
        logger.debug("Recorded run %s for task %s (attempt %s)", row_id, record.task_id, record.attempt)
        return row_id

    def get_runs(
        self,
        task_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Query runs, optionally filtered by task_id."""
        query = "SELECT * FROM runs"
        params: List[Any] = []
        if task_id is not None:
            query += " WHERE task_id = ?"
            params.append(task_id)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(query, params).fetchall()
        results = []
        for row in rows:
            d = dict(row)
            d["failure_reasons"] = json.loads(d["failure_reasons"] or "[]")
            d["recovery_actions"] = json.loads(d["recovery_actions"] or "[]")
            d["output_files"] = json.loads(d["output_files"] or "[]")
            results.append(d)
        return results

    def get_run(self, task_id: str, attempt: int) -> Optional[Dict[str, Any]]:
        """Get a specific run by task_id and attempt number."""
        row = self._conn.execute(
            "SELECT * FROM runs WHERE task_id = ? AND attempt = ?",
            (task_id, attempt),
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["failure_reasons"] = json.loads(d["failure_reasons"] or "[]")
        d["recovery_actions"] = json.loads(d["recovery_actions"] or "[]")
        d["output_files"] = json.loads(d["output_files"] or "[]")
        return d

    def get_failure_patterns(self, task_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Summarize failure kinds and counts."""
        query = """
            SELECT failure_kind, COUNT(*) as count
            FROM runs
            WHERE success = 0 AND failure_kind != ''
        """
        params: List[Any] = []
        if task_id is not None:
            query += " AND task_id = ?"
            params.append(task_id)
        query += " GROUP BY failure_kind ORDER BY count DESC"
        rows = self._conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def close(self) -> None:
        self._conn.close()
