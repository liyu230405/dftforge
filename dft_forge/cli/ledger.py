from __future__ import annotations

from pathlib import Path

from ._helpers import _error_response, _success_response


def cmd_ledger_record(args: argparse.Namespace) -> int:
    from dft_forge.ledger import EvidenceLedger, LedgerRunRecord

    db_path = Path(args.database) if args.database else Path("ledger.db")
    db_path = db_path.resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    ledger = EvidenceLedger(db_path=db_path)

    record = LedgerRunRecord(
        task_id=args.task_id,
        task_type=args.task_type,
        attempt=int(args.attempt),
        success=args.success.lower() in {"true", "1", "yes"},
        input_hash=args.input_hash or "",
        walltime_sec=float(args.walltime_sec or 0.0),
        verifier_passed=args.verifier_passed.lower() in {"true", "1", "yes"},
        failure_reasons=[s.strip() for s in (args.failure_reasons or "").split(",") if s.strip()],
        recovery_actions=[s.strip() for s in (args.recovery_actions or "").split(",") if s.strip()],
        failure_kind=args.failure_kind or "",
        evidence_path=args.evidence_path or "",
        output_files=[s.strip() for s in (args.output_files or "").split(",") if s.strip()],
    )
    row_id = ledger.record_run(record)
    data = {
        "command": "ledger.record",
        "database": str(db_path),
        "row_id": row_id,
        "task_id": record.task_id,
        "attempt": record.attempt,
    }
    ledger.close()
    return _success_response(data, args.output)



def cmd_ledger_query(args: argparse.Namespace) -> int:
    from dft_forge.ledger import EvidenceLedger

    db_path = Path(args.database).resolve() if args.database else Path("ledger.db").resolve()
    if not db_path.exists():
        return _error_response(f"Ledger database not found: {db_path}")

    ledger = EvidenceLedger(db_path=db_path)
    task_id = args.task_id if args.task_id else None
    runs = ledger.get_runs(task_id=task_id, limit=int(args.limit))
    data = {
        "command": "ledger.query",
        "database": str(db_path),
        "task_id": task_id,
        "count": len(runs),
        "runs": runs,
    }
    ledger.close()
    return _success_response(data, args.output)



def cmd_ledger_patterns(args: argparse.Namespace) -> int:
    from dft_forge.ledger import EvidenceLedger

    db_path = Path(args.database).resolve() if args.database else Path("ledger.db").resolve()
    if not db_path.exists():
        return _error_response(f"Ledger database not found: {db_path}")

    ledger = EvidenceLedger(db_path=db_path)
    task_id = args.task_id if args.task_id else None
    patterns = ledger.get_failure_patterns(task_id=task_id)
    data = {
        "command": "ledger.patterns",
        "database": str(db_path),
        "task_id": task_id,
        "patterns": patterns,
    }
    ledger.close()
    return _success_response(data, args.output)
