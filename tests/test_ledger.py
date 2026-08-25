"""Tests for the evidence ledger."""

import json
import pytest
from pathlib import Path

from dft_forge.ledger import EvidenceLedger, LedgerRunRecord
from dft_forge.runner import RunRecord


class TestLedgerRunRecord:
    def test_to_dict(self):
        record = LedgerRunRecord(
            task_id="T1_Si_vcrelax",
            task_type="T1",
            attempt=1,
            success=True,
            input_hash="abc123",
            walltime_sec=10.5,
            verifier_passed=True,
            failure_reasons=[],
            recovery_actions=[],
            failure_kind="",
            evidence_path="/tmp/evidence.json",
            output_files=["/tmp/out"],
        )
        d = record.to_dict()
        assert d["task_id"] == "T1_Si_vcrelax"
        assert d["walltime_sec"] == 10.5
        assert d["timestamp"] is not None


class TestEvidenceLedger:
    @pytest.fixture
    def db_path(self, tmp_path):
        return tmp_path / "test_ledger.db"

    @pytest.fixture
    def ledger(self, db_path):
        return EvidenceLedger(db_path=db_path)

    def _make_run_record(self, attempt=1, success=True, **kwargs):
        defaults = dict(
            attempt=attempt,
            success=success,
            input_hash="hash123",
            walltime_sec=5.0,
            verifier_passed=True,
            failure_reasons=[],
            recovery_actions=[],
        )
        defaults.update(kwargs)
        return RunRecord(**defaults)

    def test_init_creates_schema(self, ledger, db_path):
        assert db_path.exists()
        # Should not raise on second init
        ledger2 = EvidenceLedger(db_path=db_path)
        ledger2.close()

    def test_record_run_returns_row_id(self, ledger):
        run_record = self._make_run_record()
        row_id = ledger.record_run(
            run_record,
            task_id="T1_Si_vcrelax",
            task_type="T1",
            failure_kind="scf_not_converged",
            evidence_path="/tmp/evidence.json",
            output_files=["/tmp/out.txt"],
        )
        assert isinstance(row_id, int)
        assert row_id > 0

    def test_record_run_persists_data(self, ledger):
        run_record = self._make_run_record(
            success=False,
            failure_reasons=["SCF did not converge"],
            recovery_actions=["increase_ecut"],
        )
        ledger.record_run(
            run_record,
            task_id="T1_Si_vcrelax",
            task_type="T1",
            failure_kind="scf_not_converged",
            evidence_path="/tmp/evidence.json",
            output_files=["/tmp/out.txt"],
        )

        runs = ledger.get_runs()
        assert len(runs) == 1
        assert runs[0]["task_id"] == "T1_Si_vcrelax"
        assert runs[0]["task_type"] == "T1"
        assert runs[0]["success"] == 0  # SQLite stores as integer
        assert runs[0]["failure_kind"] == "scf_not_converged"
        assert runs[0]["failure_reasons"] == ["SCF did not converge"]
        assert runs[0]["recovery_actions"] == ["increase_ecut"]
        assert runs[0]["output_files"] == ["/tmp/out.txt"]

    def test_record_run_multiple_attempts(self, ledger):
        run1 = self._make_run_record(attempt=1, success=False)
        run2 = self._make_run_record(attempt=2, success=True)
        ledger.record_run(run1, task_id="T1_Si_vcrelax", task_type="T1")
        ledger.record_run(run2, task_id="T1_Si_vcrelax", task_type="T1")

        runs = ledger.get_runs()
        assert len(runs) == 2
        assert runs[0]["attempt"] == 2  # Ordered by timestamp DESC
        assert runs[1]["attempt"] == 1

    def test_get_runs_filter_by_task_id(self, ledger):
        run1 = self._make_run_record()
        run2 = self._make_run_record()
        ledger.record_run(run1, task_id="T1_Si_vcrelax", task_type="T1")
        ledger.record_run(run2, task_id="T2_Si_bands", task_type="T2")

        si_runs = ledger.get_runs(task_id="T1_Si_vcrelax")
        assert len(si_runs) == 1
        assert si_runs[0]["task_id"] == "T1_Si_vcrelax"

        bands_runs = ledger.get_runs(task_id="T2_Si_bands")
        assert len(bands_runs) == 1
        assert bands_runs[0]["task_id"] == "T2_Si_bands"

    def test_get_runs_limit(self, ledger):
        for i in range(10):
            run = self._make_run_record()
            ledger.record_run(run, task_id=f"Task_{i}", task_type="T1")

        runs = ledger.get_runs(limit=5)
        assert len(runs) == 5

    def test_get_run_by_task_and_attempt(self, ledger):
        run1 = self._make_run_record(attempt=1)
        run2 = self._make_run_record(attempt=2)
        ledger.record_run(run1, task_id="T1_Si_vcrelax", task_type="T1")
        ledger.record_run(run2, task_id="T1_Si_vcrelax", task_type="T1")

        run = ledger.get_run("T1_Si_vcrelax", attempt=1)
        assert run is not None
        assert run["attempt"] == 1

        run_missing = ledger.get_run("T1_Si_vcrelax", attempt=99)
        assert run_missing is None

    def test_get_failure_patterns(self, ledger):
        run1 = self._make_run_record(success=False, failure_reasons=["SCF did not converge"])
        run2 = self._make_run_record(success=False, failure_reasons=["NSCF did not converge"])
        run3 = self._make_run_record(success=True)
        ledger.record_run(run1, task_id="T1_Si_vcrelax", task_type="T1", failure_kind="scf_not_converged")
        ledger.record_run(run2, task_id="T1_Si_vcrelax", task_type="T1", failure_kind="nscf_not_converged")
        ledger.record_run(run3, task_id="T1_Si_vcrelax", task_type="T1", failure_kind="")

        patterns = ledger.get_failure_patterns()
        assert len(patterns) == 2
        kinds = {p["failure_kind"]: p["count"] for p in patterns}
        assert kinds["scf_not_converged"] == 1
        assert kinds["nscf_not_converged"] == 1

    def test_get_failure_patterns_filtered_by_task(self, ledger):
        run1 = self._make_run_record(success=False)
        run2 = self._make_run_record(success=False)
        ledger.record_run(run1, task_id="T1_Si_vcrelax", task_type="T1", failure_kind="scf_not_converged")
        ledger.record_run(run2, task_id="T2_Si_bands", task_type="T2", failure_kind="nscf_not_converged")

        patterns = ledger.get_failure_patterns(task_id="T1_Si_vcrelax")
        assert len(patterns) == 1
        assert patterns[0]["failure_kind"] == "scf_not_converged"

    def test_empty_failure_patterns(self, ledger):
        run = self._make_run_record(success=True)
        ledger.record_run(run, task_id="T1_Si_vcrelax", task_type="T1")

        patterns = ledger.get_failure_patterns()
        assert patterns == []

    def test_timestamp_is_set(self, ledger):
        run = self._make_run_record()
        ledger.record_run(run, task_id="T1_Si_vcrelax", task_type="T1")
        runs = ledger.get_runs()
        assert "timestamp" in runs[0]
        assert "T" in runs[0]["timestamp"]  # ISO format

    def test_ledger_repr_or_usable_after_multiple_operations(self, ledger):
        run1 = self._make_run_record(attempt=1, success=False)
        run2 = self._make_run_record(attempt=2, success=False)
        run3 = self._make_run_record(attempt=3, success=True)
        ledger.record_run(run1, task_id="T1_Si_vcrelax", task_type="T1", failure_kind="scf_not_converged")
        ledger.record_run(run2, task_id="T1_Si_vcrelax", task_type="T1", failure_kind="scf_not_converged")
        ledger.record_run(run3, task_id="T1_Si_vcrelax", task_type="T1", failure_kind="")

        all_runs = ledger.get_runs()
        assert len(all_runs) == 3

        specific = ledger.get_run("T1_Si_vcrelax", attempt=2)
        assert specific["failure_kind"] == "scf_not_converged"

        patterns = ledger.get_failure_patterns()
        assert len(patterns) == 1
        assert patterns[0]["count"] == 2
