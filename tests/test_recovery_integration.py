"""Tests for recovery controller integration with the runner."""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from dft_forge.runner import TaskRunner, RunRecord
from dft_forge.executor import FakeExecutor
from dft_forge.catalog import get_task_spec
from dft_forge.protocol.schemas import EvidenceBundle
from dft_forge.recovery import RecoveryController, FailureKind


def _make_job_result(success=True, stdout="", output_files=None, walltime=1.0,
                     error_message=""):
    result = MagicMock()
    result.success = success
    result.stdout = stdout
    result.stderr = ""
    result.output_files = output_files or []
    result.walltime_sec = walltime
    result.error_message = error_message
    result.returncode = 0 if success else 1
    return result


def _make_scf_stdout(converged=True, n_iterations=10, energy_ry=-7.5):
    if converged:
        return f"""
     convergence has been achieved in  {n_iterations}  iterations
     !
     !    total energy              =     {energy_ry} Ry
     !
     convergence threshold = 1.0E-8
     the Fermi energy is    5.123  ev
     number of k points = 64
     nbnd = 16
     JOB DONE.
"""
    return "some text without convergence"


class TestTaskRunnerRecoveryIntegration:
    """Tests that the runner uses the recovery controller correctly."""

    @pytest.fixture
    def pseudo_dir(self, tmp_path):
        pseudo_dir = tmp_path / "pseudos"
        pseudo_dir.mkdir()
        import shutil
        src = Path("/Users/liyu/Documents/Codex/dft-forge/assets/pseudos")
        for f in src.iterdir():
            if f.is_file():
                shutil.copy(f, pseudo_dir / f.name)
        return pseudo_dir

    def _create_runner(self, pseudo_dir):
        runner = TaskRunner(
            pseudo_dir=pseudo_dir,
            executor=FakeExecutor(),
            max_walltime_sec=600,
            max_retries=1,
        )
        return runner

    def test_t1_recovery_actions_populated(self, tmp_path, pseudo_dir):
        """Test that T1 runner populates recovery actions via recovery controller."""
        from dft_forge.verifier import ConvergenceReport

        runner = self._create_runner(pseudo_dir)
        task_id = "T1_Si_vcrelax"

        scf_stdout = "no convergence here"
        scf_result = _make_job_result(success=True, stdout=scf_stdout, walltime=2.0)

        with patch.object(runner.compiler, 'compile_t1', return_value="fake input"), \
             patch.object(runner.executor, 'run_pw', return_value=scf_result), \
             patch.object(runner.verifier, 'verify_t1',
                          return_value=MagicMock(
                              passed=False,
                              failure_reasons=["SCF did not converge"],
                              warnings=[],
                              summary="SCF NOT converged",
                              checks={},
                          )):
            result = runner.run_t1(task_id, tmp_path)

        assert result["status"] == "fail"
        assert len(runner.run_records) == 1
        assert len(runner.run_records[0].recovery_actions) > 0
        assert any("increase_ecut" in a for a in runner.run_records[0].recovery_actions)

    def test_t1_ledger_recorded(self, tmp_path, pseudo_dir):
        """Test that T1 runner records to ledger when provided."""
        from dft_forge.ledger import EvidenceLedger

        runner = self._create_runner(pseudo_dir)
        ledger = EvidenceLedger(db_path=tmp_path / "ledger.db")
        runner.ledger = ledger

        task_id = "T1_Si_vcrelax"
        scf_stdout = "no convergence"
        scf_result = _make_job_result(success=True, stdout=scf_stdout, walltime=2.0)

        with patch.object(runner.compiler, 'compile_t1', return_value="fake input"), \
             patch.object(runner.executor, 'run_pw', return_value=scf_result), \
             patch.object(runner.verifier, 'verify_t1',
                          return_value=MagicMock(
                              passed=False,
                              failure_reasons=["SCF did not converge"],
                              warnings=[],
                              summary="SCF NOT converged",
                              checks={},
                          )):
            runner.run_t1(task_id, tmp_path)

        runs = ledger.get_runs(task_id="T1_Si_vcrelax")
        assert len(runs) == 1
        assert runs[0]["failure_kind"] == "scf_not_converged"
        ledger.close()

    def test_t2_bands_recovery_actions_via_controller(self, tmp_path, pseudo_dir):
        """Test T2 bands uses recovery controller, not hardcoded strings."""
        runner = self._create_runner(pseudo_dir)
        task_id = "T2_Si_bands"

        scf_stdout = "no convergence"
        scf_result = _make_job_result(success=True, stdout=scf_stdout, walltime=2.0)

        fake_scf = tmp_path / "fake_scf.in"
        fake_scf.write_text("fake scf")
        fake_bands = tmp_path / "fake_bands.in"
        fake_bands.write_text("fake bands")

        with patch.object(runner.compiler, 'compile_scf', return_value="fake scf content"), \
             patch.object(runner.compiler, 'compile_bands_input', return_value="fake bands content"), \
             patch.object(runner.executor, 'run_pw', return_value=scf_result), \
             patch.object(runner.executor, 'run_bands_x'), \
             patch.object(runner.verifier, 'verify_scf',
                          return_value=MagicMock(passed=False, summary="SCF NOT converged")), \
             patch.object(runner.verifier, 'verify_t2_bands',
                          return_value=MagicMock(
                              passed=False,
                              checks={"scf": {"pass": False}},
                              summary="T2 bands: 0/3 checks passed",
                              failure_reasons=["SCF stage failed"],
                              warnings=[],
                          )):
            result = runner.run_t2_bands(task_id, tmp_path)

        assert result["status"] == "fail"
        evidence_data = json.loads((tmp_path / "evidence.json").read_text())
        # Should use recovery controller, not old hardcoded strings
        assert any("increase_ecut" in a or "increase_kpoints" in a for a in evidence_data["recovery_actions"])

    def test_t2_dos_recovery_actions_via_controller(self, tmp_path, pseudo_dir):
        """Test T2 DOS uses recovery controller."""
        runner = self._create_runner(pseudo_dir)
        task_id = "T2_Si_dos"

        scf_stdout = "no convergence"
        scf_result = _make_job_result(success=True, stdout=scf_stdout, walltime=2.0)

        with patch.object(runner.compiler, 'compile_scf', return_value="fake scf content"), \
             patch.object(runner.compiler, 'compile_dos_input', return_value="fake dos content"), \
             patch.object(runner.executor, 'run_pw', return_value=scf_result), \
             patch.object(runner.executor, 'run_dos_x'), \
             patch.object(runner.verifier, 'verify_scf',
                          return_value=MagicMock(passed=False, summary="SCF NOT converged")), \
             patch.object(runner.verifier, 'verify_t2_dos',
                          return_value=MagicMock(
                              passed=False,
                              checks={"scf": {"pass": False}},
                              summary="T2 DOS: 1/2 checks passed",
                              failure_reasons=["SCF stage failed"],
                              warnings=[],
                          )):
            result = runner.run_t2_dos(task_id, tmp_path)

        assert result["status"] == "fail"
        evidence_data = json.loads((tmp_path / "evidence.json").read_text())
        assert any("increase_ecut" in a for a in evidence_data["recovery_actions"])
