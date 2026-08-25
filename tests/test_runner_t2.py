"""Tests for the T2 task runner (bands and DOS workflows)."""

import json
import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dft_forge.runner import TaskRunner, RunRecord
from dft_forge.executor import FakeExecutor
from dft_forge.catalog import get_task_spec
from dft_forge.protocol.schemas import EvidenceBundle


# ── Fixtures ──────────────────────────────────────────────────────────────────

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


def _make_bands_xml(tmp_path, prefix="si", n_bands=8, n_kpoints=200, band_gap=0.0):
    """Create a minimal bands.xml file for testing."""
    xml_content = f"""<?xml version="1.0"?>
<qes:espresso xmlns:qes="http://www.quantum-espresso.org/ns/qes/qes-1.0">
  <qes:output>
    <qes:band_structure nk="{n_kpoints}" nbnd="{n_bands}">
"""
    for k in range(n_kpoints):
        evals = " ".join([f"{-5.0 + i * 0.1 + k * 0.001:.6f}" for i in range(n_bands)])
        xml_content += f'      <qes:ks_energies><qes:eigenvalues>{evals}</qes:eigenvalues></qes:ks_energies>\n'
    xml_content += """    </qes:band_structure>
  </qes:output>
</qes:espresso>
"""
    xml_path = tmp_path / f"{prefix}_bands.xml"
    xml_path.write_text(xml_content)
    return xml_path


def _make_dos_dat(tmp_path, prefix="si", n_points=1000):
    """Create a minimal dos.dat file for testing."""
    import numpy as np
    energies = np.linspace(-10, 10, n_points)
    # Gaussian-like DOS centered at 0
    dos = np.exp(-energies**2 / 2.0)
    data = np.column_stack([energies, dos])
    dos_path = tmp_path / f"{prefix}_dos.dat"
    np.savetxt(dos_path, data)
    return dos_path


class TestTaskRunnerT2Bands:
    """Tests for run_t2_bands with mocked QE execution."""

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

    def test_run_t2_bands_success(self, tmp_path, pseudo_dir):
        """Test successful T2 bands workflow with mocked executor."""
        runner = self._create_runner(pseudo_dir)
        task_id = "T2_Si_bands"

        scf_stdout = _make_scf_stdout(converged=True, n_iterations=10, energy_ry=-7.83958234)
        nscf_stdout = _make_scf_stdout(converged=True, n_iterations=5, energy_ry=-7.83958234)

        # Create bands.xml
        bands_xml = _make_bands_xml(tmp_path, prefix="si", n_bands=16, n_kpoints=100)

        scf_result = _make_job_result(
            success=True, stdout=scf_stdout,
            output_files=[str(tmp_path / "si_scf.save" / "data-file-schema.xml")],
            walltime=5.0,
        )
        nscf_result = _make_job_result(
            success=True, stdout=nscf_stdout,
            output_files=[str(tmp_path / "si_nscf_bands.save" / "data-file-schema.xml")],
            walltime=3.0,
        )
        bands_result = _make_job_result(
            success=True, stdout="BANDS COMPUTED",
            output_files=[str(bands_xml)],
            walltime=2.0,
        )

        with patch.object(runner.compiler, 'compile_scf', side_effect=lambda *a, **kw: self._write_fake_scf(tmp_path / "fake_scf.in", *a, **kw)), \
             patch.object(runner.compiler, 'compile_bands_input', side_effect=lambda *a, **kw: self._write_fake_bands(tmp_path / "fake_bands.in", *a, **kw)), \
             patch.object(runner.executor, 'run_pw', side_effect=[scf_result, nscf_result]), \
             patch.object(runner.executor, 'run_bands_x', return_value=bands_result), \
             patch.object(runner.verifier, 'verify_scf', side_effect=[
                 MagicMock(passed=True, checks={"scf": {"pass": True}}, summary="SCF converged"),
                 MagicMock(passed=True, checks={"scf": {"pass": True}}, summary="SCF converged"),
             ]), \
             patch.object(runner.verifier, 'verify_t2_bands',
                          return_value=MagicMock(
                              passed=True,
                              checks={"scf": {"pass": True}, "nscf": {"pass": True}, "bands_xml": {"pass": True}},
                              summary="T2 bands: 3/3 checks passed",
                              failure_reasons=[],
                              warnings=[],
                          )):
            result = runner.run_t2_bands(task_id, tmp_path)

        assert result["task_id"] == "T2_Si_bands"
        assert result["task_type"] == "T2"
        assert result["subtype"] == "bands"
        assert result["status"] == "pass"
        assert result["qe_calls"] == 3
        assert result["walltime_sec"] == pytest.approx(10.0)
        assert "scf" in result["input_files"]
        assert "nscf" in result["input_files"]
        assert "bands" in result["input_files"]
        assert (tmp_path / "result.json").exists()
        assert (tmp_path / "evidence.json").exists()

    def _write_fake_scf(self, path, material, *args, **kwargs):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fake scf input")
        return "fake scf content"

    def _write_fake_bands(self, path, material, *args, **kwargs):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fake bands input")
        return "fake bands content"

    def test_run_t2_bands_scf_failure(self, tmp_path, pseudo_dir):
        """Test T2 bands workflow when SCF fails."""
        runner = self._create_runner(pseudo_dir)
        task_id = "T2_Si_bands"

        scf_stdout = "no convergence here"
        scf_result = _make_job_result(success=True, stdout=scf_stdout, walltime=2.0)

        with patch.object(runner.compiler, 'compile_scf', side_effect=lambda *a, **kw: self._write_fake_scf(tmp_path / "fake_scf.in", *a, **kw)), \
             patch.object(runner.compiler, 'compile_bands_input', side_effect=lambda *a, **kw: self._write_fake_bands(tmp_path / "fake_bands.in", *a, **kw)), \
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
        assert result["verifier_passed"] is False
        assert "SCF stage failed" in result["failure_reasons"]
        # Check evidence.json for recovery_actions from recovery controller
        evidence_data = json.loads((tmp_path / "evidence.json").read_text())
        assert any("increase_ecut" in a for a in evidence_data["recovery_actions"])

    def test_run_t2_bands_recovery_actions(self, tmp_path, pseudo_dir):
        """Test recovery actions are recorded on failure."""
        runner = self._create_runner(pseudo_dir)
        task_id = "T2_Si_bands"

        scf_stdout = _make_scf_stdout(converged=True)
        nscf_stdout = "no convergence in nscf"

        scf_result = _make_job_result(success=True, stdout=scf_stdout, walltime=2.0)
        nscf_result = _make_job_result(success=True, stdout=nscf_stdout, walltime=2.0)
        bands_result = _make_job_result(success=True, stdout="BANDS DONE", walltime=1.0)

        with patch.object(runner.compiler, 'compile_scf', side_effect=lambda *a, **kw: self._write_fake_scf(tmp_path / "fake_scf.in", *a, **kw)), \
             patch.object(runner.compiler, 'compile_bands_input', side_effect=lambda *a, **kw: self._write_fake_bands(tmp_path / "fake_bands.in", *a, **kw)), \
             patch.object(runner.executor, 'run_pw', side_effect=[scf_result, nscf_result]), \
             patch.object(runner.executor, 'run_bands_x', return_value=bands_result), \
             patch.object(runner.verifier, 'verify_scf', side_effect=[
                 MagicMock(passed=True),
                 MagicMock(passed=False),
             ]), \
             patch.object(runner.verifier, 'verify_t2_bands',
                          return_value=MagicMock(
                              passed=False,
                              checks={},
                              summary="T2 bands: 0/3 checks passed",
                              failure_reasons=["NSCF stage did not converge"],
                              warnings=[],
                          )):
            result = runner.run_t2_bands(task_id, tmp_path)

        assert result["status"] == "fail"
        # Check evidence.json for recovery_actions from recovery controller
        evidence_data = json.loads((tmp_path / "evidence.json").read_text())
        assert any("increase_nbnd" in a for a in evidence_data["recovery_actions"])


class TestTaskRunnerT2DOS:
    """Tests for run_t2_dos with mocked QE execution."""

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

    def _write_fake_scf(self, path, material, *args, **kwargs):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fake scf input")
        return "fake scf content"

    def _write_fake_dos(self, path, material, *args, **kwargs):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fake dos input")
        return "fake dos content"

    def test_run_t2_dos_success(self, tmp_path, pseudo_dir):
        """Test successful T2 DOS workflow with mocked executor."""
        runner = self._create_runner(pseudo_dir)
        task_id = "T2_Si_dos"

        scf_stdout = _make_scf_stdout(converged=True, n_iterations=10, energy_ry=-7.83958234)
        nscf_stdout = _make_scf_stdout(converged=True, n_iterations=5, energy_ry=-7.83958234)

        # Create dos.dat
        dos_dat = _make_dos_dat(tmp_path, prefix="si", n_points=500)

        scf_result = _make_job_result(success=True, stdout=scf_stdout, walltime=5.0)
        nscf_result = _make_job_result(success=True, stdout=nscf_stdout, walltime=3.0)
        dos_result = _make_job_result(success=True, stdout="DOS COMPUTED", output_files=[str(dos_dat)], walltime=2.0)

        with patch.object(runner.compiler, 'compile_scf', side_effect=lambda *a, **kw: self._write_fake_scf(tmp_path / "fake_scf.in", *a, **kw)), \
             patch.object(runner.compiler, 'compile_dos_input', side_effect=lambda *a, **kw: self._write_fake_dos(tmp_path / "fake_dos.in", *a, **kw)), \
             patch.object(runner.executor, 'run_pw', side_effect=[scf_result, nscf_result]), \
             patch.object(runner.executor, 'run_dos_x', return_value=dos_result), \
             patch.object(runner.verifier, 'verify_scf', side_effect=[
                 MagicMock(passed=True, checks={"scf": {"pass": True}}, summary="SCF converged"),
                 MagicMock(passed=True, checks={"scf": {"pass": True}}, summary="SCF converged"),
             ]), \
             patch.object(runner.verifier, 'verify_t2_dos',
                          return_value=MagicMock(
                              passed=True,
                              checks={"scf": {"pass": True}, "dos_file": {"pass": True, "n_points": 500}},
                              summary="T2 DOS: 2/2 checks passed",
                              failure_reasons=[],
                              warnings=[],
                          )):
            result = runner.run_t2_dos(task_id, tmp_path)

        assert result["task_id"] == "T2_Si_dos"
        assert result["task_type"] == "T2"
        assert result["subtype"] == "dos"
        assert result["status"] == "pass"
        assert result["qe_calls"] == 3
        assert result["physical_results"]["n_energy_points"] == 500
        assert result["physical_results"]["dos_at_fermi"] > 0
        assert (tmp_path / "result.json").exists()
        assert (tmp_path / "evidence.json").exists()

    def test_run_t2_dos_dos_failure(self, tmp_path, pseudo_dir):
        """Test T2 DOS workflow when dos.x fails."""
        runner = self._create_runner(pseudo_dir)
        task_id = "T2_Si_dos"

        scf_stdout = _make_scf_stdout(converged=True)
        nscf_stdout = _make_scf_stdout(converged=True)

        scf_result = _make_job_result(success=True, stdout=scf_stdout, walltime=2.0)
        nscf_result = _make_job_result(success=True, stdout=nscf_stdout, walltime=2.0)
        dos_result = _make_job_result(success=False, stdout="", error_message="dos.x crashed", walltime=1.0)

        with patch.object(runner.compiler, 'compile_scf', side_effect=lambda *a, **kw: self._write_fake_scf(tmp_path / "fake_scf.in", *a, **kw)), \
             patch.object(runner.compiler, 'compile_dos_input', side_effect=lambda *a, **kw: self._write_fake_dos(tmp_path / "fake_dos.in", *a, **kw)), \
             patch.object(runner.executor, 'run_pw', side_effect=[scf_result, nscf_result]), \
             patch.object(runner.executor, 'run_dos_x', return_value=dos_result), \
             patch.object(runner.verifier, 'verify_scf', side_effect=[
                 MagicMock(passed=True),
                 MagicMock(passed=True),
             ]), \
             patch.object(runner.verifier, 'verify_t2_dos',
                          return_value=MagicMock(
                              passed=False,
                              checks={"dos_file": {"pass": False}},
                              summary="T2 DOS: 1/2 checks passed",
                              failure_reasons=["DOS file not found"],
                              warnings=[],
                          )):
            result = runner.run_t2_dos(task_id, tmp_path)

        assert result["status"] == "fail"
        evidence_data = json.loads((tmp_path / "evidence.json").read_text())
        assert any("check_nscf_output" in a for a in evidence_data["recovery_actions"])

    def test_run_t2_dos_evidence_bundle(self, tmp_path, pseudo_dir):
        """Test that evidence bundle is correctly built for DOS."""
        runner = self._create_runner(pseudo_dir)
        task_id = "T2_Si_dos"

        scf_stdout = _make_scf_stdout(converged=True)
        nscf_stdout = _make_scf_stdout(converged=True)
        dos_dat = _make_dos_dat(tmp_path, prefix="si", n_points=100)

        scf_result = _make_job_result(success=True, stdout=scf_stdout, walltime=2.0)
        nscf_result = _make_job_result(success=True, stdout=nscf_stdout, walltime=2.0)
        dos_result = _make_job_result(success=True, stdout="", output_files=[str(dos_dat)], walltime=1.0)

        with patch.object(runner.compiler, 'compile_scf', side_effect=lambda *a, **kw: self._write_fake_scf(tmp_path / "fake_scf.in", *a, **kw)), \
             patch.object(runner.compiler, 'compile_dos_input', side_effect=lambda *a, **kw: self._write_fake_dos(tmp_path / "fake_dos.in", *a, **kw)), \
             patch.object(runner.executor, 'run_pw', side_effect=[scf_result, nscf_result]), \
             patch.object(runner.executor, 'run_dos_x', return_value=dos_result), \
             patch.object(runner.verifier, 'verify_scf', side_effect=[
                 MagicMock(passed=True),
                 MagicMock(passed=True),
             ]), \
             patch.object(runner.verifier, 'verify_t2_dos',
                          return_value=MagicMock(
                              passed=True,
                              checks={"scf": {"pass": True}, "dos_file": {"pass": True, "n_points": 100}},
                              summary="T2 DOS: 2/2 checks passed",
                              failure_reasons=[],
                              warnings=[],
                          )):
            result = runner.run_t2_dos(task_id, tmp_path)

        # Check evidence.json was written and has correct structure
        evidence_path = tmp_path / "evidence.json"
        assert evidence_path.exists()
        evidence_data = json.loads(evidence_path.read_text())
        assert evidence_data["task_id"] == "T2_Si_dos"
        assert evidence_data["task_type"] == "T2"
        assert evidence_data["status"] == "pass"
        assert len(evidence_data["commands"]) == 3
        assert "pw.x" in evidence_data["commands"][0]
        assert "dos.x" in evidence_data["commands"][2]
        assert evidence_data["total_walltime_sec"] == pytest.approx(5.0)


class TestRunTaskDeterministicT2:
    """Test the top-level run_task_deterministic dispatcher for T2."""

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

    def _write_fake_scf(self, path, material, *args, **kwargs):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fake scf input")
        return "fake scf content"

    def _write_fake_bands(self, path, material, *args, **kwargs):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fake bands input")
        return "fake bands content"

    def _write_fake_dos(self, path, material, *args, **kwargs):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fake dos input")
        return "fake dos content"

    def test_dispatch_t2_bands(self, tmp_path, pseudo_dir):
        """Test that run_task_deterministic dispatches T2 bands correctly."""
        from dft_forge.runner import run_task_deterministic, TaskRunner

        scf_stdout = _make_scf_stdout(converged=True)
        nscf_stdout = _make_scf_stdout(converged=True)
        bands_xml = _make_bands_xml(tmp_path, prefix="si")

        scf_result = _make_job_result(success=True, stdout=scf_stdout, walltime=2.0)
        nscf_result = _make_job_result(success=True, stdout=nscf_stdout, walltime=2.0)
        bands_result = _make_job_result(success=True, stdout="", output_files=[str(bands_xml)], walltime=1.0)

        with patch.object(TaskRunner, '__init__', lambda self, *a, **kw: None), \
             patch.object(TaskRunner, 'run_t2_bands', return_value={"task_id": "T2_Si_bands", "status": "pass"}) as mock_run:
            # Need to set attributes manually since we bypass __init__
            result = run_task_deterministic("T2_Si_bands", tmp_path, pseudo_dir=pseudo_dir)

        mock_run.assert_called_once_with("T2_Si_bands", tmp_path, retry_params=None)
        assert result["status"] == "pass"

    def test_dispatch_t2_dos(self, tmp_path, pseudo_dir):
        """Test that run_task_deterministic dispatches T2 DOS correctly."""
        from dft_forge.runner import run_task_deterministic, TaskRunner

        dos_dat = _make_dos_dat(tmp_path, prefix="si")

        scf_result = _make_job_result(success=True, stdout=_make_scf_stdout(), walltime=2.0)
        nscf_result = _make_job_result(success=True, stdout=_make_scf_stdout(), walltime=2.0)
        dos_result = _make_job_result(success=True, stdout="", output_files=[str(dos_dat)], walltime=1.0)

        with patch.object(TaskRunner, '__init__', lambda self, *a, **kw: None), \
             patch.object(TaskRunner, 'run_t2_dos', return_value={"task_id": "T2_Si_dos", "status": "pass"}) as mock_run:
            result = run_task_deterministic("T2_Si_dos", tmp_path, pseudo_dir=pseudo_dir)

        mock_run.assert_called_once_with("T2_Si_dos", tmp_path, retry_params=None)
        assert result["status"] == "pass"
