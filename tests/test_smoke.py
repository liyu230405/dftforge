"""Smoke tests: run actual QE calculations for Si, Al, MgO."""

import pytest
import subprocess
from pathlib import Path
from dft_forge.runner import run_task_deterministic


def has_qe():
    return subprocess.run(["which", "pw.x"], capture_output=True).returncode == 0


@pytest.mark.skipif(not has_qe(), reason="QE not installed")
class TestSiSmoke:
    def test_si_vcrelax(self, tmp_path):
        result = run_task_deterministic("T1_Si_vcrelax", tmp_path)
        
        assert result["status"] == "pass", f"Si vc-relax failed: {result.get('verifier_summary')}"
        assert result["verifier_passed"] is True
        assert result["physical_results"]["final_energy_ry"] < 0  # Bound system
        assert result["physical_results"]["natoms"] == 2
        assert (tmp_path / "result.json").exists()
        assert (tmp_path / "evidence.json").exists()
        assert (tmp_path / "Si_vcrelax.out").exists()


@pytest.mark.skipif(not has_qe(), reason="QE not installed")
class TestAlSmoke:
    def test_al_vcrelax(self, tmp_path):
        result = run_task_deterministic("T1_Al_vcrelax", tmp_path)
        
        assert result["status"] == "pass", f"Al vc-relax failed: {result.get('verifier_summary')}"
        assert result["verifier_passed"] is True
        assert result["physical_results"]["final_energy_ry"] < 0
        assert result["physical_results"]["natoms"] == 1


@pytest.mark.skipif(not has_qe(), reason="QE not installed")
class TestMgOSmoke:
    def test_mgo_vcrelax(self, tmp_path):
        result = run_task_deterministic("T1_MgO_vcrelax", tmp_path)
        
        assert result["status"] == "pass", f"MgO vc-relax failed: {result.get('verifier_summary')}"
        assert result["verifier_passed"] is True
        assert result["physical_results"]["final_energy_ry"] < 0
        assert result["physical_results"]["natoms"] == 2
