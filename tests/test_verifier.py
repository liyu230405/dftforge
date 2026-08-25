"""Tests for the scientific verifier."""

import pytest
from pathlib import Path
from dft_forge.verifier import ScientificVerifier, ConvergenceReport
from dft_forge.parser import QEParser, ParsedVCResult, ParsedCell


class TestScientificVerifier:
    @pytest.fixture
    def verifier(self):
        return ScientificVerifier(
            force_threshold_ry=1.0e-3,
            pressure_threshold_kbar=0.5,
        )
    
    @pytest.fixture
    def passing_job_result(self):
        """A job result that passed."""
        class FakeJob:
            stdout = """
     convergence has been achieved in  24    iterations
     !
     !    total energy              =     -15.67916468 Ry
     !
     JOB DONE.
     P=       0.12
     atom    1 type  1   force = 0.00000000 0.00000000 0.00000000
     atom    2 type  1   force = 0.00010000 0.00000000 0.00000000
     CELL_PARAMETERS (alat=  10.26000000)
      0.000000000  0.500000000  0.500000000
      0.500000000  0.000000000  0.500000000
      0.500000000  0.500000000  0.000000000
     number of atoms/cell =           2
"""
        return FakeJob()
    
    @pytest.fixture
    def passing_parsed(self):
        return ParsedVCResult(
            final_energy_ry=-15.67916468,
            final_energy_ev_per_atom=-53.35,
            max_force_ry_bohr=0.0001,
            pressure_kbar=0.12,
            n_iterations=24,
            converged=True,
            natoms=2,
            cell=ParsedCell(a_bohr=10.26, b_bohr=10.26, c_bohr=10.26, ibrav=2),
        )
    
    def test_pass_converged(self, verifier, passing_job_result, passing_parsed):
        report = verifier.verify_t1(passing_job_result, passing_parsed)
        assert report.passed is True
        assert report.checks["job_done"]["pass"] is True
        assert report.checks["scf_convergence"]["pass"] is True
        assert report.checks["forces"]["pass"] is True
    
    def test_fail_no_job_done(self, verifier, passing_parsed):
        class BadJob:
            stdout = "Calculation finished normally without completion marker"
        report = verifier.verify_t1(BadJob(), passing_parsed)
        assert report.passed is False
        assert any("JOB DONE" in r for r in report.failure_reasons)
    
    def test_fail_high_forces(self, verifier, passing_job_result):
        bad_parsed = ParsedVCResult(
            final_energy_ry=-15.0,
            max_force_ry_bohr=0.01,  # Above threshold
            pressure_kbar=0.0,
            natoms=2,
            converged=True,
            cell=ParsedCell(a_bohr=10.26, b_bohr=10.26, c_bohr=10.26, ibrav=2),
        )
        report = verifier.verify_t1(passing_job_result, bad_parsed)
        assert report.passed is False
        assert any("force" in r.lower() for r in report.failure_reasons)
    
    def test_fail_bad_cell(self, verifier, passing_job_result):
        bad_parsed = ParsedVCResult(
            final_energy_ry=-15.0,
            max_force_ry_bohr=0.0,
            pressure_kbar=0.0,
            natoms=2,
            converged=True,
            cell=ParsedCell(a_bohr=0.5, b_bohr=0.5, c_bohr=0.5, ibrav=2),  # Too small
        )
        report = verifier.verify_t1(passing_job_result, bad_parsed)
        assert report.passed is False
        assert any("cell" in r.lower() for r in report.failure_reasons)
    
    def test_warning_high_pressure(self, verifier, passing_job_result, passing_parsed):
        high_p_parsed = ParsedVCResult(
            final_energy_ry=-15.0,
            max_force_ry_bohr=0.0,
            pressure_kbar=10.0,  # Above threshold but only warning
            natoms=2,
            converged=True,
            cell=ParsedCell(a_bohr=10.26, b_bohr=10.26, c_bohr=10.26, ibrav=2),
        )
        report = verifier.verify_t1(passing_job_result, high_p_parsed)
        # Pressure is warning, not failure
        assert report.passed is True
        assert any("pressure" in w.lower() for w in report.warnings)


class TestVerifyT2Bands:
    @pytest.fixture
    def verifier(self):
        return ScientificVerifier()

    def test_bands_pass_all_checks(self, verifier, tmp_path):
        """Test successful T2 bands verification."""
        scf_stdout = """
     convergence has been achieved in  10    iterations
     !
     !    total energy              =     -7.83958234 Ry
     !
     convergence threshold = 1.0E-8
     number of k points = 64
     JOB DONE.
"""
        nscf_stdout = """
     !
     !    total energy              =     -7.83958234 Ry
     !
     number of k points = 512
     the Fermi energy is     6.5397 ev
     End of band structure calculation
     JOB DONE.
"""
        # Create bands.xml
        bands_xml = tmp_path / "si_bands.xml"
        bands_xml.write_text("""<?xml version="1.0"?>
<qes:espresso xmlns:qes="http://www.quantum-espresso.org/ns/qes/qes-1.0">
  <qes:output>
    <qes:band_structure nks="100" nbnd="16">
      <qes:ks_energies><qes:eigenvalues>""" + " ".join(["-5.0"] * 16) + """</qes:eigenvalues></qes:ks_energies>
    </qes:band_structure>
  </qes:output>
</qes:espresso>
""")

        report = verifier.verify_t2_bands(scf_stdout, nscf_stdout, bands_xml)
        assert report.passed is True
        assert "scf" in report.checks
        assert "nscf" in report.checks
        assert "bands_xml" in report.checks
        assert "bands_data" in report.checks
        assert report.checks["bands_data"]["n_bands"] == 16

    def test_bands_fail_scf(self, verifier, tmp_path):
        """Test T2 bands fails when SCF doesn't converge."""
        scf_stdout = "no convergence"
        nscf_stdout = "no convergence"
        bands_xml = tmp_path / "si_bands.xml"
        bands_xml.write_text("")

        report = verifier.verify_t2_bands(scf_stdout, nscf_stdout, bands_xml)
        assert report.passed is False
        assert any("SCF stage failed" in r for r in report.failure_reasons)

    def test_bands_fail_no_xml(self, verifier):
        """Test T2 bands fails when bands.xml is missing."""
        scf_stdout = _make_scf_stdout(converged=True)
        nscf_stdout = _make_scf_stdout(converged=True)
        bands_xml = Path("/nonexistent/bands.xml")

        report = verifier.verify_t2_bands(scf_stdout, nscf_stdout, bands_xml)
        assert report.passed is False
        assert any("bands.xml not found" in r for r in report.failure_reasons)


class TestVerifyT2DOS:
    @pytest.fixture
    def verifier(self):
        return ScientificVerifier()

    def test_dos_pass_all_checks(self, verifier, tmp_path):
        """Test successful T2 DOS verification."""
        scf_stdout = """
     convergence has been achieved in  10    iterations
     !
     !    total energy              =     -7.83958234 Ry
     !
     convergence threshold = 1.0E-8
     number of k points = 64
     JOB DONE.
"""
        nscf_stdout = """
     !
     !    total energy              =     -7.83958234 Ry
     !
     number of k points = 512
     the Fermi energy is     6.5397 ev
     End of band structure calculation
     JOB DONE.
"""
        # Create dos.dat
        import numpy as np
        dos_file = tmp_path / "si_dos.dat"
        energies = np.linspace(-10, 10, 100)
        dos = np.exp(-energies**2 / 2.0)
        np.savetxt(dos_file, np.column_stack([energies, dos]))

        report = verifier.verify_t2_dos(scf_stdout, nscf_stdout, dos_file)
        assert report.passed is True
        assert report.checks["scf"]["pass"] is True
        assert report.checks["dos_file"]["n_points"] == 100

    def test_dos_fail_scf(self, verifier):
        """Test T2 DOS fails when SCF doesn't converge."""
        scf_stdout = "no convergence"
        nscf_stdout = "no convergence"
        dos_file = Path("/nonexistent/dos.dat")

        report = verifier.verify_t2_dos(scf_stdout, nscf_stdout, dos_file)
        assert report.passed is False
        assert any("SCF stage failed" in r for r in report.failure_reasons)

    def test_dos_fail_no_file(self, verifier):
        """Test T2 DOS fails when dos.dat is missing."""
        scf_stdout = _make_scf_stdout(converged=True)
        nscf_stdout = _make_scf_stdout(converged=True)
        dos_file = Path("/nonexistent/dos.dat")

        report = verifier.verify_t2_dos(scf_stdout, nscf_stdout, dos_file)
        assert report.passed is False
        assert any("DOS file not found" in r for r in report.failure_reasons)


# Helper for verifier tests
def _make_scf_stdout(converged=True, n_iterations=10, energy_ry=-7.5):
    if converged:
        return f"""
     convergence has been achieved in  {n_iterations}  iterations
     !
     !    total energy              =     {energy_ry} Ry
     !
     convergence threshold = 1.0E-8
     number of k points = 64
     JOB DONE.
"""
    return "some text without convergence"
