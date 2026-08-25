"""Tests for the QE parser."""

import pytest
from dft_forge.parser import QEParser, ParsedVCResult, ParsedSCF


class TestParseSCF:
    def test_converged_scf(self):
        stdout = """
     convergence has been achieved in  12    iterations
     !
     !    total energy              =     -7.83958234 Ry
     !
     convergence threshold = 1.0E-8
"""
        result = QEParser.parse_scf(stdout)
        assert result.converged is True
        assert result.n_iterations == 12
        assert result.total_energy_ry == -7.83958234
        assert result.total_energy_ev == pytest.approx(-7.83958234 * 13.605693122994, rel=1e-6)
    
    def test_unconverged_scf(self):
        stdout = "some text without convergence"
        result = QEParser.parse_scf(stdout)
        assert result.converged is False
    
    def test_last_energy_taken(self):
        stdout = """
     convergence has been achieved in  12    iterations
     !
     !    total energy              =     -7.00000000 Ry
     !
     !    total energy              =     -7.83958234 Ry
     !
"""
        result = QEParser.parse_scf(stdout)
        assert result.total_energy_ry == -7.83958234  # Last one


class TestParseVCRelax:
    def test_basic_vcrelax(self):
        stdout = """
     bravais-lattice index =           2
     !
     !    total energy              =     -15.67916468 Ry
     !
     convergence has been achieved in  24    iterations
     P=       0.12
     CELL_PARAMETERS (alat=   5.43100000)
      0.000000000  0.500000000  0.500000000
      0.500000000  0.000000000  0.500000000
      0.500000000  0.500000000  0.000000000
     number of atoms/cell =           2
"""
        result = QEParser.parse_vc_relax(stdout)
        assert result.final_energy_ry == -15.67916468
        assert result.n_iterations == 24
        assert result.pressure_kbar == 0.12
        assert result.natoms == 2
        assert result.cell is not None
    
    def test_force_parsing(self):
        stdout = """
     atom    1 type  1   force = 0.00000000 0.00000000 0.00000000
     atom    2 type  1   force = 0.00100000 0.00000000 0.00000000
"""
        result = QEParser.parse_vc_relax(stdout)
        assert result.max_force_ry_bohr == 0.001
    
    def test_energy_per_atom(self):
        stdout = """
     !
     !    total energy              =     -31.35832936 Ry
     !
     convergence has been achieved in  24    iterations
     number of atoms/cell =           4
"""
        result = QEParser.parse_vc_relax(stdout)
        assert result.natoms == 4
        assert result.final_energy_ev_per_atom == pytest.approx(
            -31.35832936 * 13.605693122994 / 4, rel=1e-6
        )
