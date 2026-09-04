"""Tests for the QE parser."""

import numpy as np
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

    def test_force_uses_final_configuration_only(self):
        """Early BFGS steps are far from equilibrium; only the block after
        'Begin final coordinates' (or the last block) represents the result."""
        stdout = """
     atom    1 type  1   force = 1.00000000 0.00000000 0.00000000
     atom    2 type  1   force = 0.00000000 0.00000000 0.00000000

Begin final coordinates

End final coordinates

     atom    1 type  1   force = 0.00010000 0.00000000 0.00000000
     atom    2 type  1   force = 0.00000000 0.00000000 0.00000000
"""
        result = QEParser.parse_vc_relax(stdout)
        assert result.max_force_ry_bohr == 1.0e-4

    def test_force_last_block_without_final_marker(self):
        stdout = """
     atom    1 type  1   force = 0.50000000 0.00000000 0.00000000

     atom    1 type  1   force = 0.00020000 0.00000000 0.00000000
"""
        result = QEParser.parse_vc_relax(stdout)
        assert result.max_force_ry_bohr == 2.0e-4

    def test_converged_requires_real_marker(self):
        """An energy line alone does not mean the relax converged."""
        stdout = """
     !    total energy              =     -15.67916468 Ry
"""
        result = QEParser.parse_vc_relax(stdout)
        assert result.final_energy_ry == -15.67916468
        assert result.converged is False

    def test_scientific_notation_forces(self):
        stdout = """
     atom    1 type  1   force =   -1.234567E-04  0.000000E+00  0.000000E+00
"""
        result = QEParser.parse_vc_relax(stdout)
        assert result.max_force_ry_bohr == pytest.approx(1.234567e-4)

    def test_kpoint_and_band_count_formats(self):
        stdout = """
     number of k points=           63
     number of Kohn-Sham states=           12
"""
        result = QEParser.parse_scf(stdout)
        assert result.k_points == 63
        assert result.n_bands == 12


class TestParseDOSUnits:
    """dos.x/pdos outputs are in eV — the old ×13.6 inflation bug."""

    def _write(self, tmp_path, header):
        p = tmp_path / "si.dos"
        e = np.linspace(-10, 10, 41)
        d = np.exp(-e**2 / 2.0)
        with open(p, "w") as fh:
            fh.write(header)
            for ei, di in zip(e, d):
                fh.write(f"{ei:12.6f}{di:14.6e}\n")
        return p

    def test_energies_are_ev_passthrough(self, tmp_path):
        p = self._write(tmp_path, "#  E (eV)   dos(E)     EFermi =    6.2124 eV\n")
        r = QEParser.parse_dos(p)
        assert r.energies[0] == pytest.approx(-10.0)
        assert r.energies[-1] == pytest.approx(10.0)
        assert r.fermi_energy_ev == pytest.approx(6.2124)
        assert r.dos_at_fermi is not None

    def test_fermi_none_without_header(self, tmp_path):
        p = self._write(tmp_path, "#  E (eV)   dos(E)\n")
        r = QEParser.parse_dos(p)
        assert r.n_energy_points == 41
        assert r.fermi_energy_ev is None

    def test_corrupt_file_returns_empty(self, tmp_path):
        p = tmp_path / "bad.dos"
        p.write_text("not,a,numeric,file\n")
        r = QEParser.parse_dos(p)
        assert r.n_energy_points == 0


class TestParseXMLEnergy:
    def _write(self, tmp_path, body):
        xml = tmp_path / "out.xml"
        xml.write_text(
            '<?xml version="1.0"?>'
            '<qes:espresso xmlns:qes="http://www.quantum-espresso.org/ns/qes/qes-1.0">'
            f"<output>{body}</output>"
            "</qes:espresso>"
        )
        return xml

    def test_etot_hartree_converted_to_ry(self, tmp_path):
        xml = self._write(tmp_path, "<total_energy><etot>-2.278332e+02</etot></total_energy>")
        r = QEParser.parse_xml(xml)
        assert r["total_energy_ry"] == pytest.approx(-2.278332e2 * 2.0, rel=1e-4)

    def test_prefixed_tags_also_matched(self, tmp_path):
        xml = tmp_path / "prefixed.xml"
        xml.write_text(
            '<?xml version="1.0"?>'
            '<qes:espresso xmlns:qes="http://www.quantum-espresso.org/ns/qes/qes-1.0">'
            "<qes:output><qes:total_energy><qes:etot>-1.0</qes:etot></qes:total_energy></qes:output>"
            "</qes:espresso>"
        )
        r = QEParser.parse_xml(xml)
        assert r["total_energy_ry"] == pytest.approx(-2.0)

    def test_default_namespace_matched(self, tmp_path):
        xml = tmp_path / "defaultns.xml"
        xml.write_text(
            '<?xml version="1.0"?>'
            '<espresso xmlns="http://www.quantum-espresso.org/ns/qes/qes-1.0">'
            "<output><total_energy><etot>-1.0</etot></total_energy></output>"
            "</espresso>"
        )
        r = QEParser.parse_xml(xml)
        assert r["total_energy_ry"] == pytest.approx(-2.0)
    
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
