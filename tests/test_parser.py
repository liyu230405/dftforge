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


class TestParseMagnetization:
    def test_legacy_atom_magnetization_format(self):
        stdout = """
     atom:    1    charge:  7.0    magnetization:  2.10
     atom:    2    charge:  7.0    magnetization: -2.10
     total magnetization       =     0.0000 Bohr mag/cell
     absolute magnetization    =     4.2000 Bohr mag/cell
"""
        result = QEParser.parse_magnetization(stdout)
        assert result.total_magnetization_muB == pytest.approx(0.0)
        assert result.absolute_magnetization_muB == pytest.approx(4.2)
        assert result.atom_magnetizations_muB == pytest.approx([2.1, -2.1])

    def test_qe75_uses_last_complete_per_site_block(self):
        stdout = """
     number of atoms/cell      =  3
     Magnetic moment per site  (integrated on atomic sphere of radius R)
     atom   1 (R=0.357)  charge= 14.5422  magn=  1.1000
     atom   2 (R=0.357)  charge= 14.5422  magn= -1.1000
     atom   3 (R=0.357)  charge= 14.5422  magn=  0.0000

     Magnetic moment per site  (integrated on atomic sphere of radius R)
     atom   1 (R=0.357)  charge= 14.5422  magn=  2.7720
     atom   2 (R=0.357)  charge= 14.5422  magn= -2.7719
     atom   3 (R=0.357)  charge= 14.5422  magn= -0.0001

     Magnetic moment per site  (truncated output)
     atom   1 (R=0.357)  charge= 14.5422  magn=  9.0000
     atom   2 (R=0.357)  charge= 14.5422  magn= -9.0000
"""
        result = QEParser.parse_magnetization(stdout)
        assert result.atom_magnetizations_muB == pytest.approx([2.7720, -2.7719, -0.0001])


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

    def test_qe_xml_cell_vectors_are_absolute_bohr_not_alat_scaled(self, tmp_path):
        xml = self._write(
            tmp_path,
            '<atomic_structure nat="2" alat="4.0" bravais_index="0">'
            '<cell><a1>4.0 0.0 0.0</a1><a2>0.0 5.0 0.0</a2>'
            '<a3>0.0 0.0 6.0</a3></cell></atomic_structure>',
        )

        result = QEParser.parse_xml(xml)

        assert result["cell"].a_bohr == pytest.approx(4.0)
        assert result["cell"].b_bohr == pytest.approx(5.0)
        assert result["cell"].c_bohr == pytest.approx(6.0)
        assert result["cell"].volume_bohr3 == pytest.approx(120.0)

    def test_fixed_cell_structure_preserves_nonperiodic_axes(self):
        stdout = """
Begin final coordinates
ATOMIC_POSITIONS (crystal)
C 0.0 0.0 0.5
C 0.5 0.5 0.5
End final coordinates
"""

        atoms = QEParser.extract_final_structure(
            stdout,
            fixed_cell=np.diag([2.5, 2.5, 12.0]),
            pbc=[True, True, False],
        )

        assert atoms is not None
        assert atoms.pbc.tolist() == [True, True, False]
        assert atoms.cell.lengths() == pytest.approx([2.5, 2.5, 12.0])
    
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


# ── Bands XML Fermi / gap semantics ──────────────────────────────────────────


def _bands_xml(tmp_path, *, fermi_ha=None, degauss_ha=None, evals_per_k=((-0.4, -0.3), (-0.1, 0.2))):
    smearing = (
        f'<qes:smearing degauss="{degauss_ha}">mv</qes:smearing>' if degauss_ha is not None else ""
    )
    fermi = f"<qes:fermi_energy>{fermi_ha}</qes:fermi_energy>" if fermi_ha is not None else ""
    ks = "".join(
        "<qes:ks_energies><qes:eigenvalues>" + " ".join(str(v) for v in evals) + "</qes:eigenvalues></qes:ks_energies>"
        for evals in evals_per_k
    )
    p = tmp_path / "bands.xml"
    p.write_text(
        '<?xml version="1.0"?><qes:espresso xmlns:qes="http://www.quantum-espresso.org/ns/qes/qes-1.0">'
        f'<qes:output><qes:band_structure nks="2" nbnd="2">{ks}{smearing}</qes:band_structure>'
        f"{fermi}</qes:output></qes:espresso>"
    )
    return p


class TestParseBandsFermi:
    def test_missing_fermi_leaves_gap_unknown(self, tmp_path):
        # Regression: parser used to set fermi = max(eigenvalue)/2, which
        # produced plausible-looking but fabricated VBM/CBM/gap numbers.
        xml = _bands_xml(tmp_path, fermi_ha=None, evals_per_k=((-0.4, 0.3), (-0.2, 0.4)))
        r = QEParser.parse_bands(xml)
        assert r.fermi_energy_ev is None
        assert r.band_gap_ev is None
        assert r.gap_known is False

    def test_smearing_floor_from_xml(self, tmp_path):
        from dft_forge.parser import HA_TO_EV

        # fermi at 0.0 Ha; occupied up to -0.002 Ha, cbm at +0.003 Ha
        xml = _bands_xml(
            tmp_path,
            fermi_ha=0.0,
            degauss_ha=0.005,  # floor = 2*0.005*27.21 ≈ 0.272 eV
            evals_per_k=((-0.01, 0.003), (-0.002, 0.004)),
        )
        r = QEParser.parse_bands(xml)
        assert r.smearing_resolution_ev == pytest.approx(2 * 0.005 * HA_TO_EV)
        # path gap = 0.005 Ha ≈ 0.136 eV < 0.272 → effectively metallic
        assert r.band_gap_ev == pytest.approx(0.005 * HA_TO_EV)
        assert r.is_metal is True
        assert r.gap_known is True

    def test_un_smeared_gap_is_semiconductor(self, tmp_path):
        # No smearing element: any positive path gap is reported as a gap.
        # fermi 0; vbm = -0.2 Ha, cbm = +0.05 Ha → gap 0.25 Ha.
        xml = _bands_xml(
            tmp_path,
            fermi_ha=0.0,
            degauss_ha=None,
            evals_per_k=((-0.3, 0.05), (-0.2, 0.05)),
        )
        r = QEParser.parse_bands(xml)
        assert r.smearing_resolution_ev is None
        assert r.band_gap_ev == pytest.approx(0.25 * 27.211386245988)
        assert r.is_metal is False
