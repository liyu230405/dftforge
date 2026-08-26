"""Tests for structure analysis (bonds/symmetry), PDOS, E_ads and formation energy."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from dft_forge.compiler import build_atoms
from dft_forge.parser import QEParser
from dft_forge.structure_analysis import analyze, coordination_and_bonds, symmetry_info
from dft_forge.structure_builder import build_2d, build_molecule, build_reference


# ── Bond geometry: physics checks against textbook values ─────────────────────


class TestBondGeometry:
    def test_nacl_octahedral(self):
        r = coordination_and_bonds(build_atoms("NaCl"))
        assert r["coordination"]["Na"]["mean"] == 6
        assert r["coordination"]["Cl"]["mean"] == 6
        nacl = r["bond_lengths"]["Cl-Na"]
        assert nacl["count"] == 6
        assert nacl["mean_angstrom"] == pytest.approx(5.64 / 2, abs=0.05)
        # octahedral: 12 right angles + 3 straight angles per center
        ang = r["bond_angles"]["Cl-Na-Cl"]
        assert ang["min_deg"] == pytest.approx(90.0, abs=0.5)
        assert ang["max_deg"] == pytest.approx(180.0, abs=0.5)

    def test_si_tetrahedral(self):
        r = coordination_and_bonds(build_atoms("Si"))
        assert r["coordination"]["Si"]["mean"] == 4
        assert r["bond_lengths"]["Si-Si"]["mean_angstrom"] == pytest.approx(2.35, abs=0.02)
        assert r["bond_angles"]["Si-Si-Si"]["mean_deg"] == pytest.approx(109.47, abs=0.3)

    def test_mos2_trigonal_prismatic(self):
        atoms = build_2d("mos2").atoms
        r = coordination_and_bonds(atoms)
        assert r["coordination"]["Mo"]["mean"] == 6
        assert r["coordination"]["S"]["mean"] == 3
        assert r["bond_lengths"]["Mo-S"]["mean_angstrom"] == pytest.approx(2.41, abs=0.03)

    def test_graphene_honeycomb(self):
        atoms = build_2d("graphene").atoms
        r = coordination_and_bonds(atoms)
        assert r["coordination"]["C"]["mean"] == 3
        assert r["bond_lengths"]["C-C"]["mean_angstrom"] == pytest.approx(1.42, abs=0.02)
        assert r["bond_angles"]["C-C-C"]["mean_deg"] == pytest.approx(120.0, abs=0.5)

    def test_water_molecule(self):
        r = coordination_and_bonds(build_molecule("h2o").atoms)
        assert r["bond_lengths"]["H-O"]["mean_angstrom"] == pytest.approx(0.958, abs=0.01)
        assert r["bond_angles"]["H-O-H"]["mean_deg"] == pytest.approx(104.5, abs=1.0)

    def test_overlapping_atoms_do_not_crash(self):
        atoms = Atoms("H2", positions=[[0, 0, 0], [0.1, 0, 0]], cell=[8, 8, 8], pbc=False)
        r = coordination_and_bonds(atoms)
        assert "bond_analysis_error" in r or "bond_lengths" in r


# ── Symmetry: space group + primitive/conventional conversion ────────────────


class TestSymmetry:
    def test_nacl_space_group(self):
        sym = symmetry_info(build_atoms("NaCl"))
        assert sym["available"] is True
        assert sym["space_group_number"] == 225
        assert sym["space_group"] == "Fm-3m"
        assert sym["conventional_cell"]["natoms"] == 8
        assert sym["primitive_cell"]["natoms"] == 2

    def test_si_space_group(self):
        sym = symmetry_info(build_atoms("Si"))
        assert sym["space_group_number"] == 227
        assert sym["conventional_cell"]["natoms"] == 8
        assert sym["primitive_cell"]["natoms"] == 2

    def test_mos2_2d_space_group(self):
        sym = symmetry_info(build_2d("mos2").atoms)
        assert sym["available"] is True
        assert sym["space_group_number"] in (187, 164)  # P-6m2 monolayer / P63/mmc bulk

    def test_analyze_report_shape(self):
        r = analyze(build_atoms("NaCl"))
        assert r["formula"] == "ClNa"  # ASE orders alphabetically
        assert r["symmetry"]["available"]
        assert r["bond_lengths"]["Cl-Na"]["count"] == 6
        assert r["is_2d"] is False


# ── extract_final_structure: vc-relax stdout → ASE Atoms ─────────────────────

_CELL_ALAT = "CELL_PARAMETERS (alat= 10.262)\n 0.0 0.5 0.5\n 0.5 0.0 0.5\n 0.5 0.5 0.0\n"


class TestExtractFinalStructure:
    def test_crystal_positions(self):
        stdout = (
            "Begin final coordinates\n"
            "CELL_PARAMETERS (alat= 10.262)\n"
            " -0.0183 0.5 0.5\n 0.5 -0.0183 0.5\n 0.5 0.5 -0.0183\n"
            "ATOMIC_POSITIONS (crystal)\n"
            " Si 0.0 0.0 0.0\n"
            " Si 0.25 0.25 0.25\n"
            "End final coordinates\n"
        )
        atoms = QEParser.extract_final_structure(stdout)
        assert atoms is not None
        assert len(atoms) == 2
        d = atoms.get_distance(0, 1)
        assert 2.2 < d < 2.5  # Å, not bohr (4.4) or collapsed

    def test_alat_positions(self):
        stdout = (
            "CELL_PARAMETERS (alat= 10.262)\n"
            " 0.0 0.5 0.5\n 0.5 0.0 0.5\n 0.5 0.5 0.0\n"
            "ATOMIC_POSITIONS (alat)\n"
            " Si 0.0 0.0 0.0\n"
            " Si 0.25 0.25 0.25\n"
        )
        atoms = QEParser.extract_final_structure(stdout)
        assert atoms is not None
        assert 2.2 < atoms.get_distance(0, 1) < 2.5

    def test_angstrom_positions(self):
        stdout = (
            "CELL_PARAMETERS (alat= 10.262)\n"
            " 0.0 0.5 0.5\n 0.5 0.0 0.5\n 0.5 0.5 0.0\n"
            "ATOMIC_POSITIONS (angstrom)\n"
            " Si 0.0 0.0 0.0\n"
            " Si 1.358 1.358 1.358\n"
        )
        atoms = QEParser.extract_final_structure(stdout)
        assert atoms is not None
        assert 2.2 < atoms.get_distance(0, 1) < 2.5

    def test_no_final_block(self):
        assert QEParser.extract_final_structure("JOB DONE.") is None


# ── parse_pdos: per-atom pdos files → element-resolved curves ────────────────


class TestParsePdos:
    @staticmethod
    def _write(workdir, prefix, n_points=5):
        # fake mgo nscf XML: species + fermi
        xml = workdir / f"{prefix}.save" / "data-file-schema.xml"
        xml.parent.mkdir(parents=True, exist_ok=True)
        xml.write_text(
            "<qes:espresso xmlns:qes='http://www.quantum-espresso.org/ns/qes/qes-1.0'>"
            "<input><atomic_structure><atomic_positions>"
            "<atom name='Mg' index='1'/><atom name='O' index='2'/>"
            "</atomic_positions></atomic_structure></input>"
            "<output><fermi_energy>0.2</fermi_energy></output>"
            "</qes:espresso>"
        )
        e = np.linspace(-1.0, 1.0, n_points)  # Ry
        np.savetxt(workdir / f"{prefix}.pdos_atm#1(Mg)", np.c_[e, np.full(n_points, 2.0)])
        np.savetxt(workdir / f"{prefix}.pdos_atm#2(O)", np.c_[e, np.full(n_points, 1.0)])

    def test_element_summation(self, tmp_path):
        self._write(tmp_path, "mgo")
        parsed = QEParser.parse_pdos(tmp_path, "mgo")
        assert parsed.element_dos is not None
        assert set(parsed.element_dos) == {"Mg", "O"}
        assert parsed.n_energy_points == 5
        assert parsed.element_dos["Mg"][0] == pytest.approx(2.0)
        assert parsed.element_dos["O"][0] == pytest.approx(1.0)
        assert parsed.energies[0] == pytest.approx(-1.0 * 13.6057, abs=0.01)  # Ry→eV

    def test_fermi_from_xml(self, tmp_path):
        self._write(tmp_path, "mgo")
        parsed = QEParser.parse_pdos(tmp_path, "mgo")
        assert parsed.fermi_energy_ev == pytest.approx(0.2 * 27.2114, abs=0.01)

    def test_no_files(self, tmp_path):
        parsed = QEParser.parse_pdos(tmp_path, "none")
        assert parsed.element_dos is None


# ── Molecule + reference phase builders ──────────────────────────────────────


class TestBuilders:
    def test_molecule_pbc_false(self):
        for kind in ("o2", "h2o", "co2", "nh3"):
            res = build_molecule(kind)
            assert not any(res.atoms.pbc), kind
            assert res.atoms.cell.lengths()[0] >= 10.0

    def test_o2_bond_length(self):
        r = coordination_and_bonds(build_molecule("o2").atoms)
        assert r["bond_lengths"]["O-O"]["mean_angstrom"] == pytest.approx(1.21, abs=0.02)

    def test_reference_si_diamond(self):
        res = build_reference("Si")
        assert len(res.atoms) == 2
        sym = symmetry_info(res.atoms)
        assert sym["space_group_number"] == 227

    def test_reference_o2_molecule(self):
        res = build_reference("O")
        assert res.atoms.get_chemical_formula() == "O2"

    def test_reference_unknown_element(self):
        from dft_forge.structure_builder import StructureBuildError

        with pytest.raises(StructureBuildError):
            build_reference("Xe")


# ── Thermochemistry CLI math ──────────────────────────────────────────────────


class TestThermoCli:
    def test_eads_math(self, capsys):
        from dft_forge.cli import cmd_thermo_eads
        import argparse

        # no atom counts → plain E_a − E_s − E_m
        cmd_thermo_eads(argparse.Namespace(ry_a=-158.0, ry_s=-156.5, ry_m=-1.2, output=None,
                                           nat_a=None, nat_mol=None))
        import json

        data = json.loads(capsys.readouterr().out)
        assert data["ok"] is True
        assert data["e_ads_ev"] == pytest.approx(-0.3 * 13.6057, abs=0.01)
        assert data["e_ads_ry"] == pytest.approx(-0.3)

    def test_eads_o2_half_normalization(self, capsys):
        from dft_forge.cli import cmd_thermo_eads
        import argparse
        import json

        # 1 O atom adsorbed, reference gas O2 → subtract E(O2)/2
        # E_ads = -238.8 + 206.8 + 64.2/2 = +0.1 Ry
        cmd_thermo_eads(argparse.Namespace(ry_a=-238.8, ry_s=-206.8, ry_m=-64.2, output=None,
                                           nat_a=1, nat_mol=2))
        data = json.loads(capsys.readouterr().out)
        assert data["ok"] is True
        assert data["e_ads_ry"] == pytest.approx(0.1, abs=0.001)
        assert data["molecule_ratio"] == pytest.approx(0.5)
        assert data["e_ads_ev_per_adsorbate"] == pytest.approx(0.1 * 13.6057, abs=0.01)

    def test_formation_math(self, capsys):
        from dft_forge.cli import cmd_thermo_formation
        import argparse
        import json

        # NaCl: E=-40 Ry; refs: Na(bcc, 2 atoms)=-16 Ry, Cl2=-30 Ry
        cmd_thermo_formation(
            argparse.Namespace(
                formula="NaCl",
                compound_energy=-40.0,
                ref_element=["Na", "Cl"],
                ref_energy=["-16.0", "-30.0"],
                ref_natoms=["2", "2"],
                output=None,
            )
        )
        data = json.loads(capsys.readouterr().out)
        assert data["ok"] is True
        # E_form = E(NaCl) - E(Na)/2 - E(Cl2)/2 = -40 + 8 + 15 = -17 Ry
        assert data["formation_energy_ev_per_formula"] == pytest.approx(-17.0 * 13.6057, abs=0.01)
        assert data["formation_energy_ev_per_atom"] == pytest.approx(-17.0 * 13.6057 / 2, abs=0.01)


# ── Agent planning: E_ads / formation workflows ───────────────────────────────


class TestThermoPlanning:
    def test_eads_full_workflow(self):
        from dft_forge.agent_loop import AgentLoop

        steps = AgentLoop().plan("O吸附在石墨烯的吸附能")
        tools = [s["tool"] for s in steps]
        assert tools == [
            "structure.build2d", "graph.run",
            "structure.build2d", "graph.run",
            "structure.molecule", "graph.run",
            "thermo.eads",
        ]
        assert steps[0]["args"]["adsorb"]["element"] == "O"
        assert "adsorb" not in steps[2]["args"]
        assert steps[4]["args"]["kind"] == "o2"
        assert steps[6]["args"] == {}  # energies auto-injected at run time

    def test_formation_workflow(self):
        from dft_forge.agent_loop import AgentLoop

        steps = AgentLoop().plan("NaCl的形成能")
        tools = [s["tool"] for s in steps]
        assert tools == [
            "graph.run",
            "structure.reference", "graph.run",
            "structure.reference", "graph.run",
            "thermo.formation",
        ]
        assert steps[0]["args"]["inputs"]["material"] == "NaCl"
        assert {s["args"]["element"] for s in steps if s["tool"] == "structure.reference"} == {"Na", "Cl"}

    def test_thermo_tools_registered(self):
        from dft_forge.agent_loop import AgentLoop

        ids = {t["id"] for t in AgentLoop().list_tools()}
        assert {"structure.molecule", "structure.reference", "thermo.eads", "thermo.formation"} <= ids
