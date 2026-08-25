"""Tests for the MP-verified material library and GBRV pseudo resolution."""

from pathlib import Path

import numpy as np
import pytest
from ase.formula import Formula

from dft_forge.catalog import (
    build_library_task_spec,
    get_material_profile,
    list_materials,
)
from dft_forge.catalog.library import LIBRARY_MATERIALS
from dft_forge.compiler import (
    MATERIAL_DB,
    QECompiler,
    build_atoms,
    get_kpoints,
    get_lattice_constant_angstrom,
)

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "pseudos"


class TestLibraryData:
    def test_library_size(self):
        assert len(LIBRARY_MATERIALS) >= 50

    def test_builtin_materials_not_duplicated(self):
        for key in ("Si", "Al", "MgO"):
            assert key not in LIBRARY_MATERIALS
            assert key in MATERIAL_DB

    def test_entries_have_required_fields(self):
        for name, db in LIBRARY_MATERIALS.items():
            for field in (
                "formula", "structure_type", "cellpar", "sites", "species",
                "masses", "pseudos", "ecutwfc_default", "ecutrho_default",
                "kpoints_default", "is_metal",
            ):
                assert field in db, f"{name} missing {field}"
            assert len(db["cellpar"]) == 6
            assert db["nspecies"] == len(set(db["species"]))

    def test_all_pseudos_exist_on_disk(self):
        missing = []
        for name, db in LIBRARY_MATERIALS.items():
            for element, pseudo in db["pseudos"].items():
                if not (ASSETS / pseudo).exists():
                    missing.append(f"{name}/{element}: {pseudo}")
        assert missing == []


class TestBuildAtomsLibrary:
    def test_zincblende_gaas(self):
        atoms = build_atoms("GaAs")
        assert len(atoms) == 2
        assert Formula(atoms.get_chemical_formula()) == Formula("GaAs")

    def test_rocksalt_nacl(self):
        atoms = build_atoms("NaCl")
        assert Formula(atoms.get_chemical_formula()) == Formula("NaCl")
        # Rocksalt primitive cell has 2 atoms
        assert len(atoms) == 2

    def test_wurtzite_aln_cell_and_stoichiometry(self):
        atoms = build_atoms("AlN")
        assert len(atoms) == 4
        assert Formula(atoms.get_chemical_formula()).reduce()[0] == Formula("AlN").reduce()[0]
        db = LIBRARY_MATERIALS["AlN"]
        a_ang, c_ang = db["cellpar"][0], db["cellpar"][2]
        # |a1| in Bohr should match a * conversion
        lengths = np.linalg.norm(atoms.cell.array, axis=1)
        assert lengths.max() == pytest.approx(max(a_ang, c_ang) * 1.8897261246, rel=1e-4)

    def test_bcc_li(self):
        atoms = build_atoms("Li")
        assert len(atoms) == 1

    def test_every_library_material_builds(self):
        for name in LIBRARY_MATERIALS:
            atoms = build_atoms(name)
            assert len(atoms) >= 1

    def test_lattice_constant(self):
        db = LIBRARY_MATERIALS["GaAs"]
        assert get_lattice_constant_angstrom("GaAs") == db["cellpar"][0]

    def test_builtin_lattice_constant_unchanged(self):
        assert get_lattice_constant_angstrom("Si") == 5.43


class TestDefaults:
    def test_metal_kpoints(self):
        assert tuple(get_kpoints("Cu")) == (8, 8, 8, 1, 1, 1)

    def test_hexagonal_kpoints(self):
        assert tuple(get_kpoints("AlN")) == (6, 6, 4, 1, 1, 1)

    def test_hard_elements_get_higher_cutoff(self):
        assert LIBRARY_MATERIALS["LiF"]["ecutwfc_default"] >= 60.0


class TestCatalogIntegration:
    def test_get_material_profile_library(self):
        profile = get_material_profile("GaAs")
        assert profile.formula == "GaAs"
        assert profile.species == ["Ga", "As"]
        assert "Ga.upf" in profile.pseudos.values()

    def test_list_materials_includes_library(self):
        materials = list_materials()
        assert "GaAs" in materials
        assert "NaCl" in materials
        assert "Si" in materials
        assert len(materials) >= 55

    def test_build_library_task_spec_t1(self):
        spec = build_library_task_spec("GaAs", task_type="T1")
        assert spec["task_id"] == "T1_GaAs_vcrelax"
        assert spec["material"] == "GaAs"
        params = spec["parameters"]
        assert params["ecutwfc"] == LIBRARY_MATERIALS["GaAs"]["ecutwfc_default"]
        assert len(params["kpoints"]) == 6

    def test_build_library_task_spec_t2_bands_and_dos(self):
        bands = build_library_task_spec("GaAs", task_type="T2", subtype="bands")
        assert bands["task_id"] == "T2_GaAs_bands"
        assert bands["parameters"]["nkpoints_bands"] == 100
        dos = build_library_task_spec("GaAs", task_type="T2", subtype="dos")
        assert dos["parameters"]["dos_deltae"] == 0.01

    def test_unknown_material_raises(self):
        with pytest.raises(ValueError):
            get_material_profile("Unobtainium")
        with pytest.raises(ValueError):
            build_library_task_spec("Unobtainium")


class TestCompileLibraryMaterial:
    def test_compile_t1_gaas(self, tmp_path):
        compiler = QECompiler(pseudo_dir=ASSETS)
        out = tmp_path / "gaas.in"
        content = compiler.compile_t1("GaAs", out)
        assert out.exists()
        assert "Ga.upf" in content
        assert "As.upf" in content
        assert "CELL_PARAMETERS" in content
        assert "ibrav = 0" in content
        assert "nat = 2" in content

    def test_compile_scf_aln(self, tmp_path):
        compiler = QECompiler(pseudo_dir=ASSETS)
        out = tmp_path / "aln.in"
        content = compiler.compile_scf("AlN", out)
        assert "nat = 4" in content
        assert "Al.upf" in content
        # Cell vectors expressed in alat units should have magnitude ~1 for hex a
        assert "CELL_PARAMETERS" in content
