"""Tests for the material catalog and task specs."""

import pytest
from dft_forge.catalog import (
    get_material_profile,
    get_task_spec,
    list_tasks,
    MATERIALS,
    TASKS,
)


class TestMaterialProfiles:
    def test_si_profile(self):
        profile = get_material_profile("Si")
        assert profile.formula == "Si"
        assert profile.space_group == "Fd-3m"
        assert profile.ibrav == 2
        assert profile.natoms == 2
        assert profile.nspecies == 1
        assert profile.is_metal is False

    def test_al_profile(self):
        profile = get_material_profile("Al")
        assert profile.formula == "Al"
        assert profile.space_group == "Fm-3m"
        assert profile.natoms == 1
        assert profile.is_metal is True

    def test_mgo_profile(self):
        profile = get_material_profile("MgO")
        assert profile.formula == "MgO"
        assert profile.nspecies == 2
        assert set(profile.species) == {"Mg", "O"}

    def test_unknown_material_raises(self):
        with pytest.raises(ValueError, match="Unknown material"):
            get_material_profile("Unknown")


class TestTaskSpecs:
    def test_t1_si_vcrelax(self):
        spec = get_task_spec("T1_Si_vcrelax")
        assert spec.task_id == "T1_Si_vcrelax"
        assert spec.task_type == "T1"
        assert spec.material == "Si"
        assert spec.material_profile is not None
        assert spec.parameters["ecutwfc"] == 25.0

    def test_t2_si_bands(self):
        spec = get_task_spec("T2_Si_bands")
        assert spec.task_id == "T2_Si_bands"
        assert spec.task_type == "T2"
        assert spec.material == "Si"
        assert spec.parameters["subtype"] == "bands"
        assert spec.parameters["kpoints_scf"] == [8, 8, 8, 1, 1, 1]
        assert spec.parameters["nbnd"] == 16
        assert spec.parameters["nkpoints_bands"] == 100

    def test_t2_al_dos(self):
        spec = get_task_spec("T2_Al_dos")
        assert spec.task_id == "T2_Al_dos"
        assert spec.task_type == "T2"
        assert spec.material == "Al"
        assert spec.parameters["subtype"] == "dos"
        assert spec.parameters["dos_deltae"] == 0.01

    def test_t2_mgo_bands(self):
        spec = get_task_spec("T2_MgO_bands")
        assert spec.task_id == "T2_MgO_bands"
        assert spec.task_type == "T2"
        assert spec.material == "MgO"
        assert spec.parameters["nbnd"] == 24

    def test_t2_mgo_dos(self):
        spec = get_task_spec("T2_MgO_dos")
        assert spec.task_id == "T2_MgO_dos"
        assert spec.task_type == "T2"
        assert spec.material == "MgO"
        assert spec.parameters["subtype"] == "dos"

    def test_unknown_task_raises(self):
        with pytest.raises(ValueError, match="Unknown task"):
            get_task_spec("T2_Unknown_bands")


class TestListTasks:
    def test_list_all(self):
        tasks = list_tasks()
        assert "T1_Si_vcrelax" in tasks
        assert "T2_Si_bands" in tasks
        assert "T2_Si_dos" in tasks
        assert len(tasks) >= 9  # 3 T1 + 6 T2

    def test_list_t1_only(self):
        tasks = list_tasks(task_type="T1")
        assert "T1_Si_vcrelax" in tasks
        assert "T1_Al_vcrelax" in tasks
        assert "T1_MgO_vcrelax" in tasks
        assert "T2_Si_bands" not in tasks

    def test_list_t2_only(self):
        tasks = list_tasks(task_type="T2")
        assert "T2_Si_bands" in tasks
        assert "T2_Si_dos" in tasks
        assert "T2_Al_bands" in tasks
        assert "T2_MgO_dos" in tasks
        assert "T1_Si_vcrelax" not in tasks
        assert len(tasks) == 6
