"""Tests for the protocol layer: schemas, validation."""

import pytest
from dft_forge.protocol.schemas import (
    MaterialProfile,
    StepIR,
    WorkflowIR,
    PlanPatch,
    VerificationResult,
    EvidenceBundle,
    TaskSpec,
    TaskType,
    CalculationType,
    Verdict,
)


class TestMaterialProfile:
    def test_create_si(self):
        mp = MaterialProfile(
            formula="Si",
            structure_type="diamond",
            space_group="Fd-3m",
            ibrav=2,
            lattice_constant_bohr=10.26,
            natoms=2,
            nspecies=1,
            species=["Si"],
            masses=[28.086],
            pseudos={"Si": "Si_r.upf"},
        )
        assert mp.formula == "Si"
        assert mp.is_metal is False
    
    def test_create_al(self):
        mp = MaterialProfile(
            formula="Al",
            structure_type="fcc",
            space_group="Fm-3m",
            ibrav=2,
            lattice_constant_bohr=7.63,
            natoms=1,
            nspecies=1,
            species=["Al"],
            masses=[26.982],
            pseudos={"Al": "Al.pbe-n-rrkjus_psl.1.0.2.UPF"},
            is_metal=True,
            degauss=0.02,
        )
        assert mp.is_metal is True
        assert mp.degauss == 0.02


class TestWorkflowIR:
    def test_valid_workflow(self):
        wf = WorkflowIR(
            task_id="T1_Si_vcrelax",
            task_type="T1",
            material="Si",
            steps=[
                StepIR(step_id="step1", step_type="vc-relax", prefix="si"),
            ],
        )
        errors = wf.validate()
        assert errors == []
    
    def test_invalid_task_type(self):
        wf = WorkflowIR(
            task_id="T1_Si_vcrelax",
            task_type="T5",  # Invalid
            material="Si",
            steps=[StepIR(step_id="step1", step_type="vc-relax", prefix="si")],
        )
        errors = wf.validate()
        assert any("task_type" in e for e in errors)
    
    def test_invalid_prefix(self):
        wf = WorkflowIR(
            task_id="T1_Si_vcrelax",
            task_type="T1",
            material="Si",
            steps=[StepIR(step_id="step1", step_type="vc-relax", prefix="si test")],
        )
        errors = wf.validate()
        assert any("prefix" in e for e in errors)
    
    def test_no_steps(self):
        wf = WorkflowIR(
            task_id="T1_Si_vcrelax",
            task_type="T1",
            material="Si",
            steps=[],
        )
        errors = wf.validate()
        assert any("no steps" in e.lower() for e in errors)


class TestWorkflowIRT2:
    """Tests for T2 workflow validation."""

    def test_valid_bands_workflow(self):
        wf = WorkflowIR(
            task_id="T2_Si_bands",
            task_type="T2",
            material="Si",
            steps=[
                StepIR(step_id="scf", step_type="scf", prefix="si"),
                StepIR(step_id="nscf", step_type="bands_nscf", prefix="si"),
                StepIR(step_id="bands", step_type="bands_nscf", prefix="si"),
            ],
        )
        errors = wf.validate()
        assert errors == []

    def test_valid_dos_workflow(self):
        wf = WorkflowIR(
            task_id="T2_Si_dos",
            task_type="T2",
            material="Si",
            steps=[
                StepIR(step_id="scf", step_type="scf", prefix="si"),
                StepIR(step_id="nscf", step_type="dos_nscf", prefix="si"),
                StepIR(step_id="dos", step_type="dos_nscf", prefix="si"),
            ],
        )
        errors = wf.validate()
        assert errors == []

    def test_t2_invalid_step_type(self):
        wf = WorkflowIR(
            task_id="T2_Si_bands",
            task_type="T2",
            material="Si",
            steps=[
                StepIR(step_id="bad", step_type="invalid_step", prefix="si"),
            ],
        )
        errors = wf.validate()
        assert any("step_type" in e for e in errors)


class TestPlanPatch:
    def test_valid_patch(self):
        patch = PlanPatch(
            target_step="step1",
            action="modify_params",
            new_params={"ecutwfc": 30.0},
        )
        errors = patch.validate()
        assert errors == []
    
    def test_invalid_action(self):
        patch = PlanPatch(
            target_step="step1",
            action="delete_files",  # Invalid action
        )
        errors = patch.validate()
        assert any("action" in e for e in errors)


class TestEvidenceBundle:
    def test_to_json(self, tmp_path):
        bundle = EvidenceBundle(
            task_id="T1_Si_vcrelax",
            task_type="T1",
            status="pass",
            input_hash="abc123",
            physical_results={"final_energy_ry": -7.8},
        )
        out_file = tmp_path / "evidence.json"
        bundle.to_json(out_file)
        
        import json
        data = json.loads(out_file.read_text())
        assert data["task_id"] == "T1_Si_vcrelax"
        assert data["status"] == "pass"
        assert data["physical_results"]["final_energy_ry"] == -7.8
