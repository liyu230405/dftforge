"""Core data structures for the DFT-Forge protocol layer.

All LLM-facing outputs must be validated against these schemas.
No free-form shell or full QE input from the LLM.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
from enum import Enum


class TaskType(str, Enum):
    T1 = "T1"      # Structure optimization (vc-relax)
    T2 = "T2"      # Band structure + DOS


class CalculationType(str, Enum):
    SCF = "scf"
    VC_RELAX = "vc-relax"
    RELAX = "relax"
    BANDS_NSCF = "bands_nscf"
    DOS_NSCF = "dos_nscf"
    NEB = "neb"


class Verdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"
    INCOMPLETE = "incomplete"


# ── Material profile ──────────────────────────────────────────────────────────

@dataclass
class MaterialProfile:
    """Deterministic description of a material's structure and pseudopotentials."""
    formula: str                          # e.g. "Si", "Al", "MgO"
    structure_type: str                   # e.g. "diamond", "fcc", "nacl"
    space_group: str                      # e.g. "Fd-3m"
    ibrav: int                            # QE bravais-lattice index
    lattice_constant_bohr: float          # Initial guess in Bohr
    natoms: int                           # Atoms per primitive cell
    nspecies: int                         # Number of distinct species
    species: List[str]                    # Element symbols in order
    masses: List[float]                   # Atomic masses in amu
    pseudos: Dict[str, str]               # element -> pseudo filename (in pseudo_dir)
    occupations: str = "smearing"
    smearing: str = "mp"
    degauss: float = 0.005
    charge: int = 0
    is_metal: bool = False                # Metals need different smearing/nbnd

    def to_dict(self) -> dict:
        return asdict(self)


# ── Workflow IR (what the LLM outputs) ────────────────────────────────────────

@dataclass
class StepIR:
    """A single step in a workflow plan."""
    step_id: str
    step_type: str                         # "scf" | "vc-relax" | "bands" | "dos"
    prefix: str
    outdir: str = "./"                     # Always relative to workdir
    depends_on: List[str] = field(default_factory=list)
    overrides: Dict[str, Any] = field(default_factory=dict)
    # overrides can change: ecutwfc, kpoints, nbnd, occupations, etc.

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class WorkflowIR:
    """Structured workflow plan output by the LLM.
    
    The LLM may ONLY output WorkflowIR or PlanPatch.  No shell commands.
    """
    task_id: str
    task_type: str                         # "T1" | "T2"
    material: str
    steps: List[StepIR] = field(default_factory=list)
    reasoning: str = ""

    def validate(self) -> List[str]:
        """Return list of validation errors (empty = valid)."""
        errors = []
        if self.task_type not in ("T1", "T2"):
            errors.append(f"Invalid task_type: {self.task_type}")
        if not self.steps:
            errors.append("WorkflowIR has no steps")
        for step in self.steps:
            if step.step_type not in ("scf", "vc-relax", "relax", "bands_nscf", "dos_nscf"):
                errors.append(f"Invalid step_type: {step.step_type}")
            # Prefix must be plain ASCII, no spaces
            if not step.prefix.replace("_", "").replace("-", "").isalnum():
                errors.append(f"Invalid prefix: {step.prefix}")
        return errors

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PlanPatch:
    """A modification to an existing WorkflowIR, used for recovery."""
    target_step: str                       # step_id to patch
    action: str                            # "modify_params" | "change_kpoints" | "increase_nbnd" | "retry"
    new_params: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""

    def validate(self) -> List[str]:
        errors = []
        if self.action not in ("modify_params", "change_kpoints", "increase_nbnd", "retry", "restart_from_prev"):
            errors.append(f"Invalid patch action: {self.action}")
        return errors

    def to_dict(self) -> dict:
        return asdict(self)


# ── Verification result ───────────────────────────────────────────────────────

@dataclass
class VerificationInput:
    """Unified, typed input for the Scientific Verifier.

    Replaces duck-typed job-result objects (and the CLI-side FakeJobResult):
    one shape for real executions, CLI replay of result JSON, and tests.
    """
    stdout: str = ""
    stderr: str = ""
    xml_path: Optional[Path] = None
    job_success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationResult:
    """Deterministic verdict from the Scientific Verifier."""
    passed: bool
    checks: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    failure_reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    summary: str = ""
    verifier_version: str = "0.1.0"

    def to_dict(self) -> dict:
        return asdict(self)


# ── Evidence bundle ───────────────────────────────────────────────────────────

@dataclass
class EvidenceBundle:
    """Complete evidence record for one task execution."""
    task_id: str
    task_type: str
    status: str                             # "pass" | "fail" | "incomplete"
    input_hash: str = ""
    pseudo_hashes: Dict[str, str] = field(default_factory=dict)
    qe_version: str = ""
    parser_version: str = "0.1.0"
    commands: List[str] = field(default_factory=list)
    walltimes_sec: List[float] = field(default_factory=list)
    total_walltime_sec: float = 0.0
    tokens_used: int = 0
    errors: List[str] = field(default_factory=list)
    recovery_actions: List[str] = field(default_factory=list)
    verifier_verdict: Optional[Dict] = None
    physical_results: Dict[str, Any] = field(default_factory=dict)
    output_files: List[str] = field(default_factory=list)
    log_files: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))


# ── Task spec (input to the agent) ────────────────────────────────────────────

@dataclass
class TaskSpec:
    """Structured specification of a DFT task."""
    task_id: str
    task_type: str
    material: str
    description: str = ""
    material_profile: Optional[MaterialProfile] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    prompt_file: Optional[str] = None       # Path to task_prompt.txt (read-only)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        if d.get("material_profile"):
            d["material_profile"] = d["material_profile"].to_dict()
        return d
