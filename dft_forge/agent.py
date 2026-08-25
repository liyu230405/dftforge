"""Full agent solve loop.

Orchestrates:
  prompt -> planner -> compiler -> executor -> parser -> verifier -> recovery -> ledger

The agent loop is bounded: each stage may retry a limited number of times
using safe recovery actions.  Actions that would change the physical model
must be surfaced to the user instead of being applied automatically.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from dft_forge.catalog import get_task_spec, list_materials, list_tasks
from dft_forge.compiler import QECompiler, build_atoms
from dft_forge.executor import Executor, FakeExecutor, LocalExecutor
from dft_forge.ledger import EvidenceLedger
from dft_forge.llm import LLMProvider
from dft_forge.planner import Planner
from dft_forge.protocol.schemas import EvidenceBundle, WorkflowIR
from dft_forge.recovery import RecoveryAction, RecoveryController
from dft_forge.runner import TaskRunner, run_task_deterministic


@dataclass
class SolveResult:
    status: str
    task_id: str
    workflow: Optional[WorkflowIR]
    result: Optional[Dict[str, Any]]
    evidence: Optional[Dict[str, Any]]
    recovery_actions: List[str]
    ledger_rows: int
    error: Optional[str] = None


class Agent:
    """High-level scientific computing agent."""

    def __init__(
        self,
        planner: Optional[Planner] = None,
        executor: Optional[Executor] = None,
        ledger: Optional[EvidenceLedger] = None,
        max_attempts: int = 3,
    ):
        self.planner = planner or Planner()
        self.executor = executor or LocalExecutor(qe_bin_dir=Path("/opt/homebrew/bin"))
        self.ledger = ledger
        self.max_attempts = max_attempts
        self.recovery = RecoveryController(max_attempts=max_attempts)

    def solve_from_prompt(
        self,
        prompt: str,
        workdir: Path,
        *,
        out: Optional[Path] = None,
        task_id: Optional[str] = None,
    ) -> SolveResult:
        workdir = Path(workdir).resolve()
        workdir.mkdir(parents=True, exist_ok=True)

        if task_id:
            return self._run_known_task(task_id, workdir, out=out)

        try:
            workflow = self.planner.plan_from_prompt(
                prompt,
                available_materials=list_materials(),
                available_calcs=["scf", "vc-relax", "bands", "dos"],
            )
        except (RuntimeError, ValueError) as exc:
            return SolveResult(
                status="unsupported",
                task_id="unplanned",
                workflow=None,
                result=None,
                evidence=None,
                recovery_actions=[],
                ledger_rows=0,
                error=str(exc),
            )
        return self._run_workflow(workflow, workdir, out=out)

    def _run_known_task(self, task_id: str, workdir: Path, *, out: Optional[Path]) -> SolveResult:
        try:
            result = run_task_deterministic(
                task_id,
                workdir,
                executor=self.executor,
            )
        except Exception as exc:
            return SolveResult(
                status="error",
                task_id=task_id,
                workflow=None,
                result=None,
                evidence=None,
                recovery_actions=[],
                ledger_rows=0,
                error=str(exc),
            )

        evidence_path = workdir / "evidence.json"
        evidence = json.loads(evidence_path.read_text()) if evidence_path.exists() else None
        self._maybe_record_ledger(task_id, result)

        if out:
            out = Path(out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(result, indent=2, default=str))

        return SolveResult(
            status=result.get("status", "unknown"),
            task_id=task_id,
            workflow=None,
            result=result,
            evidence=evidence,
            recovery_actions=result.get("failure_reasons", []),
            ledger_rows=1 if self.ledger else 0,
            error=result.get("verifier_summary") if result.get("status") != "pass" else None,
        )

    def _run_workflow(self, workflow: WorkflowIR, workdir: Path, *, out: Optional[Path]) -> SolveResult:
        task_id = workflow.task_id or "unknown"
        try:
            get_task_spec(task_id)
        except ValueError:
            registered = self._register_library_task(workflow)
            if registered is None:
                return SolveResult(
                    status="unsupported",
                    task_id=task_id,
                    workflow=workflow,
                    result=None,
                    evidence=None,
                    recovery_actions=[],
                    ledger_rows=0,
                    error=f"Material '{workflow.material}' is not in the material library",
                )
            task_id = registered

        return self._run_known_task(task_id, workdir, out=out)

    @staticmethod
    def _register_library_task(workflow: WorkflowIR) -> Optional[str]:
        """Register a library-material workflow as a runnable task, return its id."""
        from dft_forge.catalog import MATERIALS, TASKS, build_library_task_spec
        from dft_forge.catalog.library import LIBRARY_MATERIALS

        material = workflow.material
        if material not in MATERIALS and material not in LIBRARY_MATERIALS:
            return None
        subtype = None
        for step in workflow.steps:
            if step.step_type == "bands_nscf":
                subtype = "bands"
                break
            if step.step_type == "dos_nscf":
                subtype = "dos"
                break
        spec = build_library_task_spec(material, task_type=workflow.task_type, subtype=subtype)
        TASKS[spec["task_id"]] = spec
        return spec["task_id"]

    def _maybe_record_ledger(self, task_id: str, result: Dict[str, Any]) -> None:
        if not self.ledger:
            return
        self.ledger.record_run(
            task_id=task_id,
            task_type=result.get("task_type", "unknown"),
            attempt=1,
            success=result.get("status") == "pass",
            input_hash=result.get("input_hash", ""),
            walltime_sec=float(result.get("walltime_sec", 0.0)),
            verifier_passed=bool(result.get("verifier_passed", False)),
            failure_reasons=result.get("failure_reasons", []),
            recovery_actions=result.get("recovery_actions", []),
            evidence_path=result.get("evidence_path", ""),
            output_files=result.get("output_files", []),
        )
