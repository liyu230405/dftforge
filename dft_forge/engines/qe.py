"""QE engine tool: one graph node = one QE calculation.

Bridges the CatGo-style graph runtime to the deterministic compiler /
executor / parser / verifier stack. The tool never shells out directly —
it reuses the Executor abstraction (Local/SSH/Fake), so a node can run
locally or on an HPC backend without the graph knowing.

Calc types:
- vc-relax : pw.x variable-cell relaxation   (T1)
- scf      : pw.x self-consistent field
- nscf     : pw.x non-self-consistent run
- bands    : bands.x band-structure extraction (needs nscf upstream)
- dos      : dos.x density-of-states         (needs nscf upstream)

Data flow follows QE physics plus CatGo-style bindings: nscf/bands/dos nodes
symlink the upstream ``{prefix}.save`` directory into their workdir, and the
T2 verifiers receive upstream stdout files through
``${nodes.<id>.outputs.stdout_file}`` param references.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from dft_forge.compiler import MATERIAL_DB, QECompiler
from dft_forge.executor import Executor, FakeExecutor
from dft_forge.parser import QEParser
from dft_forge.runtime.run import ExecutionContext, NodeRun, ToolError
from dft_forge.verifier import ScientificVerifier

DEFAULT_PSEUDO_DIR = Path(__file__).resolve().parents[2] / "assets" / "pseudos"

BOHR_TO_ANG = 0.529177210903
BOHR3_TO_ANG3 = BOHR_TO_ANG ** 3


class QECalcTool:
    """Graph tool executing a single Quantum ESPRESSO calculation."""

    name = "qe"

    def __init__(self, executor: Optional[Executor] = None, pseudo_dir: Optional[Path] = None):
        self.executor = executor or FakeExecutor()
        self.compiler = QECompiler(pseudo_dir or DEFAULT_PSEUDO_DIR)
        self.verifier = ScientificVerifier()

    def execute(self, node: NodeRun, ctx: ExecutionContext) -> Dict[str, Any]:
        params = node.params
        calc = str(params.get("calc", "scf"))
        material = str(params.get("material", ""))
        structure = params.get("structure")
        atoms = None
        if structure:
            from dft_forge.compiler import read_structure

            atoms = read_structure(structure)
            if not material or material.lower() in ("", "si"):
                # Si is the template default; a real structure overrides it
                material = Path(str(structure)).stem.replace(" ", "_")
        elif material and material not in MATERIAL_DB and Path(material).suffix.lower() in (".cif", ".poscar", ".xyz"):
            # material param itself pointing at a structure file
            from dft_forge.compiler import read_structure

            atoms = read_structure(material)
            material = Path(material).stem.replace(" ", "_")
        elif material and material not in MATERIAL_DB:
            # unknown material key: try formula → prototype (e.g. CaTiO3)
            from dft_forge.compiler import formula_atoms

            atoms, _proto = formula_atoms(material)
        if not material and atoms is None:
            raise ToolError("qe node missing 'material' param", category="config")
        prefix = str(params.get("prefix", re.sub(r"[^A-Za-z0-9]", "_", material.lower())[:24] or "calc"))
        workdir = ctx.node_workdir(node.node_id)
        workdir.mkdir(parents=True, exist_ok=True)

        if calc in ("nscf", "bands", "dos"):
            self._link_upstream_save(ctx, workdir, prefix)

        if calc == "vc-relax":
            return self._run_vc_relax(material, prefix, params, workdir, atoms)
        if calc == "scf":
            return self._run_scf(material, prefix, params, workdir, atoms)
        if calc == "nscf":
            return self._run_nscf(material, prefix, params, workdir, atoms)
        if calc == "bands":
            return self._run_bands(material, prefix, params, workdir, atoms)
        if calc == "dos":
            return self._run_dos(material, prefix, params, workdir, atoms)
        raise ToolError(f"unknown qe calc type '{calc}'", category="config")

    # ── Calc implementations ─────────────────────────────────────────────────

    def _run_vc_relax(self, material, prefix, params, workdir, atoms=None) -> Dict[str, Any]:
        input_file = workdir / f"{prefix}_vcrelax.in"
        self.compiler.compile_t1(
            material,
            input_file,
            prefix=prefix,
            ecutwfc=params.get("ecutwfc"),
            ecutrho=params.get("ecutrho"),
            kpoints=tuple(params["kpoints"]) if params.get("kpoints") else None,
            atoms=atoms,
        )
        result = self._run_pw(input_file, workdir)
        xml = workdir / f"{prefix}.xml"
        parsed = QEParser.parse_vc_relax(result.stdout, xml if xml.exists() else None)
        report = self.verifier.verify_t1(result, parsed)
        if not report.passed:
            raise ToolError(
                "vc-relax verification failed: " + "; ".join(report.failure_reasons),
                category="verification",
                repairable=True,
            )
        cell = parsed.cell
        return {
            "energy_ry": parsed.final_energy_ry,
            "max_force_ry_bohr": parsed.max_force_ry_bohr,
            "pressure_kbar": parsed.pressure_kbar,
            "n_iterations": parsed.n_iterations,
            "a_angstrom": (cell.a_bohr * BOHR_TO_ANG) if cell else None,
            "volume_a3": (cell.volume_bohr3 * BOHR3_TO_ANG3) if cell else None,
            "stdout_file": str(input_file.with_suffix(".out")),
        }

    def _run_scf(self, material, prefix, params, workdir, atoms=None) -> Dict[str, Any]:
        input_file = workdir / f"{prefix}_scf.in"
        self.compiler.compile_scf(
            material,
            input_file,
            prefix=prefix,
            ecutwfc=params.get("ecutwfc"),
            ecutrho=params.get("ecutrho"),
            kpoints=tuple(params["kpoints"]) if params.get("kpoints") else None,
            conv_thr=float(params.get("conv_thr", 1.0e-8)),
            nbnd=params.get("nbnd"),
            atoms=atoms,
        )
        result = self._run_pw(input_file, workdir)
        parsed = QEParser.parse_scf(result.stdout)
        report = self.verifier.verify_scf(result.stdout)
        if not report.passed:
            raise ToolError(
                "scf verification failed: " + "; ".join(report.failure_reasons),
                category="scf_not_converged",
                repairable=True,
            )
        return {
            "energy_ry": parsed.total_energy_ry,
            "fermi_ev": parsed.fermi_energy_ev,
            "n_iterations": parsed.n_iterations,
            "stdout_file": str(input_file.with_suffix(".out")),
        }

    def _run_nscf(self, material, prefix, params, workdir, atoms=None) -> Dict[str, Any]:
        input_file = workdir / f"{prefix}_nscf.in"
        self.compiler.compile_scf(
            material,
            input_file,
            prefix=prefix,
            calculation="nscf",
            ecutwfc=params.get("ecutwfc"),
            ecutrho=params.get("ecutrho"),
            kpoints=tuple(params["kpoints"]) if params.get("kpoints") else None,
            conv_thr=float(params.get("conv_thr", 1.0e-8)),
            nbnd=params.get("nbnd"),
            nkpoints_bands=int(params.get("nkpoints_bands", 60)),
            atoms=atoms,
        )
        result = self._run_pw(input_file, workdir)
        parsed = QEParser.parse_scf(result.stdout)
        report = self.verifier.verify_nscf(result.stdout)
        if not report.passed:
            raise ToolError(
                "nscf verification failed: " + "; ".join(report.failure_reasons),
                category="scf_not_converged",
                repairable=True,
            )
        return {
            "energy_ry": parsed.total_energy_ry,
            "nbnd_used": params.get("nbnd"),
            "stdout_file": str(input_file.with_suffix(".out")),
        }

    def _run_bands(self, material, prefix, params, workdir, atoms=None) -> Dict[str, Any]:
        input_file = workdir / f"{prefix}_bands.in"
        self.compiler.compile_bands_input(
            material,
            input_file,
            prefix=prefix,
            nkpoints=int(params.get("nkpoints_bands", 100)),
            atoms=atoms,
        )
        result = self._run_tool("bands.x", input_file, workdir)
        bands_xml = self._locate_bands_xml(workdir, prefix)
        if bands_xml is None:
            raise ToolError(
                f"bands.x produced no eigenvalue XML ({prefix}.xml / "
                f"{prefix}.save/data-file-schema.xml): "
                f"{result.error_message or 'unknown error'}",
                category="tool_failure",
                repairable=False,
            )
        parsed = QEParser.parse_bands(bands_xml)
        report = self.verifier.verify_t2_bands(
            self._read_stdout(params.get("scf_stdout")),
            self._read_stdout(params.get("nscf_stdout")),
            bands_xml,
        )
        if not report.passed:
            raise ToolError(
                "bands verification failed: " + "; ".join(report.failure_reasons),
                category="verification",
                repairable=True,
            )
        outputs = {
            "n_bands": parsed.n_bands,
            "n_kpoints": parsed.n_kpoints,
            "band_gap_ev": parsed.band_gap_ev,
            "is_metal": parsed.is_metal,
            "fermi_ev": parsed.fermi_energy_ev,
        }
        # plotting axis from the same bandpath the nscf sampling used
        try:
            from dft_forge.compiler import build_atoms

            src = atoms if atoms is not None else build_atoms(material)
            bp = src.cell.bandpath(npoints=max(10, int(params.get("nkpoints_bands", 100))))
            x, X, labels = bp.get_linear_kpoint_axis()
            outputs["k_axis"] = [round(float(v), 4) for v in x]
            outputs["k_ticks"] = [round(float(v), 4) for v in X]
            outputs["k_labels"] = [("Γ" if str(l).upper() in ("G", "GAMMA") else str(l)) for l in labels]
        except Exception:
            pass
        if parsed.eigenvalues is not None:
            outputs["eigenvalues_ev"] = np.round(parsed.eigenvalues, 4).tolist()
        return outputs

    def _run_dos(self, material, prefix, params, workdir, atoms=None) -> Dict[str, Any]:
        input_file = workdir / f"{prefix}_dos.in"
        self.compiler.compile_dos_input(
            material,
            input_file,
            prefix=prefix,
            deltae=float(params.get("dos_deltae", 0.01)),
            fwhm=float(params.get("dos_fwhm", 0.05)),
            atoms=atoms,
        )
        result = self._run_tool("dos.x", input_file, workdir)
        dos_file = workdir / f"{prefix}.dos"
        if not dos_file.exists():
            candidates = sorted(workdir.glob("*.dos"))
            if candidates:
                dos_file = candidates[0]
            else:
                raise ToolError(
                    f"dos.x produced no .dos file: {result.error_message or 'unknown error'}",
                    category="tool_failure",
                )
        parsed = QEParser.parse_dos(dos_file)
        report = self.verifier.verify_t2_dos(
            self._read_stdout(params.get("scf_stdout")),
            self._read_stdout(params.get("nscf_stdout")),
            dos_file,
        )
        if not report.passed:
            raise ToolError(
                "dos verification failed: " + "; ".join(report.failure_reasons),
                category="verification",
                repairable=True,
            )
        outputs = {
            "n_energy_points": parsed.n_energy_points,
            "dos_at_fermi": parsed.dos_at_fermi,
            "fermi_ev": parsed.fermi_energy_ev,
        }
        if parsed.energies is not None and parsed.dos is not None:
            outputs["dos_curve"] = {
                "energies_ev": np.round(parsed.energies, 4).tolist(),
                "dos": np.round(parsed.dos, 3).tolist(),
            }
        return outputs

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _locate_bands_xml(workdir: Path, prefix: str) -> Optional[Path]:
        """Find the QE XML holding nscf eigenvalues (version-robust)."""
        candidates = [
            workdir / f"{prefix}.xml",
            workdir / f"{prefix}.save" / "data-file-schema.xml",
            workdir / f"{prefix}_bands.xml",
        ]
        for c in candidates:
            if c.exists():
                return c
        return None

    @staticmethod
    def _read_stdout(path: Any) -> str:
        if not path:
            return ""
        p = Path(str(path))
        return p.read_text() if p.exists() else ""

    def _run_pw(self, input_file: Path, workdir: Path):
        result = self.executor.run_pw(input_file, workdir)
        if not result.success or "JOB DONE" not in (result.stdout or ""):
            raise ToolError(
                f"pw.x failed: {self._qe_error_snippet(result) or result.error_message or result.stderr or 'no JOB DONE in stdout'}",
                category="execution",
                repairable=True,
            )
        return result

    @staticmethod
    def _qe_error_snippet(result) -> Optional[str]:
        """Extract the 'Error in routine (...)' block from pw.x stdout."""
        stdout = result.stdout or ""
        m = re.search(r"Error in routine\s+(\S+)\s*\(\d+\):\s*\n(.+?)\n\s*\n", stdout)
        if m:
            return f"{m.group(1)}: {m.group(2).strip()}"
        return None

    def _run_tool(self, tool: str, input_file: Path, workdir: Path):
        runner = {"bands.x": self.executor.run_bands_x, "dos.x": self.executor.run_dos_x}
        result = runner[tool](input_file, workdir)
        if not result.success:
            raise ToolError(
                f"{tool} failed: {result.error_message or result.stderr or 'exit != 0'}",
                category="tool_failure",
                repairable=True,
            )
        return result

    @staticmethod
    def _link_upstream_save(ctx: ExecutionContext, workdir: Path, prefix: str) -> None:
        """Symlink the upstream {prefix}.save into this node's workdir."""
        for dep_id in sorted(ctx.upstream):
            # resolve() makes the link absolute — a relative base_dir would
            # otherwise produce a symlink broken from the node workdir
            dep_save = (ctx.base_dir / dep_id / f"{prefix}.save").resolve()
            if dep_save.exists():
                target = workdir / f"{prefix}.save"
                if target.exists() or target.is_symlink():
                    return
                target.symlink_to(dep_save, target_is_directory=True)
                return
        raise ToolError(
            f"no upstream {prefix}.save found — add a succeeded scf/nscf dependency",
            category="missing_upstream",
            repairable=False,
        )


class QEParamRepairer:
    """Repair handler that strengthens QE parameters and requeues the node.

    Mirrors the dft-forge recovery philosophy: safe, monotone parameter
    adjustments (higher cutoff) for convergence-class failures.
    """

    name = "qe_param_bump"

    CUTOFF_BUMP_STEP = 10.0
    MAX_ECUT = 120.0
    REPAIRABLE_CATEGORIES = ("scf_not_converged", "verification", "execution")

    def repair(self, node: NodeRun, error: ToolError) -> bool:
        if error.category not in self.REPAIRABLE_CATEGORIES:
            return False
        params = node.params
        # nbnd failures must not be repaired with cutoff bumps
        if "too few bands" in str(error):
            current = int(params.get("nbnd") or 0)
            bumped = max(current * 2, 24)
            params["nbnd"] = bumped
            node.error = (node.error or "") + f" | repair: nbnd -> {bumped}"
            return True
        current = float(params.get("ecutwfc") or 45.0)
        if current >= self.MAX_ECUT:
            return False
        params["ecutwfc"] = min(current + self.CUTOFF_BUMP_STEP, self.MAX_ECUT)
        params["ecutrho"] = params["ecutwfc"] * 8.0
        node.error = (node.error or "") + f" | repair: ecutwfc -> {params['ecutwfc']}"
        return True
