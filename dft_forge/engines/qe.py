"""QE engine tool: one graph node = one QE calculation.

Bridges the graph runtime to the deterministic compiler /
executor / parser / verifier stack. The tool never shells out directly —
it reuses the Executor abstraction (Local/SSH/Fake), so a node can run
locally or on an HPC backend without the graph knowing.

Calc types:
- vc-relax : pw.x variable-cell relaxation   (T1)
- scf      : pw.x self-consistent field
- nscf     : pw.x non-self-consistent run
- bands    : bands.x band-structure extraction (needs nscf upstream)
- dos      : dos.x density-of-states         (needs nscf upstream)
- pdos     : projwfc.x projected DOS         (needs nscf upstream)
- ph       : ph.x DFPT dynamical matrices   (needs scf upstream)
- q2r      : q2r.x real-space force constants (needs ph upstream)
- matdyn   : matdyn.x frequencies / DOS     (needs q2r upstream)

Data flow follows QE physics plus graph bindings: nscf/bands/dos nodes
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
from dft_forge.protocol.schemas import VerificationInput
from dft_forge.runtime.run import ExecutionContext, NodeRun, ToolError
from dft_forge.verifier import ScientificVerifier

DEFAULT_PSEUDO_DIR = Path(__file__).resolve().parents[2] / "assets" / "pseudos"

BOHR_TO_ANG = 0.529177210903
BOHR3_TO_ANG3 = BOHR_TO_ANG ** 3


def _as_float(params: dict, key: str, default: float) -> float:
    """params.get(k, d) treats an explicit None (templates declare null) as a
    real value and float(None) crashes — treat it as missing instead."""
    v = params.get(key)
    return float(v) if v is not None else default


def _as_int(params: dict, key: str, default: int) -> int:
    v = params.get(key)
    return int(v) if v is not None else default


def _read_structure_preserving_pbc(path):
    """Use extxyz's explicit PBC instead of inferring it from vacuum gaps."""
    from dft_forge.compiler import read_structure

    if Path(str(path)).suffix.lower() == ".extxyz":
        from ase.io import read as ase_read

        if not Path(path).is_file():
            raise FileNotFoundError(f"structure file not found: {path}")
        return ase_read(str(path), format="extxyz")
    return read_structure(path)


class QECalcTool:
    """Graph tool executing a single Quantum ESPRESSO calculation."""

    name = "qe"

    def __init__(self, executor: Optional[Executor] = None, pseudo_dir: Optional[Path] = None):
        self.executor = executor or FakeExecutor()
        self.compiler = QECompiler(pseudo_dir or DEFAULT_PSEUDO_DIR)
        self.verifier = ScientificVerifier()

    def execute(self, node: NodeRun, ctx: ExecutionContext) -> Dict[str, Any]:
        from dft_forge.compiler.qe_parameters import validate_source_lock

        try:
            validate_source_lock(node.params.get("_source_lock"), node.params.get("structure"), self.compiler.pseudo_dir)
        except (ValueError, OSError) as exc:
            raise ToolError(str(exc), category="method_source", repairable=False) from exc
        output = self._execute(node, ctx)
        try:
            validate_source_lock(node.params.get("_source_lock"), node.params.get("structure"), self.compiler.pseudo_dir)
        except (ValueError, OSError) as exc:
            raise ToolError(str(exc), category="method_source", repairable=False) from exc
        return output

    def _execute(self, node: NodeRun, ctx: ExecutionContext) -> Dict[str, Any]:
        params = node.params
        calc = str(params.get("calc", "scf"))
        material = str(params.get("material", ""))
        structure = params.get("structure")
        atoms = None
        if structure:
            source = Path(str(structure)).resolve()
            # Existing graphs refer to the CIF output.  When the same upstream
            # relax also produced an extxyz, consume its exact PBC metadata
            # without changing the graph binding or its source-lock contract.
            relaxed_source = structure
            for upstream in ctx.upstream.values():
                produced = upstream.get("final_structure_cif")
                if produced and source == Path(str(produced)).resolve():
                    extxyz = upstream.get("final_structure_extxyz")
                    if extxyz:
                        if not Path(str(extxyz)).is_file():
                            raise ToolError(
                                "upstream relaxed extxyz structure is missing",
                                category="missing_upstream", repairable=False,
                            )
                        relaxed_source = extxyz
                    break
            atoms = _read_structure_preserving_pbc(relaxed_source)
            if not material or material.lower() in ("", "si"):
                # Si is the template default; a real structure overrides it
                material = Path(str(structure)).stem.replace(" ", "_")
        elif material and material not in MATERIAL_DB and Path(material).suffix.lower() in (".cif", ".poscar", ".xyz", ".extxyz"):
            # material param itself pointing at a structure file
            atoms = _read_structure_preserving_pbc(material)
            material = Path(material).stem.replace(" ", "_")
        elif material and material not in MATERIAL_DB:
            # unknown material key: try formula → prototype (e.g. CaTiO3)
            from dft_forge.compiler import formula_atoms

            atoms, _proto = formula_atoms(material)
        if not material and atoms is None:
            raise ToolError("qe node missing 'material' param", category="config")
        if params.get("supercell") and str(params.get("supercell")) not in {"1x1x1", "1x1x1.0"}:
            from ase.build import make_supercell
            from dft_forge.structure_builder import parse_supercell_3d

            if atoms is None:
                from dft_forge.compiler import build_atoms, formula_atoms

                if material in MATERIAL_DB:
                    atoms = build_atoms(material)
                else:
                    atoms, _proto = formula_atoms(material)
            m, n, p = parse_supercell_3d(str(params["supercell"]))
            atoms = make_supercell(atoms, [[m, 0, 0], [0, n, 0], [0, 0, p]])
        prefix = str(params.get("prefix", re.sub(r"[^A-Za-z0-9]", "_", material.lower())[:24] or "calc"))
        if atoms is not None:
            atoms.info.pop("_qe_method_profile", None)
            # Structure files carry geometry, not the electronic method.
            # Recover it only from the exact upstream relaxation producing it.
            for upstream in ctx.upstream.values():
                produced = (upstream.get("final_structure_cif"), upstream.get("final_structure_extxyz"))
                if structure and any(path and Path(str(structure)).resolve() == Path(str(path)).resolve()
                                     for path in produced):
                    if not upstream.get("method_profile"):
                        raise ToolError("relaxed structure has no electronic-method provenance", category="method_source")
                    atoms.info["_qe_method_profile"] = upstream["method_profile"]
        def with_method(output):
            profile, _, _ = self.compiler._resolve(material, atoms)
            output["method_profile"] = {
                key: profile[key] for key in ("pseudos", "species", "masses", "occupations", "smearing", "degauss",
                    "is_metal", "input_dft", "ecutwfc_default", "ecutrho_default", "kpoints_default") if key in profile
            }
            output["method_profile"]["effective"] = {
                "ecutwfc": params.get("ecutwfc") if params.get("ecutwfc") is not None else profile.get("ecutwfc_default"),
                "ecutrho": params.get("ecutrho") if params.get("ecutrho") is not None else profile.get("ecutrho_default"),
                "kpoints": list(params.get("kpoints") or profile.get("kpoints_default") or []),
                "conv_thr": _as_float(params, "conv_thr", 1.0e-8),
                "tot_charge": float(params.get("tot_charge", 0.0) or 0.0),
                "spin_polarized": bool(params.get("spin_polarized", False)
                    or str(params.get("magnetic_order") or "").lower().replace("-", "") in {"fm", "afm"}),
                "magnetic_order": params.get("magnetic_order"),
                "magnetic_element": params.get("magnetic_element"),
                "initial_moment_muB": params.get("initial_moment_muB"),
                "total_magnetization_muB": params.get("total_magnetization_muB"),
                "occupations": params.get("occupations") if params.get("occupations") is not None else profile.get("occupations"),
                "smearing": params.get("smearing") if params.get("smearing") is not None else profile.get("smearing"),
                "degauss": params.get("degauss") if params.get("degauss") is not None else profile.get("degauss"),
            }
            return output
        workdir = ctx.node_workdir(node.node_id)
        workdir.mkdir(parents=True, exist_ok=True)

        if calc in ("nscf", "bands", "dos", "pdos", "ph"):
            self._link_upstream_save(ctx, workdir, prefix)

        if calc == "vc-relax":
            return with_method(self._run_vc_relax(material, prefix, params, workdir, atoms, timeout=ctx.timeout_seconds))
        if calc == "scf":
            return with_method(self._run_scf(material, prefix, params, workdir, atoms, timeout=ctx.timeout_seconds))
        if calc == "relax":
            params = dict(params)
            params["cell_optimization"] = False
            return with_method(self._run_vc_relax(material, prefix, params, workdir, atoms, timeout=ctx.timeout_seconds))
        if calc == "nscf":
            return with_method(self._run_nscf(material, prefix, params, workdir, atoms, timeout=ctx.timeout_seconds))
        if calc == "bands":
            return self._run_bands(material, prefix, params, workdir, atoms, timeout=ctx.timeout_seconds)
        if calc == "dos":
            return self._run_dos(material, prefix, params, workdir, atoms, timeout=ctx.timeout_seconds)
        if calc == "pdos":
            return self._run_pdos(material, prefix, params, workdir, atoms, timeout=ctx.timeout_seconds)
        if calc == "ph":
            return self._run_ph(material, prefix, params, workdir, timeout=ctx.timeout_seconds)
        if calc == "q2r":
            self._link_upstream_artifacts(ctx, workdir, "dyn_files")
            return self._run_q2r(material, prefix, params, workdir, timeout=ctx.timeout_seconds)
        if calc == "matdyn":
            self._link_upstream_artifacts(ctx, workdir, "force_constants_file")
            return self._run_matdyn(material, prefix, params, workdir, timeout=ctx.timeout_seconds)
        raise ToolError(f"unknown qe calc type '{calc}'", category="config")

    # ── Calc implementations ─────────────────────────────────────────────────

    def _run_vc_relax(self, material, prefix, params, workdir, atoms=None, timeout=None) -> Dict[str, Any]:
        input_file = workdir / f"{prefix}_vcrelax.in"
        self.compiler.compile_t1(
            material,
            input_file,
            prefix=prefix,
            ecutwfc=params.get("ecutwfc"),
            ecutrho=params.get("ecutrho"),
            kpoints=tuple(params["kpoints"]) if params.get("kpoints") else None,
            force_threshold=_as_float(params, "force_threshold", 1.0e-3),
            pressure_threshold=_as_float(params, "pressure_threshold", 0.5),
            conv_thr=_as_float(params, "conv_thr", 1.0e-8),
            nstep=_as_int(params, "nstep", 200),
            cell_optimization=bool(params.get("cell_optimization", True)),
            atoms=atoms,
            spin_polarized=bool(params.get("spin_polarized", False)),
            magnetic_order=params.get("magnetic_order"),
            magnetic_element=params.get("magnetic_element"),
            initial_moment_muB=params.get("initial_moment_muB"),
            total_magnetization_muB=params.get("total_magnetization_muB"),
            tot_charge=float(params.get("tot_charge", 0.0) or 0.0),
            occupations=params.get("occupations"),
            smearing=params.get("smearing"),
            degauss=params.get("degauss"),
        )
        result = self._run_pw(input_file, workdir, timeout=timeout)
        xml = workdir / f"{prefix}.xml"
        parsed = QEParser.parse_vc_relax(result.stdout, xml if xml.exists() else None)
        if bool(params.get("cell_optimization", True)):
            report = self.verifier.verify_t1(
                VerificationInput(
                    stdout=result.stdout,
                    xml_path=xml if xml.exists() else None,
                    job_success=bool(getattr(result, "success", True)),
                    force_threshold_ry=_as_float(params, "force_threshold", 1.0e-3),
                    pressure_threshold_kbar=_as_float(params, "pressure_threshold", 0.5),
                ),
                parsed,
            )
            if not report.passed:
                raise ToolError(
                    "vc-relax verification failed: " + "; ".join(report.failure_reasons),
                    category="verification",
                    repairable=True,
                )
        else:
            scf = QEParser.parse_scf(result.stdout)
            force_present = bool(re.search(r"\batom\s+\d+.*?\bforce\s*=", result.stdout, re.IGNORECASE))
            requested_force_threshold = _as_float(params, "force_threshold", 1.0e-3)
            if (not bool(getattr(result, "success", True)) or "JOB DONE" not in result.stdout
                    or not parsed.converged or not scf.converged or not force_present
                    or parsed.max_force_ry_bohr > requested_force_threshold):
                raise ToolError(
                    "relax verification failed: missing job/geometry/SCF/force convergence evidence",
                    category="verification",
                    repairable=True,
                )
        fixed_cell_relax = not bool(params.get("cell_optimization", True))
        cell = parsed.cell
        fixed_atoms = None
        if fixed_cell_relax:
            # Fixed-cell relaxations must report the exact input lattice.  QE's
            # XML cell is a fallback for parsing, but geometry cannot change by
            # construction and the input also preserves non-periodic axes.
            _, fixed_atoms, _ = self.compiler._resolve(material, atoms)
            a_angstrom = float(fixed_atoms.cell.lengths()[0])
            volume_a3 = float(fixed_atoms.get_volume())
        else:
            a_angstrom = (cell.a_bohr * BOHR_TO_ANG) if cell else None
            volume_a3 = (cell.volume_bohr3 * BOHR3_TO_ANG3) if cell else None
        outputs = {
            "energy_ry": parsed.final_energy_ry,
            "max_force_ry_bohr": parsed.max_force_ry_bohr,
            "force_threshold_ry_bohr": _as_float(params, "force_threshold", 1.0e-3),
            "pressure_kbar": parsed.pressure_kbar if bool(params.get("cell_optimization", True)) else None,
            "n_iterations": parsed.n_iterations,
            "a_angstrom": a_angstrom,
            "volume_a3": volume_a3,
            "stdout_file": str(input_file.with_suffix(".out")),
            "natoms": len(atoms) if atoms is not None else None,
            "fixed_cell_relax": fixed_cell_relax,
            "tot_charge": float(params.get("tot_charge", 0.0) or 0.0),
        }
        if (params.get("spin_polarized") or params.get("magnetic_order")
                or params.get("total_magnetization_muB") is not None):
            magnet = QEParser.parse_magnetization(result.stdout)
            outputs.update({
                "total_magnetization_muB": magnet.total_magnetization_muB,
                "absolute_magnetization_muB": magnet.absolute_magnetization_muB,
                "atom_magnetizations_muB": magnet.atom_magnetizations_muB,
            })

        # export the relaxed structure and attach a geometry/symmetry report so
        # downstream steps (and the chat reply) can cite bond lengths directly
        fixed_cell = fixed_atoms.cell if fixed_atoms is not None else None
        input_pbc = fixed_atoms.pbc if fixed_atoms is not None else (atoms.pbc if atoms is not None else None)
        final_atoms = QEParser.extract_final_structure(
            result.stdout,
            fixed_cell=fixed_cell,
            pbc=input_pbc,
        )
        if final_atoms is not None:
            cif_path = workdir / f"{prefix}_relaxed.cif"
            extxyz_path = workdir / f"{prefix}_relaxed.extxyz"
            from ase.io import write as ase_write

            ase_write(str(cif_path), final_atoms, format="cif")
            ase_write(str(extxyz_path), final_atoms, format="extxyz")
            outputs["final_structure_cif"] = str(cif_path)
            outputs["final_structure_extxyz"] = str(extxyz_path)
            from dft_forge.structure_analysis import analyze

            try:
                outputs["analysis"] = analyze(final_atoms)
            except Exception as exc:
                outputs["analysis_error"] = str(exc)
        return outputs

    def _run_scf(self, material, prefix, params, workdir, atoms=None, timeout=None) -> Dict[str, Any]:
        input_file = workdir / f"{prefix}_scf.in"
        self.compiler.compile_scf(
            material,
            input_file,
            prefix=prefix,
            ecutwfc=params.get("ecutwfc"),
            ecutrho=params.get("ecutrho"),
            kpoints=tuple(params["kpoints"]) if params.get("kpoints") else None,
            conv_thr=_as_float(params, "conv_thr", 1.0e-8),
            nbnd=params.get("nbnd"),
            atoms=atoms,
            spin_polarized=bool(params.get("spin_polarized", False)),
            magnetic_order=params.get("magnetic_order"),
            magnetic_element=params.get("magnetic_element"),
            initial_moment_muB=params.get("initial_moment_muB"),
            total_magnetization_muB=params.get("total_magnetization_muB"),
            tot_charge=float(params.get("tot_charge", 0.0) or 0.0),
            occupations=params.get("occupations"),
            smearing=params.get("smearing"),
            degauss=params.get("degauss"),
        )
        result = self._run_pw(input_file, workdir, timeout=timeout)
        parsed = QEParser.parse_scf(result.stdout)
        report = self.verifier.verify_scf(result.stdout)
        if not report.passed:
            raise ToolError(
                "scf verification failed: " + "; ".join(report.failure_reasons),
                category="scf_not_converged",
                repairable=True,
            )
        nat_match = re.search(r"number of atoms/cell\s+=\s+(\d+)", result.stdout)
        outputs = {
            "energy_ry": parsed.total_energy_ry,
            "fermi_ev": parsed.fermi_energy_ev,
            "n_iterations": parsed.n_iterations,
            "natoms": int(nat_match.group(1)) if nat_match else (len(atoms) if atoms is not None else None),
            "stdout_file": str(input_file.with_suffix(".out")),
            "tot_charge": float(params.get("tot_charge", 0.0) or 0.0),
        }
        if (params.get("spin_polarized") or params.get("magnetic_order")
                or params.get("total_magnetization_muB") is not None):
            magnet = QEParser.parse_magnetization(result.stdout)
            outputs.update({
                "requested_order": (
                    str(params["magnetic_order"]).upper() if params.get("magnetic_order") else None
                ),
                "spin_polarized": bool(params.get("spin_polarized", False)
                    or str(params.get("magnetic_order") or "").lower() in {"fm", "afm"}),
                "total_magnetization_muB": magnet.total_magnetization_muB,
                "absolute_magnetization_muB": magnet.absolute_magnetization_muB,
                "atom_magnetizations_muB": magnet.atom_magnetizations_muB,
            })
        return outputs

    def _run_nscf(self, material, prefix, params, workdir, atoms=None, timeout=None) -> Dict[str, Any]:
        input_file = workdir / f"{prefix}_nscf.in"
        self.compiler.compile_scf(
            material,
            input_file,
            prefix=prefix,
            calculation="nscf",
            ecutwfc=params.get("ecutwfc"),
            ecutrho=params.get("ecutrho"),
            kpoints=tuple(params["kpoints"]) if params.get("kpoints") else None,
            conv_thr=_as_float(params, "conv_thr", 1.0e-8),
            nbnd=params.get("nbnd"),
            nkpoints_bands=_as_int(params, "nkpoints_bands", 60),
            atoms=atoms,
            kmode=str(params.get("kmode", "path")),
        )
        result = self._run_pw(input_file, workdir, timeout=timeout)
        parsed = QEParser.parse_scf(result.stdout)
        report = self.verifier.verify_nscf(result.stdout)
        if not report.passed:
            raise ToolError(
                "nscf verification failed: " + "; ".join(report.failure_reasons),
                category="scf_not_converged",
                repairable=True,
            )
        return {
            # QE NSCF often emits no total energy; parser's legacy default
            # zero is not an observation and must not enter thermodynamics.
            "energy_ry": parsed.total_energy_ry if re.search(r"!\s+total energy\s*=", result.stdout) else None,
            "nbnd_used": params.get("nbnd"),
            "stdout_file": str(input_file.with_suffix(".out")),
        }

    def _run_bands(self, material, prefix, params, workdir, atoms=None, timeout=None) -> Dict[str, Any]:
        input_file = workdir / f"{prefix}_bands.in"
        self.compiler.compile_bands_input(
            material,
            input_file,
            prefix=prefix,
            nkpoints=_as_int(params, "nkpoints_bands", 100),
            atoms=atoms,
        )
        result = self._run_tool("bands.x", input_file, workdir, timeout=timeout)
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
            "gap_known": parsed.gap_known,
            "is_metal": parsed.is_metal,
            "smearing_resolution_ev": parsed.smearing_resolution_ev,
            "fermi_ev": parsed.fermi_energy_ev,
        }
        # plotting axis from the same bandpath the nscf sampling used
        try:
            from dft_forge.compiler import build_atoms

            src = atoms if atoms is not None else build_atoms(material)
            pbc = tuple(bool(b) for b in src.pbc) if len(src.pbc) == 3 else None
            bp = src.cell.bandpath(npoints=max(10, _as_int(params, "nkpoints_bands", 100)), pbc=pbc)
            x, X, labels = bp.get_linear_kpoint_axis()
            outputs["k_axis"] = [round(float(v), 4) for v in x]
            outputs["k_ticks"] = [round(float(v), 4) for v in X]
            outputs["k_labels"] = [("Γ" if str(l).upper() in ("G", "GAMMA") else str(l)) for l in labels]
        except Exception:
            pass
        if parsed.eigenvalues is not None:
            outputs["eigenvalues_ev"] = np.round(parsed.eigenvalues, 4).tolist()
        return outputs

    def _run_dos(self, material, prefix, params, workdir, atoms=None, timeout=None) -> Dict[str, Any]:
        input_file = workdir / f"{prefix}_dos.in"
        self.compiler.compile_dos_input(
            material,
            input_file,
            prefix=prefix,
            deltae=_as_float(params, "dos_deltae", 0.01),
            fwhm=_as_float(params, "dos_fwhm", 0.5),
            ngauss=_as_int(params, "dos_ngauss", 0),
            atoms=atoms,
        )
        result = self._run_tool("dos.x", input_file, workdir, timeout=timeout)
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

    def _run_pdos(self, material, prefix, params, workdir, atoms=None, timeout=None) -> Dict[str, Any]:
        """projwfc.x element-projected DOS. Requires upstream nscf save dir."""
        input_file = workdir / f"{prefix}_pdos.in"
        self.compiler.compile_pdos_input(
            material,
            input_file,
            prefix=prefix,
            deltae=_as_float(params, "dos_deltae", 0.01),
            fwhm=_as_float(params, "dos_fwhm", 0.5),
            ngauss=_as_int(params, "dos_ngauss", 0),
            atoms=atoms,
        )
        result = self._run_projwfc(input_file, workdir, timeout=timeout)
        atm_files = sorted(workdir.glob(f"{prefix}.pdos_atm*"))
        if not atm_files:
            raise ToolError(
                f"projwfc.x produced no pdos_atm files: {result.error_message or 'unknown error'}",
                category="tool_failure",
            )
        parsed = QEParser.parse_pdos(workdir, prefix)
        if parsed.element_dos is None or not parsed.element_dos:
            raise ToolError(
                "pdos parse failed: no element-resolved projections recovered",
                category="tool_failure",
                repairable=False,
            )
        outputs: Dict[str, Any] = {
            "n_energy_points": parsed.n_energy_points,
            "elements": sorted(parsed.element_dos.keys()),
            "fermi_ev": parsed.fermi_energy_ev,
        }
        ef = parsed.fermi_energy_ev
        if (ef is not None and np.isfinite(ef) and parsed.energies is not None
                and len(parsed.energies) > 1 and parsed.energies[0] <= ef <= parsed.energies[-1]):
            outputs["element_dos_at_fermi"] = {
                el: float(np.interp(ef, parsed.energies, values))
                for el, values in sorted(parsed.element_dos.items())
            }
        outputs["pdos_curve"] = {
            "energies_ev": np.round(parsed.energies, 4).tolist(),
            "element_dos": {
                el: np.round(arr, 3).tolist()
                for el, arr in sorted(parsed.element_dos.items())
            },
        }
        return outputs

    def _run_ph(self, material, prefix, params, workdir, timeout=None) -> Dict[str, Any]:
        input_file = workdir / f"{prefix}_ph.in"
        self.compiler.compile_ph_input(
            material,
            input_file,
            prefix=prefix,
            q_grid=tuple(params.get("q_grid") or (1, 1, 1)),
            tr2_ph=float(params.get("tr2_ph", 1.0e-14)),
            fildyn=params.get("fildyn") or f"{prefix}.dyn",
        )
        result = self._run_program("ph.x", input_file, workdir, timeout=timeout)
        dyn_files = sorted(str(p) for p in workdir.glob(f"{prefix}.dyn*"))
        if not dyn_files:
            raise ToolError(
                f"ph.x produced no dynamical matrix files: {result.error_message or 'unknown error'}",
                category="tool_failure",
                repairable=False,
            )
        return {
            "dynamics_prefix": str(workdir / f"{prefix}.dyn"),
            "dyn_files": dyn_files,
            "q_grid": [int(x) for x in (params.get("q_grid") or (1, 1, 1))],
        }

    def _run_q2r(self, material, prefix, params, workdir, timeout=None) -> Dict[str, Any]:
        input_file = workdir / f"{prefix}_q2r.in"
        self.compiler.compile_q2r_input(
            material,
            input_file,
            prefix=prefix,
            fildyn=params.get("dynamics_prefix") or f"{prefix}.dyn",
            flfrc=params.get("force_constants_prefix") or f"{prefix}.fc",
            zasr=str(params.get("zasr", "simple")),
        )
        result = self._run_program("q2r.x", input_file, workdir, timeout=timeout)
        fc = workdir / f"{prefix}.fc"
        if not fc.exists():
            candidates = sorted(workdir.glob(f"{prefix}.fc*"))
            if candidates:
                fc = candidates[0]
            else:
                raise ToolError(
                    f"q2r.x produced no force-constant file: {result.error_message or 'unknown error'}",
                    category="tool_failure",
                    repairable=False,
                )
        return {"force_constants_file": str(fc), "force_constants_prefix": str(fc)}

    def _run_matdyn(self, material, prefix, params, workdir, timeout=None) -> Dict[str, Any]:
        input_file = workdir / f"{prefix}_matdyn.in"
        self.compiler.compile_matdyn_input(
            material,
            input_file,
            prefix=prefix,
            flfrc=params.get("force_constants_prefix") or f"{prefix}.fc",
            flfrq=params.get("frequencies_file") or f"{prefix}.freq",
            q_grid=tuple(params.get("q_grid") or (1, 1, 1)),
            asr=str(params.get("asr", "simple")),
            dos=bool(params.get("dos", True)),
            fldos=params.get("phonon_dos_file") or f"{prefix}.phdos",
        )
        result = self._run_program("matdyn.x", input_file, workdir, timeout=timeout)
        parsed = QEParser.parse_phonon(result.stdout)
        freq_file = workdir / f"{prefix}.freq"
        dos_file = workdir / f"{prefix}.phdos"
        modes_candidates = [workdir / "matdyn.modes", workdir / f"{prefix}.modes"]
        if not parsed.frequencies_thz and freq_file.exists():
            parsed = QEParser.parse_phonon(freq_file.read_text(errors="ignore"))
        if not parsed.frequencies_thz:
            for candidate in modes_candidates + sorted(workdir.glob("*.modes")):
                if candidate.exists():
                    parsed = QEParser.parse_phonon(candidate.read_text(errors="ignore"))
                    if parsed.frequencies_thz:
                        break
        if not parsed.frequencies_thz:
            raise ToolError(
                "matdyn.x produced no parseable phonon frequencies",
                category="verification",
                repairable=False,
            )
        return {
            "frequencies_thz": parsed.frequencies_thz,
            "n_modes": len(parsed.frequencies_thz),
            "frequency_file": str(freq_file) if freq_file.exists() else None,
            "phonon_dos_file": str(dos_file) if dos_file.exists() else None,
            "modes_file": next((str(p) for p in modes_candidates if p.exists()), None),
            "q_grid": [int(x) for x in (params.get("q_grid") or (1, 1, 1))],
        }

    def _run_program(self, binary: str, input_file: Path, workdir: Path, timeout=None):
        runner = getattr(self.executor, {
            "ph.x": "run_ph_x", "q2r.x": "run_q2r_x", "matdyn.x": "run_matdyn_x",
        }.get(binary, ""), None)
        if runner is None:
            raise ToolError(f"executor does not provide {binary}", category="config", repairable=False)
        result = runner(input_file, workdir, timeout=timeout)
        if not result.success or "JOB DONE" not in (result.stdout or ""):
            raise ToolError(
                f"{binary} failed: {result.error_message or result.stderr or 'no JOB DONE in stdout'}",
                category="tool_failure", repairable=True,
            )
        return result

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

    def _run_pw(self, input_file: Path, workdir: Path, timeout=None):
        result = self.executor.run_pw(input_file, workdir, timeout=timeout)
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
        m = re.search(
            r"Error in routine\s+([^\s(]+)\s*\(\d+\):\s*\n\s*([^\n]+)",
            stdout,
        )
        if m:
            return f"{m.group(1)}: {m.group(2).strip()}"
        return None

    def _run_tool(self, tool: str, input_file: Path, workdir: Path, timeout=None):
        runner = {
            "bands.x": self.executor.run_bands_x,
            "dos.x": self.executor.run_dos_x,
            "projwfc.x": self.executor.run_projwfc_x,
        }
        result = runner[tool](input_file, workdir, timeout=timeout)
        if not result.success:
            raise ToolError(
                f"{tool} failed: {result.error_message or result.stderr or 'exit != 0'}",
                category="tool_failure",
                repairable=True,
            )
        return result

    def _run_projwfc(self, input_file: Path, workdir: Path, timeout=None):
        result = self.executor.run_projwfc_x(input_file, workdir, timeout=timeout)
        if not result.success or "JOB DONE" not in (result.stdout or ""):
            raise ToolError(
                f"projwfc.x failed: {self._qe_error_snippet(result) or result.error_message or result.stderr or 'no JOB DONE in stdout'}",
                category="tool_failure",
                repairable=True,
            )
        return result

    @staticmethod
    def _link_upstream_save(ctx: ExecutionContext, workdir: Path, prefix: str) -> None:
        """Symlink the upstream {prefix}.save into this node's workdir.

        Prefer the nscf dependency when present: bands/dos/projwfc must reuse
        the nscf wavefunctions and k-point set — linking scf's save instead
        silently computes on the wrong k-mesh (alphabetical order only worked
        by luck: 'nscf' < 'scf')."""
        deps = list(ctx.upstream)
        preferred = [d for d in deps if "nscf" in d] + [d for d in deps if "nscf" not in d]
        for dep_id in preferred:
            # resolve() makes the link absolute — a relative base_dir would
            # otherwise produce a symlink broken from the node workdir
            dep_save = (ctx.base_dir / dep_id / f"{prefix}.save").resolve()
            if dep_save.exists():
                target = workdir / f"{prefix}.save"
                if target.is_symlink() and not target.exists():
                    # stale link from a moved/renamed upstream — replace it,
                    # otherwise the node would silently reuse the dead path
                    target.unlink()
                elif target.exists() or target.is_symlink():
                    return
                target.symlink_to(dep_save, target_is_directory=True)
                return
        raise ToolError(
            f"no upstream {prefix}.save found — add a succeeded scf/nscf dependency",
            category="missing_upstream",
            repairable=False,
        )

    @staticmethod
    def _link_upstream_artifacts(ctx: ExecutionContext, workdir: Path, output_key: str) -> None:
        """Make file outputs of a predecessor visible to q2r/matdyn nodes."""
        for outputs in ctx.upstream.values():
            values = outputs.get(output_key, []) if isinstance(outputs, dict) else []
            if isinstance(values, str):
                values = [values]
            for raw in values:
                source = Path(str(raw))
                if not source.exists():
                    continue
                target = workdir / source.name
                if target.exists() or target.is_symlink():
                    continue
                target.symlink_to(source.resolve())


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
        if node.params.get("_parameters_locked"):
            node.error = (node.error or str(error)) + " | 参数已确认；调整计算参数后请重新预览并确认"
            return False
        if error.category not in self.REPAIRABLE_CATEGORIES:
            return False
        params = node.params
        if "charge is wrong" in str(error).lower():
            # This is an occupations/Fermi-level inconsistency, not a basis-set
            # convergence problem.  A node-local cutoff bump both wastes work
            # and breaks matched-energy comparisons.  Require the caller to
            # revise the spin/occupation policy for the whole workflow.
            node.error = (node.error or "") + (
                " | charge/occupation inconsistency; revise occupations, smearing "
                "or the spin constraint consistently for the full comparison"
            )
            return False
        # nbnd failures must not be repaired with cutoff bumps
        if "too few bands" in str(error):
            current = int(params.get("nbnd") or 0)
            bumped = max(current * 2, 24)
            params["nbnd"] = bumped
            node.error = (node.error or "") + f" | repair: nbnd -> {bumped}"
            return True
        # timeouts need LESS work, not more: a cutoff bump makes every retry
        # slower and guarantees another timeout, so shrink k-point cost instead
        if "timed out" in str(error).lower():
            return self._repair_timeout(node, params)
        current = float(params.get("ecutwfc") or 45.0)
        if current >= self.MAX_ECUT:
            return False
        params["ecutwfc"] = min(current + self.CUTOFF_BUMP_STEP, self.MAX_ECUT)
        params["ecutrho"] = params["ecutwfc"] * 8.0
        node.error = (node.error or "") + f" | repair: ecutwfc -> {params['ecutwfc']}"
        return True

    MIN_BAND_PATH_POINTS = 24

    def _repair_timeout(self, node: NodeRun, params: dict) -> bool:
        """Reduce k-point cost: band-path sampling first, then uniform mesh."""
        if params.get("nkpoints_bands"):
            current = int(params["nkpoints_bands"])
            if current <= self.MIN_BAND_PATH_POINTS:
                return False
            reduced = max(self.MIN_BAND_PATH_POINTS, int(current * 0.6))
            params["nkpoints_bands"] = reduced
            node.error = (node.error or "") + f" | repair: nkpoints_bands -> {reduced}"
            return True
        kpts = params.get("kpoints")
        if isinstance(kpts, (list, tuple)) and len(kpts) >= 3 and any(n > 2 for n in kpts[:3]):
            reduced = [max(2, int(n) // 2) if n > 2 else int(n) for n in kpts[:3]]
            params["kpoints"] = reduced + [int(n) for n in kpts[3:]]
            node.error = (node.error or "") + f" | repair: kpoints -> {params['kpoints']}"
            return True
        return False
