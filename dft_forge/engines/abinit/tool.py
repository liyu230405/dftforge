"""ABINIT graph tool: one graph node = one ABINIT calculation.

Bridges the graph runtime to the deterministic ABINIT writer / parser. Like the
QE tool, ABINIT reads a single ``.abi`` input file, so a node runs via
``executor.run_abinit(input_file, workdir)`` — locally, over SSH/HPC, or against
a FakeExecutor in tests without the graph knowing which.

Calc types mirror the QE/VASP engines: scf, relax, vc-relax, nscf, bands, dos,
pdos, plus a native DFPT phonon chain (dfpt -> anaddb) analogous to QE's
ph -> q2r -> matdyn.
"""

from __future__ import annotations

from pathlib import Path
import re
import hashlib
import time
from typing import Any, Dict, Optional

from dft_forge.engines.abinit import parser as ab_parser
from dft_forge.engines.abinit import writer as ab_writer
from dft_forge.engines.abinit.writer import ABINITInputError, default_pp_dir
from dft_forge.executor import Executor, FakeExecutor
from dft_forge.runtime.run import ExecutionContext, NodeRun, ToolError

# Calc types that consume a frozen density/WFK from an upstream scf/relax node.
_NEEDS_WFK = {"nscf", "bands", "dos", "pdos"}


class ABINITCalcTool:
    """Graph tool executing a single ABINIT calculation."""

    name = "abinit"

    def __init__(self, executor: Optional[Executor] = None, pp_dir: Optional[Path] = None):
        self.executor = executor or FakeExecutor()
        self.pp_dir = pp_dir if pp_dir is not None else default_pp_dir()

    def execute(self, node: NodeRun, ctx: ExecutionContext) -> Dict[str, Any]:
        params = node.params
        calc = str(params.get("calc", "scf"))
        material = str(params.get("material", ""))
        structure = params.get("structure")
        workdir = ctx.node_workdir(node.node_id)

        runner = getattr(self.executor, "run_abinit", None)
        if runner is None:
            raise ToolError(
                "configured executor does not support ABINIT (no run_abinit)",
                category="config", repairable=False)

        # The anaddb post-processing node runs a different binary on the
        # upstream DDB; it does not follow the single-.abi flow.
        if calc == "anaddb":
            return self._run_anaddb(node, ctx, workdir, material, structure)

        # nscf/bands/dos/pdos reuse the upstream self-consistent density,
        # the ABINIT analogue of QE linking an upstream {prefix}.save.
        upstream_atoms = None
        if calc in _NEEDS_WFK:
            upstream_atoms = self._stage_density(ctx, workdir)
        elif ctx.upstream:
            upstream_atoms, _ = self._resolve_structure(ctx, material, structure)
        write_params = {k: v for k, v in params.items()
                        if k != "supercell" or upstream_atoms is None}

        try:
            meta = ab_writer.write_inputs(
                material, structure, calc, write_params, workdir,
                pp_dir=self.pp_dir, atoms=upstream_atoms,
            )
        except ABINITInputError as exc:
            raise ToolError(str(exc), category="config", repairable=False) from exc

        abi_file = Path(meta["abi_file"])
        started = time.monotonic()
        result = runner(abi_file, workdir, timeout=ctx.timeout_seconds)
        stdout = result.stdout or ""
        error = ab_parser.classify_error(stdout) or ab_parser.classify_error(
            self._read_abo(workdir))
        if not result.success:
            raise ToolError(
                (error.message if error else result.error_message or "ABINIT run failed"),
                category=(error.category if error else "execution"),
                repairable=bool(error and error.category in {"verification", "electronic_instability"}),
            )

        dispatch = {
            "scf": self._finish_scf, "relax": self._finish_relax,
            "vc-relax": self._finish_relax, "nscf": self._finish_nscf,
            "bands": self._finish_bands, "dos": self._finish_dos,
            "pdos": self._finish_pdos, "dfpt": self._finish_dfpt,
        }
        handler = dispatch.get(calc)
        if handler is None:
            raise ToolError(f"unknown abinit calc type '{calc}'", category="config", repairable=False)
        if calc == "dfpt":
            timeout = max(0.001, ctx.timeout_seconds - (time.monotonic() - started)) if ctx.timeout_seconds else None
            self._merge_ddb(workdir, params, timeout)
        outputs = handler(workdir, stdout, meta, params)
        outputs.update({"engine": "abinit", "calc": calc,
                        "natoms": meta["natoms"], "species": meta["species"],
                        "stdout_file": str(abi_file.with_suffix(".abo")),
                        "input_structure_cif": meta["input_structure_cif"]})
        outputs["method_profile"] = self._method_profile(meta, params)
        return outputs

    # ── Per-calc finishers ────────────────────────────────────────────────────

    def _finish_scf(self, workdir, stdout, meta, params) -> Dict[str, Any]:
        parsed = ab_parser.parse_result(workdir, stdout)
        if not parsed.converged or parsed.total_energy_ev is None:
            raise ToolError("ABINIT scf did not reach electronic convergence",
                            category="verification", repairable=True)
        out = {"energy_ev": parsed.total_energy_ev, "converged": True,
               "fermi_energy_ev": parsed.fermi_energy_ev,
               "n_scf_steps": parsed.n_scf_steps}
        self._attach_magnetization(out, parsed, meta)
        return out

    def _finish_relax(self, workdir, stdout, meta, params) -> Dict[str, Any]:
        parsed = ab_parser.parse_result(workdir, stdout)
        if not parsed.converged or parsed.total_energy_ev is None:
            raise ToolError("ABINIT relaxation did not converge electronically",
                            category="verification", repairable=True)
        # tolmxf is in Ha/Bohr; convert the parsed eV/Å force back for the check.
        threshold_ev_ang = ab_writer._as_float(params, "tolmxf", 5.0e-5) * ab_parser.HA_BOHR_TO_EV_ANG
        if not parsed.ionic_converged or parsed.max_force_ev_ang is None:
            raise ToolError("ABINIT relaxation has no converged force/stress evidence",
                            category="verification", repairable=False)
        if parsed.max_force_ev_ang > threshold_ev_ang:
            raise ToolError(
                f"ABINIT relaxation force {parsed.max_force_ev_ang:.4f} eV/Å not converged",
                category="verification", repairable=True)
        out = {"energy_ev": parsed.total_energy_ev, "converged": True,
               "fermi_energy_ev": parsed.fermi_energy_ev,
               "final_volume_ang3": parsed.final_volume_ang3,
               "max_force_ev_ang": parsed.max_force_ev_ang}
        self._attach_magnetization(out, parsed, meta)
        self._export_structure(workdir, meta, out)
        return out

    def _finish_nscf(self, workdir, stdout, meta, params) -> Dict[str, Any]:
        parsed = ab_parser.parse_result(workdir, stdout)
        return {"energy_ev": parsed.total_energy_ev or None, "converged": True,
                "fermi_energy_ev": parsed.fermi_energy_ev}

    def _finish_bands(self, workdir, stdout, meta, params) -> Dict[str, Any]:
        bands = ab_parser.parse_bands(workdir, stdout)
        if bands.n_kpoints == 0:
            raise ToolError("bands run produced no eigenvalues",
                            category="verification", repairable=False)
        return {"converged": True, "n_bands": bands.n_bands,
                "n_kpoints": bands.n_kpoints, "band_gap_ev": bands.band_gap_ev,
                "is_metal": bands.is_metal, "fermi_energy_ev": bands.fermi_energy_ev,
                "vbm_ev": bands.vbm_ev, "cbm_ev": bands.cbm_ev}

    def _finish_dos(self, workdir, stdout, meta, params) -> Dict[str, Any]:
        dos = ab_parser.parse_dos(workdir, projected=False)
        if dos.n_energy_points == 0:
            raise ToolError("dos run produced no *_DOS data",
                            category="verification", repairable=False)
        out = {"converged": True, "n_energy_points": dos.n_energy_points,
               "fermi_energy_ev": dos.fermi_energy_ev, "dos_at_fermi": dos.dos_at_fermi}
        if dos.energies_ev is not None and dos.dos is not None:
            out["dos_curve"] = {"energies_ev": [round(e, 4) for e in dos.energies_ev],
                                "dos": [round(d, 3) for d in dos.dos]}
        return out

    def _finish_pdos(self, workdir, stdout, meta, params) -> Dict[str, Any]:
        dos = ab_parser.parse_dos(workdir, projected=True, atom_symbols=meta.get("atom_symbols"))
        if dos.n_energy_points == 0:
            raise ToolError("pdos run produced no *_DOS data",
                            category="verification", repairable=False)
        if not dos.element_dos:
            raise ToolError("pdos parse recovered no projected DOS (need prtdos 3 *_DOS_AT files)",
                            category="verification", repairable=False)
        out = {"converged": True, "n_energy_points": dos.n_energy_points,
               "fermi_energy_ev": dos.fermi_energy_ev, "dos_unit": "states/eV",
               "elements": sorted(dos.element_dos.keys())}
        if dos.energies_ev is not None:
            out["pdos_curve"] = {
                "energies_ev": [round(e, 4) for e in dos.energies_ev],
                "element_dos": {el: [round(v, 3) for v in arr]
                                for el, arr in sorted(dos.element_dos.items())}}
        return out

    def _finish_dfpt(self, workdir, stdout, meta, params) -> Dict[str, Any]:
        """The DFPT run must leave a DDB behind for the anaddb node."""
        ddb = self._find_ddb(workdir)
        if ddb is None:
            raise ToolError("DFPT run produced no DDB file (expected *_DDB)",
                            category="verification", repairable=False)
        parsed = ab_parser.parse_result(workdir, stdout)
        if not parsed.converged:
            raise ToolError("DFPT did not converge", category="verification", repairable=False)
        return {"converged": True, "energy_ev": parsed.total_energy_ev or None,
                "ddb_file": str(ddb), "ngqpt": ab_writer.phonon_qpoints(params)[0],
                "n_qpoints": len(ab_writer.phonon_qpoints(params)[1])}

    def _merge_ddb(self, workdir, params, timeout):
        """Require every q-point dataset before merging, never select the GS DDB."""
        _, qpoints = ab_writer.phonon_qpoints(params)
        files = []
        for i in range(2, len(qpoints) + 2):
            found = sorted(Path(workdir).glob(f"*_DS{i}_DDB"))
            if len(found) != 1:
                raise ToolError(f"missing DFPT DDB for q-point dataset {i}",
                                category="verification", repairable=False)
            files.append(found[0])
        merged = Path(workdir) / "phonons_DDB"
        if len(files) == 1:
            merged.write_bytes(files[0].read_bytes())
        else:
            merge = getattr(self.executor, "merge_abinit_ddb", None)
            if merge is None:
                raise ToolError("executor cannot merge DFPT DDB files (mrgddb required)",
                                category="config", repairable=False)
            result = merge(files, merged, workdir, timeout=timeout)
            if not result.success or not merged.is_file():
                raise ToolError(f"mrgddb failed: {result.error_message}",
                                category="execution", repairable=False)

    # ── anaddb phonon post-processing ─────────────────────────────────────────

    def _run_anaddb(self, node, ctx, workdir, material, structure):
        """Interpolate the upstream DDB onto a band q-path + DOS via anaddb."""
        from dft_forge.workflows import phonon_stability

        params = node.params
        Path(workdir).mkdir(parents=True, exist_ok=True)
        runner = getattr(self.executor, "run_abinit", None)
        ddb = None
        producer = None
        for outputs in ctx.upstream.values():
            if isinstance(outputs, dict) and outputs.get("ddb_file"):
                ddb = outputs["ddb_file"]
                producer = outputs
                break
        if not ddb or not Path(str(ddb)).is_file():
            raise ToolError("no upstream DDB — add a dfpt dependency",
                            category="missing_upstream", repairable=False)
        grid = ab_writer.phonon_qpoints(params)[0]
        if producer.get("ngqpt") != grid:
            raise ToolError("anaddb q-mesh does not match the upstream DDB",
                            category="verification", repairable=False)
        # Link the DDB in and write the anaddb input.
        target = Path(workdir) / "in.ddb"
        if not (target.exists() or target.is_symlink()):
            target.symlink_to(Path(str(ddb)).resolve())
        atoms, label = self._resolve_structure(ctx, material, structure)
        anaddb_in = ab_writer.write_anaddb_input(params, workdir, label, atoms)

        result = runner(anaddb_in, workdir, timeout=ctx.timeout_seconds, binary="anaddb")
        text = self._read_anaddb_out(workdir) or result.stdout or ""
        if not result.success:
            err = ab_parser.classify_error(text)
            raise ToolError((err.message if err else result.error_message or "anaddb failed"),
                            category=(err.category if err else "execution"), repairable=False)
        ph = ab_parser.parse_phonon(text, workdir=workdir)
        if not ph.frequencies_thz:
            raise ToolError("anaddb produced no parseable phonon frequencies",
                            category="verification", repairable=False)
        out = {"engine": "abinit", "calc": "anaddb", "converged": True,
               "frequencies_thz": ph.frequencies_thz, "n_modes": ph.n_modes,
               "has_imaginary_modes": ph.has_imaginary,
               "min_frequency_thz": min(ph.frequencies_thz),
               "max_frequency_thz": max(ph.frequencies_thz)}
        out["ngqpt"] = grid
        out["scope"] = "sampled_qmesh_and_path; no non-analytic LO/TO correction"
        curves = {}
        for key, suffix in [("phonon_bands_file", "*PHBST.nc"), ("phonon_dos_file", "*PHDOS.nc")]:
            found = sorted(Path(workdir).glob(suffix))
            if found:
                curves[key] = str(found[0])
        out.update(curves)
        out["stability"] = phonon_stability(
            ph.frequencies_thz,
            imaginary_threshold_thz=ab_writer._as_float(params, "imaginary_threshold_thz", 0.1))
        return out

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _attach_magnetization(self, out: Dict[str, Any], parsed, meta) -> None:
        if meta.get("spin_polarized") or meta.get("magnetic_order"):
            out["spin_polarized"] = True
            out["magnetic_order"] = meta.get("magnetic_order")
            out["total_magnetization_muB"] = parsed.total_magnetization_muB

    def _export_structure(self, workdir, meta, out: Dict[str, Any]) -> None:
        """Export the relaxed structure as CIF from the ABINIT output, when the
        run wrote a final geometry we can read (via ASE's abinit reader)."""
        try:
            from ase.io import write as ase_write
            final_atoms = ab_parser.read_final_atoms(workdir)
        except Exception as exc:
            raise ToolError(f"cannot export final ABINIT geometry: {exc}",
                            category="verification", repairable=False) from exc
        cif_path = Path(workdir) / f"{meta['label']}_relaxed.cif"
        try:
            ase_write(str(cif_path), final_atoms, format="cif")
            out["final_structure_cif"] = str(cif_path)
            from dft_forge.structure_analysis import analyze

            out["analysis"] = analyze(final_atoms)
        except Exception as exc:
            out["analysis_error"] = str(exc)

    def _method_profile(self, meta, params) -> Dict[str, Any]:
        return {
            "engine": "abinit",
            "is_metal": meta.get("is_metal"),
            "species": meta.get("species"),
            "pseudo_sources": meta.get("pseudo_sources"),
            "pseudopotential_sha256": {el: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                       for el, p in meta.get("pseudo_sources", {}).items()},
            "effective": {
                **meta.get("effective_parameters", {}),
                "spin_polarized": meta.get("spin_polarized"),
                "magnetic_order": meta.get("magnetic_order"),
                "supercell": meta.get("supercell"),
            },
        }

    def _stage_density(self, ctx: ExecutionContext, workdir: Path):
        """Symlink an upstream density/WFK into this node's workdir and return
        the upstream structure so nscf/bands/dos run on the same geometry.
        Fails closed when no producer node is present."""
        Path(workdir).mkdir(parents=True, exist_ok=True)
        atoms = None
        for dep_id in ctx.upstream:
            dep_dir = (ctx.base_dir / dep_id).resolve()
            # ABINIT density/WFK files carry an _DEN / _WFK suffix.
            produced = list(dep_dir.glob("*_DEN")) + list(dep_dir.glob("*_DEN.nc"))
            if produced:
                def step(path):
                    m = re.search(r"_TIM(\d+)_", path.name)
                    return int(m.group(1)) if m else -1
                src = max(produced, key=step)
                target = Path(workdir) / "upstream_DEN"
                if not (target.exists() or target.is_symlink()):
                    target.symlink_to(src)
                outputs = ctx.upstream[dep_id]
                cif = outputs.get("final_structure_cif") or outputs.get("input_structure_cif")
                if cif and Path(str(cif)).is_file():
                    try:
                        from ase.io import read as ase_read

                        atoms = ase_read(str(cif))
                    except Exception as exc:
                        raise ToolError(f"cannot read upstream geometry: {exc}", category="missing_upstream", repairable=False) from exc
                if atoms is None:
                    raise ToolError("density producer has no matching structure", category="missing_upstream", repairable=False)
                return atoms
        raise ToolError(
            "no upstream density (_DEN/_WFK) found — add a succeeded scf/relax dependency",
            category="missing_upstream", repairable=False)

    def _resolve_structure(self, ctx, material, structure):
        """Prefer an upstream relaxed CIF; else resolve from material/structure."""
        for outputs in ctx.upstream.values():
            cif = (outputs.get("final_structure_cif") or outputs.get("input_structure_cif")) if isinstance(outputs, dict) else None
            if cif and Path(str(cif)).is_file():
                from ase.io import read as ase_read

                return ase_read(str(cif)), (material or Path(str(cif)).stem)
        atoms, label = ab_writer.resolve_atoms(material, structure)
        return atoms, label

    @staticmethod
    def _find_ddb(workdir: Path) -> Optional[Path]:
        path = Path(workdir) / "phonons_DDB"
        return path if path.is_file() else None

    @staticmethod
    def _read_abo(workdir: Path) -> str:
        abo = ab_parser._find_abo(Path(workdir))
        return abo.read_text(errors="ignore") if abo is not None else ""

    @staticmethod
    def _read_anaddb_out(workdir: Path) -> str:
        matches = sorted(Path(workdir).glob("*anaddb*out*")) + sorted(Path(workdir).glob("*.abo"))
        return matches[0].read_text(errors="ignore") if matches else ""


class ABINITParamRepairer:
    """Conservative repair for recoverable ABINIT failures: raise the
    plane-wave cutoff (ecut) monotonically for convergence-class failures,
    the ABINIT analogue of QE's ecutwfc / VASP's ENCUT bump."""

    name = "abinit_param_bump"

    ECUT_BUMP_STEP = 5.0   # Hartree
    MAX_ECUT = 50.0        # Hartree

    def repair(self, node: NodeRun, error: ToolError) -> bool:
        if node.params.get("_parameters_locked"):
            node.error = (node.error or str(error)) + " | 参数已确认；调整计算参数后请重新预览并确认"
            return False
        params = node.params
        if error.category in ("verification", "electronic_instability", "execution"):
            current = float(params.get("ecut") or 15.0)
            if current >= self.MAX_ECUT:
                return False
            params["ecut"] = min(current + self.ECUT_BUMP_STEP, self.MAX_ECUT)
            node.error = (node.error or "") + f" | repair: ecut -> {params['ecut']}"
            return True
        return False
