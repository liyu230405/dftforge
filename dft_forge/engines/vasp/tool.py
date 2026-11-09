"""VASP graph tool: one graph node = one VASP calculation.

Bridges the graph runtime to the deterministic VASP writer / parser. Like the
QE tool, it never shells out directly — it calls ``executor.run_vasp(workdir)``
so a node runs locally, on SSH/HPC, or against a FakeExecutor in tests without
the graph knowing which.

Calc types: scf, relax (ions), vc-relax (ions + cell).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from dft_forge.engines.vasp import parser as vasp_parser
from dft_forge.engines.vasp import writer as vasp_writer
from dft_forge.engines.vasp.writer import VASPInputError, default_pp_dir
from dft_forge.executor import Executor, FakeExecutor
from dft_forge.runtime.run import ExecutionContext, NodeRun, ToolError

# Calc types that consume a frozen CHGCAR from an upstream scf/relax node.
_NEEDS_CHGCAR = {"nscf", "bands", "dos", "pdos"}


class VASPCalcTool:
    """Graph tool executing a single VASP calculation.

    Calc types mirror the QE engine: scf, relax, vc-relax, nscf, bands, dos,
    pdos, phonon. Unlike QE (one binary per calc), VASP runs the same
    ``vasp_std`` for every type — only INCAR/KPOINTS and staged files differ —
    so a single ``executor.run_vasp(workdir)`` covers all of them.
    """

    name = "vasp"

    def __init__(self, executor: Optional[Executor] = None, pp_dir: Optional[Path] = None):
        self.executor = executor or FakeExecutor()
        self.pp_dir = pp_dir if pp_dir is not None else default_pp_dir()

    def execute(self, node: NodeRun, ctx: ExecutionContext) -> Dict[str, Any]:
        params = node.params
        calc = str(params.get("calc", "scf"))
        material = str(params.get("material", ""))
        structure = params.get("structure")
        workdir = ctx.node_workdir(node.node_id)

        runner = getattr(self.executor, "run_vasp", None)
        if runner is None:
            raise ToolError(
                "configured executor does not support VASP (no run_vasp)",
                category="config",
                repairable=False,
            )

        # Phonopy finite-displacement dispersion: two dedicated calc types that
        # do not follow the single-run flow. phonopy_disp fans out over many
        # displaced supercells internally; phonopy_bands runs no VASP at all.
        if calc == "phonopy_disp":
            return self._run_phonopy_disp(node, ctx, workdir, runner, material, structure)
        if calc == "phonopy_bands":
            return self._run_phonopy_bands(node, ctx, workdir, material)

        # bands/dos/pdos/nscf reuse the upstream self-consistent charge density,
        # the VASP analogue of QE linking an upstream {prefix}.save directory.
        upstream_atoms = None
        if calc in _NEEDS_CHGCAR:
            upstream_atoms = self._stage_chgcar(ctx, workdir)

        try:
            meta = vasp_writer.write_inputs(
                material, structure, calc, params, workdir,
                pp_dir=self.pp_dir, atoms=upstream_atoms,
            )
        except VASPInputError as exc:
            raise ToolError(str(exc), category="config", repairable=False) from exc

        result = runner(workdir, timeout=ctx.timeout_seconds)
        stdout = result.stdout or ""
        error = vasp_parser.classify_error(stdout) or vasp_parser.classify_error(
            self._read_outcar(workdir)
        )
        if not result.success:
            raise ToolError(
                (error.message if error else result.error_message or "VASP run failed"),
                category=(error.category if error else "execution"),
                repairable=bool(error and error.category in {"charge_mixing", "ionic_minimizer"}),
            )

        dispatch = {
            "scf": self._finish_scf, "relax": self._finish_relax,
            "vc-relax": self._finish_relax, "nscf": self._finish_nscf,
            "bands": self._finish_bands, "dos": self._finish_dos,
            "pdos": self._finish_pdos, "phonon": self._finish_phonon,
        }
        handler = dispatch.get(calc)
        if handler is None:
            raise ToolError(f"unknown vasp calc type '{calc}'", category="config", repairable=False)
        outputs = handler(workdir, stdout, meta, params)
        outputs.update({"engine": "vasp", "calc": calc,
                        "natoms": meta["natoms"], "species": meta["species"],
                        "stdout_file": str(Path(workdir) / "OUTCAR")})
        outputs["method_profile"] = self._method_profile(meta, params)
        return outputs

    # ── Per-calc finishers ────────────────────────────────────────────────────

    def _finish_scf(self, workdir, stdout, meta, params) -> Dict[str, Any]:
        parsed = vasp_parser.parse_result(workdir, stdout)
        if not parsed.converged:
            raise ToolError("VASP scf did not reach electronic convergence",
                            category="verification", repairable=True)
        out = {"energy_ev": parsed.total_energy_ev, "converged": True,
               "fermi_energy_ev": parsed.fermi_energy_ev,
               "n_electronic_steps": parsed.n_electronic_steps}
        self._attach_magnetization(out, parsed, meta)
        return out

    def _finish_relax(self, workdir, stdout, meta, params) -> Dict[str, Any]:
        parsed = vasp_parser.parse_result(workdir, stdout)
        if not parsed.converged:
            raise ToolError("VASP relaxation did not converge electronically",
                            category="verification", repairable=True)
        threshold = abs(vasp_writer._as_float(params, "ediffg", -0.02))
        if parsed.max_force_ev_ang and parsed.max_force_ev_ang > threshold + 1e-6:
            raise ToolError(
                f"VASP relaxation force {parsed.max_force_ev_ang:.3f} > "
                f"{threshold:.3f} eV/Å; not converged geometrically",
                category="verification", repairable=True)
        out = {"energy_ev": parsed.total_energy_ev, "converged": True,
               "fermi_energy_ev": parsed.fermi_energy_ev,
               "final_volume_ang3": parsed.final_volume_ang3,
               "max_force_ev_ang": parsed.max_force_ev_ang,
               "n_ionic_steps": parsed.n_ionic_steps}
        self._attach_magnetization(out, parsed, meta)
        self._export_structure(workdir, meta, out)
        return out

    def _finish_nscf(self, workdir, stdout, meta, params) -> Dict[str, Any]:
        parsed = vasp_parser.parse_result(workdir, stdout)
        return {"energy_ev": parsed.total_energy_ev or None,
                "converged": True, "fermi_energy_ev": parsed.fermi_energy_ev}

    def _finish_bands(self, workdir, stdout, meta, params) -> Dict[str, Any]:
        bands = vasp_parser.parse_bands(workdir)
        if bands.n_kpoints == 0:
            raise ToolError("bands run produced no eigenvalues (EIGENVAL/vasprun.xml)",
                            category="verification", repairable=False)
        return {"converged": True, "n_bands": bands.n_bands,
                "n_kpoints": bands.n_kpoints, "band_gap_ev": bands.band_gap_ev,
                "is_metal": bands.is_metal, "fermi_energy_ev": bands.fermi_energy_ev,
                "vbm_ev": bands.vbm_ev, "cbm_ev": bands.cbm_ev}

    def _finish_dos(self, workdir, stdout, meta, params) -> Dict[str, Any]:
        dos = vasp_parser.parse_dos(workdir, projected=False)
        if dos.n_energy_points == 0:
            raise ToolError("dos run produced no DOSCAR data",
                            category="verification", repairable=False)
        out = {"converged": True, "n_energy_points": dos.n_energy_points,
               "fermi_energy_ev": dos.fermi_energy_ev, "dos_at_fermi": dos.dos_at_fermi}
        if dos.energies_ev is not None and dos.dos is not None:
            out["dos_curve"] = {"energies_ev": [round(e, 4) for e in dos.energies_ev],
                                "dos": [round(d, 3) for d in dos.dos]}
        return out

    def _finish_pdos(self, workdir, stdout, meta, params) -> Dict[str, Any]:
        dos = vasp_parser.parse_dos(workdir, projected=True)
        if dos.n_energy_points == 0:
            raise ToolError("pdos run produced no DOSCAR data",
                            category="verification", repairable=False)
        if not dos.element_dos:
            raise ToolError("pdos parse recovered no element-projected DOS "
                            "(need LORBIT=11 and per-ion DOSCAR blocks)",
                            category="verification", repairable=False)
        out = {"converged": True, "n_energy_points": dos.n_energy_points,
               "fermi_energy_ev": dos.fermi_energy_ev,
               "elements": sorted(dos.element_dos.keys())}
        if dos.energies_ev is not None:
            out["pdos_curve"] = {
                "energies_ev": [round(e, 4) for e in dos.energies_ev],
                "element_dos": {el: [round(v, 3) for v in arr]
                                for el, arr in sorted(dos.element_dos.items())}}
        return out

    def _finish_phonon(self, workdir, stdout, meta, params) -> Dict[str, Any]:
        text = self._read_outcar(workdir) or stdout
        ph = vasp_parser.parse_phonon(text)
        if not ph.frequencies_thz:
            raise ToolError("phonon run produced no parseable frequencies (IBRION=8 OUTCAR)",
                            category="verification", repairable=False)
        return {"converged": True, "frequencies_thz": ph.frequencies_thz,
                "n_modes": ph.n_modes, "has_imaginary_modes": ph.has_imaginary}

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _attach_magnetization(self, out: Dict[str, Any], parsed, meta) -> None:
        if meta.get("spin_polarized") or meta.get("magnetic_order"):
            out["spin_polarized"] = True
            out["magnetic_order"] = meta.get("magnetic_order")
            out["total_magnetization_muB"] = parsed.total_magnetization_muB

    def _export_structure(self, workdir, meta, out: Dict[str, Any]) -> None:
        """Read the relaxed CONTCAR, export CIF, and attach a geometry report."""
        contcar = Path(workdir) / "CONTCAR"
        if not contcar.is_file() or contcar.stat().st_size == 0:
            return
        try:
            from ase.io import read as ase_read, write as ase_write

            final_atoms = ase_read(str(contcar), format="vasp")
        except Exception:
            return
        cif_path = Path(workdir) / f"{meta['label']}_relaxed.cif"
        try:
            ase_write(str(cif_path), final_atoms, format="cif")
            out["final_structure_cif"] = str(cif_path)
            from dft_forge.structure_analysis import analyze

            out["analysis"] = analyze(final_atoms)
        except Exception as exc:
            out["analysis_error"] = str(exc)

    def _method_profile(self, meta, params) -> Dict[str, Any]:
        """Record the effective method so downstream steps can cite provenance."""
        return {
            "engine": "vasp",
            "is_metal": meta.get("is_metal"),
            "species": meta.get("species"),
            "potcar_sources": meta.get("potcar_sources"),
            "effective": {
                "encut": params.get("encut"),
                "kpoints": params.get("kpoints"),
                "kspacing": params.get("kspacing"),
                "ediff": params.get("ediff"),
                "ediffg": params.get("ediffg"),
                "sigma": params.get("sigma"),
                "spin_polarized": meta.get("spin_polarized"),
                "magnetic_order": meta.get("magnetic_order"),
                "tot_charge": meta.get("tot_charge"),
                "nelect": meta.get("nelect"),
                "supercell": meta.get("supercell"),
            },
        }

    def _stage_chgcar(self, ctx: ExecutionContext, workdir: Path):
        """Symlink an upstream CHGCAR/CONTCAR into this node's workdir.

        Returns the upstream relaxed structure (from CONTCAR) so nscf/bands/dos
        run on the same geometry the charge density was computed for. Fails
        closed when no producer node is present, mirroring QE's save-dir check.
        """
        Path(workdir).mkdir(parents=True, exist_ok=True)
        atoms = None
        for dep_id in ctx.upstream:
            dep_dir = (ctx.base_dir / dep_id).resolve()
            chgcar = dep_dir / "CHGCAR"
            if chgcar.exists():
                target = Path(workdir) / "CHGCAR"
                if not (target.exists() or target.is_symlink()):
                    target.symlink_to(chgcar)
                # prefer the relaxed CONTCAR geometry, else the input POSCAR
                for name in ("CONTCAR", "POSCAR"):
                    src = dep_dir / name
                    if src.is_file() and src.stat().st_size > 0:
                        try:
                            from ase.io import read as ase_read

                            atoms = ase_read(str(src), format="vasp")
                        except Exception:
                            atoms = None
                        break
                return atoms
        raise ToolError(
            "no upstream CHGCAR found — add a succeeded scf/relax dependency",
            category="missing_upstream", repairable=False)

    # ── Phonopy finite-displacement dispersion ────────────────────────────────

    def _resolve_phonon_structure(self, ctx, material, structure):
        """Prefer an upstream relaxed CONTCAR/CIF; else the given material."""
        for outputs in ctx.upstream.values():
            cif = outputs.get("final_structure_cif") if isinstance(outputs, dict) else None
            if cif and Path(str(cif)).is_file():
                from ase.io import read as ase_read

                return ase_read(str(cif)), (material or Path(str(cif)).stem)
        atoms, label = vasp_writer.resolve_atoms(material, structure)
        return atoms, label

    def _run_phonopy_disp(self, node, ctx, workdir, runner, material, structure):
        """Build displaced supercells, run one SCF force calc per displacement,
        collect FORCE_SETS, and persist the phonopy force constants."""
        from dft_forge.engines.vasp import phonopy_support as ph

        params = node.params
        Path(workdir).mkdir(parents=True, exist_ok=True)
        atoms, label = self._resolve_phonon_structure(ctx, material, structure)
        atoms = vasp_writer.apply_supercell(atoms, params.get("supercell"))
        sc = params.get("phonon_supercell") or [2, 2, 2]
        matrix = [[int(sc[0]), 0, 0], [0, int(sc[1]), 0], [0, 0, int(sc[2])]]
        distance = vasp_writer._as_float(params, "displacement_distance", 0.01)

        try:
            phonon = ph.build_phonopy(atoms, matrix, distance=distance)
            displaced = ph.displacement_atoms(phonon)
        except ph.PhonopyUnavailable as exc:
            raise ToolError(str(exc), category="config", repairable=False) from exc

        if not displaced:
            raise ToolError("phonopy generated no displacements (check supercell/symmetry)",
                            category="verification", repairable=False)

        # The phonon supercell is already baked into each displaced cell; drop
        # the user "supercell" key so write_inputs does not multiply it again.
        disp_params = {k: v for k, v in params.items() if k != "supercell"}
        force_sets: list = []
        for idx, disp_atoms in enumerate(displaced):
            sub = Path(workdir) / f"disp-{idx + 1:03d}"
            try:
                vasp_writer.write_inputs(
                    label, None, "scf", {**disp_params, "prefix": f"{label}_disp{idx + 1}"},
                    sub, pp_dir=self.pp_dir, atoms=disp_atoms,
                )
            except VASPInputError as exc:
                raise ToolError(str(exc), category="config", repairable=False) from exc
            result = runner(sub, timeout=ctx.timeout_seconds)
            if not result.success:
                err = vasp_parser.classify_error(result.stdout or "") or \
                    vasp_parser.classify_error(self._read_outcar(sub))
                raise ToolError(
                    f"displacement {idx + 1}/{len(displaced)} failed: "
                    + (err.message if err else result.error_message or "VASP run failed"),
                    category=(err.category if err else "execution"),
                    repairable=bool(err and err.category in {"charge_mixing", "ionic_minimizer"}),
                )
            forces = vasp_parser.parse_forces(sub)
            if forces is None:
                raise ToolError(f"displacement {idx + 1} produced no parseable forces",
                                category="verification", repairable=False)
            force_sets.append(forces)

        try:
            ph.set_forces_and_produce(phonon, force_sets)
            yaml_path = Path(workdir) / "phonopy_params.yaml"
            ph.save(phonon, yaml_path)
        except ph.PhonopyUnavailable as exc:
            raise ToolError(str(exc), category="config", repairable=False) from exc

        return {
            "engine": "vasp", "calc": "phonopy_disp", "converged": True,
            "n_displacements": len(displaced),
            "phonon_supercell": [int(x) for x in sc],
            "displacement_distance": distance,
            "phonopy_yaml": str(yaml_path),
            "natoms": len(atoms), "species": vasp_writer.poscar_species_order(atoms),
        }

    def _run_phonopy_bands(self, node, ctx, workdir, material):
        """Reload force constants and compute dispersion + DOS along the k-path."""
        from dft_forge.engines.vasp import phonopy_support as ph
        from dft_forge.workflows import phonon_stability

        params = node.params
        Path(workdir).mkdir(parents=True, exist_ok=True)
        yaml_path = None
        atoms = None
        for outputs in ctx.upstream.values():
            if isinstance(outputs, dict) and outputs.get("phonopy_yaml"):
                yaml_path = outputs["phonopy_yaml"]
                break
        if not yaml_path or not Path(str(yaml_path)).is_file():
            raise ToolError("no upstream phonopy_params.yaml — add a phonopy_disp dependency",
                            category="missing_upstream", repairable=False)

        atoms, _ = self._resolve_phonon_structure(ctx, material, params.get("structure"))
        # Derive the standard high-symmetry q-path from the primitive cell.
        pbc = tuple(bool(b) for b in atoms.pbc) if len(atoms.pbc) == 3 else None
        bandpath = atoms.cell.bandpath(pbc=pbc)
        special = bandpath.special_points
        path_str = bandpath.path.replace(",", "")
        seg = [special[ch].tolist() for ch in path_str if ch in special]
        labels = ["Γ" if ch.upper() in ("G", "GAMMA") else ch for ch in path_str if ch in special]
        if len(seg) < 2:
            raise ToolError("could not derive a q-path for the phonon dispersion",
                            category="verification", repairable=False)

        try:
            phonon = ph.load(Path(str(yaml_path)))
            bands = ph.band_structure(phonon, [seg], labels,
                                      npoints=vasp_writer._as_int(params, "nqpoints", 51))
            mesh = params.get("dos_mesh") or [8, 8, 8]
            dos = ph.total_dos(phonon, [int(x) for x in mesh])
        except ph.PhonopyUnavailable as exc:
            raise ToolError(str(exc), category="config", repairable=False) from exc

        out = {
            "engine": "vasp", "calc": "phonopy_bands", "converged": True,
            "labels": bands["labels"],
            "min_frequency_thz": bands["min_frequency_thz"],
            "max_frequency_thz": bands["max_frequency_thz"],
            "phonon_dos": dos,
            "dispersion": {"distances": bands["distances"],
                           "frequencies_thz": bands["frequencies_thz"]},
        }
        if bands["all_frequencies_thz"]:
            out["stability"] = phonon_stability(
                bands["all_frequencies_thz"],
                imaginary_threshold_thz=vasp_writer._as_float(params, "imaginary_threshold_thz", 0.1),
            )
        return out

    @staticmethod
    def _read_outcar(workdir: Path) -> str:
        outcar = Path(workdir) / "OUTCAR"
        return outcar.read_text(errors="ignore") if outcar.is_file() else ""


class VASPParamRepairer:
    """Conservative repair for recoverable VASP failures.

    charge_mixing  -> lower mixing (add AMIX/BMIX hints via params)
    ionic_minimizer -> switch IBRION to damped MD-style relaxation fallback
    """

    name = "vasp_param_bump"

    ENCUT_BUMP_STEP = 40.0
    MAX_ENCUT = 800.0

    def repair(self, node: NodeRun, error: ToolError) -> bool:
        if node.params.get("_parameters_locked"):
            node.error = (node.error or str(error)) + " | 参数已确认；调整计算参数后请重新预览并确认"
            return False
        params = node.params
        if error.category == "charge_mixing" and not params.get("_amix_bumped"):
            params["_amix_bumped"] = True
            params["sigma"] = min(float(params.get("sigma") or 0.2) * 1.5, 0.5)
            return True
        if error.category == "ionic_minimizer" and not params.get("_ibrion_bumped"):
            params["_ibrion_bumped"] = True
            # A fresh restart from the last geometry typically clears a ZBRENT
            # bracketing failure; the writer re-emits inputs on the next attempt.
            return True
        # Convergence-class failures: raise the basis-set cutoff monotonically,
        # the VASP analogue of QE's ecutwfc bump.
        if error.category in ("verification", "electronic_instability", "execution"):
            current = float(params.get("encut") or 520.0)
            if current >= self.MAX_ENCUT:
                return False
            params["encut"] = min(current + self.ENCUT_BUMP_STEP, self.MAX_ENCUT)
            node.error = (node.error or "") + f" | repair: encut -> {params['encut']}"
            return True
        return False
