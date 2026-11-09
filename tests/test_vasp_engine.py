"""Tests for the VASP engine: writer, parser, tool, and graph integration.

No real VASP binary is required — the FakeExecutor injects OUTCAR/vasprun.xml
just as the QE engine tests inject QE stdout. POTCAR handling is exercised with
temporary fake pseudopotentials; nothing licensed is bundled.
"""

from pathlib import Path

import pytest

from dft_forge.engines.vasp import parser as vp
from dft_forge.engines.vasp import writer as vw
from dft_forge.engines.vasp.tool import VASPCalcTool, VASPParamRepairer
from dft_forge.engines.vasp.writer import VASPInputError
from dft_forge.executor import FakeExecutor
from dft_forge.runtime.run import ExecutionContext, NodeRun, ToolError

CONVERGED_OUTCAR = """
 free  energy   TOTEN  =       -10.8400 eV
 reached required accuracy - stopping structural energy minimisation
 E-fermi :   5.4200
 volume of cell :      40.88
"""


def _fake_pp(tmp_path: Path, *elements: str) -> Path:
    pp = tmp_path / "pp"
    for el in elements:
        (pp / el).mkdir(parents=True)
        (pp / el / "POTCAR").write_text(f"PAW_PBE {el} fake\n")
    return pp


class _OutcarExecutor(FakeExecutor):
    """FakeExecutor that also drops a converged OUTCAR into the workdir."""

    def __init__(self, outcar: str = CONVERGED_OUTCAR, mode: str = "success"):
        super().__init__(mode=mode, stdout=outcar)
        self._outcar = outcar

    def run_vasp(self, workdir, *, timeout=None):
        if self.mode != "failure":
            (Path(workdir) / "OUTCAR").write_text(self._outcar)
        return super().run_vasp(workdir, timeout=timeout)


# ── Writer ──────────────────────────────────────────────────────────────────

def test_incar_scf_is_static_and_insulator_smearing():
    incar = vw.build_incar("scf", {"prefix": "Si"}, is_metal=False)
    assert "IBRION = -1" in incar and "NSW = 0" in incar
    assert "ISMEAR = 0" in incar  # insulator/semiconductor

def test_incar_vc_relax_relaxes_cell_and_metal_smearing():
    incar = vw.build_incar("vc-relax", {"prefix": "Al"}, is_metal=True)
    assert "ISIF = 3" in incar and "NSW = 100" in incar
    assert "ISMEAR = 1" in incar and "EDIFFG" in incar

def test_incar_rejects_unknown_calc():
    with pytest.raises(VASPInputError):
        vw.build_incar("gw", {}, is_metal=False)

def test_kpoints_explicit_vs_auto():
    assert "8 8 8" in vw.build_kpoints({"kpoints": [8, 8, 8]})
    assert "Auto" in vw.build_kpoints({})

def test_kpoints_rejects_bad_grid():
    with pytest.raises(VASPInputError):
        vw.build_kpoints({"kpoints": [8, 0, 8]})

def test_write_poscar_returns_species_order(tmp_path):
    atoms, _ = vw.resolve_atoms("Si", None)
    species = vw.write_poscar(atoms, tmp_path / "POSCAR")
    assert species == ["Si"]
    assert (tmp_path / "POSCAR").is_file()

def test_potcar_fail_closed_without_dir():
    with pytest.raises(VASPInputError, match="license-restricted"):
        vw.resolve_potcars(["Si"], None)

def test_potcar_fail_closed_missing_element(tmp_path):
    pp = _fake_pp(tmp_path, "Si")
    with pytest.raises(VASPInputError, match="missing POTCAR for: O"):
        vw.resolve_potcars(["Si", "O"], pp)

def test_potcar_concatenation_order(tmp_path):
    pp = _fake_pp(tmp_path, "Ga", "As")
    out = tmp_path / "POTCAR"
    prov = vw.write_potcar(["Ga", "As"], pp, out)
    text = out.read_text()
    assert text.index("Ga") < text.index("As")  # POSCAR order preserved
    assert set(prov) == {"Ga", "As"}

def test_potcar_variant_folders(tmp_path):
    pp = tmp_path / "pp"
    (pp / "Fe_pv").mkdir(parents=True)
    (pp / "Fe_pv" / "POTCAR").write_text("PAW_PBE Fe_pv\n")
    resolved = vw.resolve_potcars(["Fe"], pp)
    assert resolved[0].parent.name == "Fe_pv"

def test_write_inputs_produces_four_files(tmp_path):
    pp = _fake_pp(tmp_path, "Si")
    meta = vw.write_inputs("Si", None, "scf", {}, tmp_path / "run", pp_dir=pp)
    for name in ("INCAR", "POSCAR", "KPOINTS", "POTCAR"):
        assert (tmp_path / "run" / name).is_file()
    assert meta["species"] == ["Si"] and meta["natoms"] == 2


# ── Parser ──────────────────────────────────────────────────────────────────

def test_parse_outcar_energy_and_convergence():
    parsed = vp.parse_outcar(CONVERGED_OUTCAR)
    assert parsed.total_energy_ev == -10.84
    assert parsed.converged is True
    assert parsed.fermi_energy_ev == 5.42
    assert parsed.final_volume_ang3 == 40.88

def test_parse_outcar_unconverged():
    parsed = vp.parse_outcar(" free  energy   TOTEN  =  -10.0 eV\n")
    assert parsed.converged is False

def test_parse_vasprun(tmp_path):
    xml = tmp_path / "vasprun.xml"
    xml.write_text(
        "<modeling><parameters><i name='NELM'>60</i></parameters>"
        "<calculation>"
        "<scstep><energy><i name='e_fr_energy'>-10.5</i></energy></scstep>"
        "<scstep><energy><i name='e_fr_energy'>-10.84</i></energy></scstep>"
        "<energy><i name='e_fr_energy'>-10.84</i></energy>"
        "<structure><crystal><i name='volume'>40.88</i></crystal></structure>"
        "<dos><i name='efermi'>5.42</i></dos>"
        "</calculation></modeling>"
    )
    parsed = vp.parse_vasprun(xml)
    assert parsed.total_energy_ev == -10.84
    assert parsed.fermi_energy_ev == 5.42
    assert parsed.converged is True

def test_parse_result_prefers_vasprun(tmp_path):
    (tmp_path / "OUTCAR").write_text(" free  energy   TOTEN  =  -1.0 eV\n")
    (tmp_path / "vasprun.xml").write_text(
        "<modeling><calculation><energy><i name='e_fr_energy'>-10.84</i></energy>"
        "</calculation></modeling>"
    )
    parsed = vp.parse_result(tmp_path)
    assert parsed.total_energy_ev == -10.84  # vasprun wins over OUTCAR

def test_parse_result_falls_back_to_outcar_on_bad_xml(tmp_path):
    (tmp_path / "OUTCAR").write_text(CONVERGED_OUTCAR)
    (tmp_path / "vasprun.xml").write_text("<modeling><calculation>")  # truncated
    parsed = vp.parse_result(tmp_path)
    assert parsed.total_energy_ev == -10.84

@pytest.mark.parametrize("token,category", [
    ("ZBRENT: fatal error", "ionic_minimizer"),
    ("BRMIX: problems", "charge_mixing"),
    ("WARNING: Sub-Space-Matrix", "electronic_instability"),
    ("EDDDAV failure", "electronic_instability"),
])
def test_classify_error(token, category):
    assert vp.classify_error(token).category == category

def test_classify_error_clean():
    assert vp.classify_error("all good") is None


# ── Tool ──────────────────────────────────────────────────────────────────

def _ctx(tmp_path):
    return ExecutionContext(base_dir=tmp_path, run_id="r1")

def test_tool_scf_success(tmp_path):
    pp = _fake_pp(tmp_path, "Si")
    tool = VASPCalcTool(executor=_OutcarExecutor(), pp_dir=pp)
    node = NodeRun(node_id="scf", params={"calc": "scf", "material": "Si"})
    out = tool.execute(node, _ctx(tmp_path))
    assert out["engine"] == "vasp"
    assert out["energy_ev"] == -10.84
    assert out["converged"] is True

def test_tool_missing_potcar_is_config_error(tmp_path):
    tool = VASPCalcTool(executor=_OutcarExecutor(), pp_dir=tmp_path / "empty")
    node = NodeRun(node_id="scf", params={"calc": "scf", "material": "Si"})
    with pytest.raises(ToolError) as exc:
        tool.execute(node, _ctx(tmp_path))
    assert exc.value.category == "config"

def test_tool_execution_failure_raises(tmp_path):
    pp = _fake_pp(tmp_path, "Si")
    tool = VASPCalcTool(executor=_OutcarExecutor(mode="failure"), pp_dir=pp)
    node = NodeRun(node_id="scf", params={"calc": "scf", "material": "Si"})
    with pytest.raises(ToolError):
        tool.execute(node, _ctx(tmp_path))

def test_tool_unconverged_raises_verification(tmp_path):
    pp = _fake_pp(tmp_path, "Si")
    tool = VASPCalcTool(
        executor=_OutcarExecutor(outcar=" free  energy   TOTEN  =  -1.0 eV\n"),
        pp_dir=pp,
    )
    node = NodeRun(node_id="scf", params={"calc": "scf", "material": "Si"})
    with pytest.raises(ToolError) as exc:
        tool.execute(node, _ctx(tmp_path))
    assert exc.value.category == "verification"

def test_tool_requires_run_vasp_capable_executor(tmp_path):
    pp = _fake_pp(tmp_path, "Si")
    class NoVasp(FakeExecutor):
        run_vasp = None
    tool = VASPCalcTool(executor=NoVasp(), pp_dir=pp)
    node = NodeRun(node_id="scf", params={"calc": "scf", "material": "Si"})
    with pytest.raises(ToolError, match="run_vasp"):
        tool.execute(node, _ctx(tmp_path))


# ── Repairer ────────────────────────────────────────────────────────────────

def test_repairer_charge_mixing_bumps_once():
    r = VASPParamRepairer()
    node = NodeRun(node_id="scf", params={"calc": "scf"})
    err = ToolError("brmix", category="charge_mixing", repairable=True)
    assert r.repair(node, err) is True
    assert node.params["_amix_bumped"] is True
    assert r.repair(node, err) is False  # only once

def test_repairer_ignores_unknown_category():
    r = VASPParamRepairer()
    node = NodeRun(node_id="scf", params={})
    assert r.repair(node, ToolError("x", category="config")) is False


# ── Graph integration ────────────────────────────────────────────────────────

def test_vasp_template_runs_through_graph(tmp_path):
    from dft_forge.runtime.engine import GraphEngine

    pp = _fake_pp(tmp_path, "Si")
    engine = GraphEngine(
        base_dir=tmp_path / "runs",
        tools={"vasp": VASPCalcTool(executor=_OutcarExecutor(), pp_dir=pp)},
        templates_dir=Path(__file__).resolve().parents[1] / "templates",
    )
    final = engine.run_template("v0_vasp_scf", inputs={"material": "Si"})
    assert final.state.value == "succeeded"
    assert final.nodes["scf"].outputs["energy_ev"] == -10.84


# ── Writer: new calc modes / magnetism / doping ───────────────────────────────

def test_incar_nscf_freezes_charge():
    incar = vw.build_incar("nscf", {}, is_metal=False)
    assert "ICHARG = 11" in incar

def test_incar_dos_uses_tetrahedron_and_lorbit():
    incar = vw.build_incar("dos", {}, is_metal=True)
    assert "ISMEAR = -5" in incar and "LORBIT = 11" in incar and "NEDOS" in incar

def test_incar_phonon_is_dfpt():
    incar = vw.build_incar("phonon", {}, is_metal=False)
    assert "IBRION = 8" in incar

def test_incar_scf_writes_chgcar_for_downstream():
    assert "LCHARG = .TRUE." in vw.build_incar("scf", {}, is_metal=False)
    assert "LCHARG = .FALSE." in vw.build_incar("nscf", {}, is_metal=False)

def test_incar_magmom_fm(tmp_path):
    atoms, _ = vw.resolve_atoms("Fe", None)
    incar = vw.build_incar("scf", {"magnetic_order": "FM"}, is_metal=True,
                           atoms=atoms, species=vw.poscar_species_order(atoms))
    assert "ISPIN = 2" in incar and "MAGMOM" in incar

def test_incar_magmom_afm_alternates_sign(tmp_path):
    from ase import Atoms as _A
    # two Fe atoms so AFM can split them
    atoms = _A("Fe2", positions=[[0, 0, 0], [1.4, 1.4, 1.4]], cell=[2.8, 2.8, 2.8], pbc=True)
    incar = vw.build_incar("scf", {"magnetic_order": "AFM", "magnetic_element": "Fe",
                                   "initial_moment_muB": 3.0}, is_metal=True,
                           atoms=atoms, species=["Fe"])
    mag = [l for l in incar.splitlines() if l.startswith("MAGMOM")][0]
    assert "3" in mag and "-3" in mag

def test_incar_nelect_doping():
    incar = vw.build_incar("scf", {}, is_metal=False, nelect=7.5)
    assert "NELECT = 7.5" in incar

def test_band_kpoints_line_mode(tmp_path):
    atoms, _ = vw.resolve_atoms("Si", None)
    kpoints = vw.build_kpoints({"nkpoints_bands": 30}, calc="bands", atoms=atoms)
    assert "Line-mode" in kpoints and "Reciprocal" in kpoints

def test_band_kpoints_needs_structure():
    with pytest.raises(VASPInputError, match="k-path"):
        vw.build_kpoints({}, calc="bands", atoms=None)

def test_zval_from_potcar(tmp_path):
    p = tmp_path / "POTCAR"
    p.write_text("PAW_PBE Si\n   ZVAL   =    4.000\nend\nPAW_PBE O\n ZVAL = 6.000\n")
    assert vw.zval_from_potcar(p) == 10.0

def test_write_inputs_charged_cell_sets_nelect(tmp_path):
    pp = tmp_path / "pp"
    (pp / "Si").mkdir(parents=True)
    (pp / "Si" / "POTCAR").write_text("PAW_PBE Si\n ZVAL = 4.000\n")
    meta = vw.write_inputs("Si", None, "scf", {"tot_charge": 1.0}, tmp_path / "run", pp_dir=pp)
    # Si has 2 atoms × 4 ZVAL = 8 neutral; +1 charge removes one electron -> 7
    assert meta["nelect"] == 7.0
    assert "NELECT = 7" in (tmp_path / "run" / "INCAR").read_text()

def test_write_inputs_supercell_multiplies_atoms(tmp_path):
    pp = _fake_pp(tmp_path, "Si")
    meta = vw.write_inputs("Si", None, "scf", {"supercell": "2x1x1"}, tmp_path / "run", pp_dir=pp)
    assert meta["natoms"] == 4  # 2-atom cell × 2


# ── Parser: bands / dos / pdos / phonon / forces ──────────────────────────────

def test_parse_outcar_forces_and_magnetization():
    text = (
        " free  energy   TOTEN  =  -10.0 eV\n reached required accuracy\n"
        " POSITION                   TOTAL-FORCE (eV/Angst)\n"
        " -----------------------------------------------------------\n"
        "  0.0 0.0 0.0   0.01 0.00 0.02\n"
        "  1.0 1.0 1.0  -0.30 0.00 0.00\n"
        " -----------------------------------------------------------\n"
        " number of electron   8.0000 magnetization    2.1500\n"
    )
    parsed = vp.parse_outcar(text)
    assert abs(parsed.max_force_ev_ang - 0.30) < 1e-6
    assert parsed.total_magnetization_muB == 2.15

def test_parse_bands_from_vasprun(tmp_path):
    (tmp_path / "vasprun.xml").write_text(
        "<modeling><calculation><dos><i name='efermi'>0.0</i></dos>"
        "<eigenvalues><array><set><set>"
        "<set comment='kpoint 1'><r>-2.0 1.0</r><r>-1.0 1.0</r><r>1.5 0.0</r></set>"
        "<set comment='kpoint 2'><r>-2.1 1.0</r><r>-0.9 1.0</r><r>1.4 0.0</r></set>"
        "</set></set></array></eigenvalues>"
        "</calculation></modeling>"
    )
    bands = vp.parse_bands(tmp_path)
    assert bands.n_kpoints == 2 and bands.n_bands == 3
    assert bands.is_metal is False
    assert abs(bands.band_gap_ev - 2.3) < 1e-6  # cbm 1.4 - vbm -0.9

def test_parse_bands_metal(tmp_path):
    (tmp_path / "vasprun.xml").write_text(
        "<modeling><calculation><dos><i name='efermi'>0.0</i></dos>"
        "<eigenvalues><array><set><set>"
        "<set><r>-0.02 1.0</r><r>0.03 0.0</r></set>"
        "</set></set></array></eigenvalues></calculation></modeling>"
    )
    assert vp.parse_bands(tmp_path).is_metal is True

def test_parse_dos_total(tmp_path):
    lines = ["1 1 1 1", "hdr", "hdr", "hdr", "hdr", "0 0 3 0.5 0"]
    lines += ["-1.0 0.5 0.5", "0.5 2.0 1.0", "2.0 0.1 3.0"]
    (tmp_path / "DOSCAR").write_text("\n".join(lines) + "\n")
    dos = vp.parse_dos(tmp_path)
    assert dos.n_energy_points == 3
    assert dos.fermi_energy_ev == 0.5
    assert dos.energies_ev[1] == 0.5

def test_parse_phonon_frequencies():
    text = (
        "   1 f  =   12.345678 THz   77.5 2piTHz\n"
        "   2 f  =   12.000000 THz\n"
        "   3 f/i=    0.500000 THz   (imaginary)\n"
    )
    ph = vp.parse_phonon(text)
    assert ph.n_modes == 3 and ph.has_imaginary is True
    assert -0.5 in ph.frequencies_thz


# ── Tool: relax export / bands / dos / pdos / phonon / chgcar staging ─────────

def _relax_outcar():
    return (
        " free  energy   TOTEN  =  -10.84 eV\n reached required accuracy\n"
        " POSITION                   TOTAL-FORCE (eV/Angst)\n"
        " -----------------------------------------------------------\n"
        "  0.0 0.0 0.0   0.005 0.000 0.001\n"
        " -----------------------------------------------------------\n"
        " volume of cell :      40.88\n"
    )

def test_tool_relax_exports_structure(tmp_path):
    pp = _fake_pp(tmp_path, "Si")
    class _RelaxExec(FakeExecutor):
        def __init__(self):
            super().__init__(mode="success", stdout=_relax_outcar())
        def run_vasp(self, workdir, *, timeout=None):
            wd = Path(workdir)
            (wd / "OUTCAR").write_text(_relax_outcar())
            # a minimal valid CONTCAR (Si diamond primitive-ish)
            (wd / "CONTCAR").write_text(
                "Si\n1.0\n 3.87 0 0\n 0 3.87 0\n 0 0 3.87\nSi\n2\nDirect\n"
                " 0.0 0.0 0.0\n 0.25 0.25 0.25\n"
            )
            return super().run_vasp(workdir, timeout=timeout)
    tool = VASPCalcTool(executor=_RelaxExec(), pp_dir=pp)
    node = NodeRun(node_id="vc", params={"calc": "vc-relax", "material": "Si"})
    out = tool.execute(node, _ctx(tmp_path))
    assert out["max_force_ev_ang"] < 0.02
    assert "final_structure_cif" in out and Path(out["final_structure_cif"]).is_file()
    assert "method_profile" in out and out["method_profile"]["engine"] == "vasp"

def test_tool_relax_rejects_high_force(tmp_path):
    pp = _fake_pp(tmp_path, "Si")
    bad = _relax_outcar().replace("0.005 0.000 0.001", "0.5 0.0 0.0")
    class _BadExec(FakeExecutor):
        def __init__(self):
            super().__init__(mode="success", stdout=bad)
        def run_vasp(self, workdir, *, timeout=None):
            (Path(workdir) / "OUTCAR").write_text(bad)
            return super().run_vasp(workdir, timeout=timeout)
    tool = VASPCalcTool(executor=_BadExec(), pp_dir=pp)
    node = NodeRun(node_id="vc", params={"calc": "vc-relax", "material": "Si", "ediffg": -0.02})
    with pytest.raises(ToolError) as exc:
        tool.execute(node, _ctx(tmp_path))
    assert exc.value.category == "verification"

def test_tool_bands_needs_upstream_chgcar(tmp_path):
    pp = _fake_pp(tmp_path, "Si")
    tool = VASPCalcTool(executor=_OutcarExecutor(), pp_dir=pp)
    node = NodeRun(node_id="bands", params={"calc": "bands", "material": "Si"})
    with pytest.raises(ToolError) as exc:
        tool.execute(node, _ctx(tmp_path))
    assert exc.value.category == "missing_upstream"

def test_tool_magnetic_scf_reports_moment(tmp_path):
    pp = _fake_pp(tmp_path, "Fe")
    mag_outcar = CONVERGED_OUTCAR + " number of electron  8.0 magnetization   2.2\n"
    class _MagExec(FakeExecutor):
        def __init__(self):
            super().__init__(mode="success", stdout=mag_outcar)
        def run_vasp(self, workdir, *, timeout=None):
            (Path(workdir) / "OUTCAR").write_text(mag_outcar)
            return super().run_vasp(workdir, timeout=timeout)
    tool = VASPCalcTool(executor=_MagExec(), pp_dir=pp)
    node = NodeRun(node_id="scf", params={"calc": "scf", "material": "Fe",
                                          "magnetic_order": "FM"})
    out = tool.execute(node, _ctx(tmp_path))
    assert out["spin_polarized"] is True
    assert out["total_magnetization_muB"] == 2.2


# ── Repairer: encut bump ──────────────────────────────────────────────────────

def test_repairer_bumps_encut_on_verification():
    r = VASPParamRepairer()
    node = NodeRun(node_id="scf", params={"calc": "scf", "encut": 520})
    assert r.repair(node, ToolError("x", category="verification", repairable=True)) is True
    assert node.params["encut"] == 560.0

def test_repairer_respects_locked_params():
    r = VASPParamRepairer()
    node = NodeRun(node_id="scf", params={"calc": "scf", "_parameters_locked": True})
    assert r.repair(node, ToolError("x", category="verification")) is False


# ── Graph: new templates run through the runtime ──────────────────────────────

def _bands_engine(tmp_path, extra_files):
    from dft_forge.runtime.engine import GraphEngine

    pp = _fake_pp(tmp_path, "Si")

    class _MultiExec(FakeExecutor):
        def __init__(self):
            super().__init__(mode="success", stdout=CONVERGED_OUTCAR)
        def run_vasp(self, workdir, *, timeout=None):
            wd = Path(workdir)
            (wd / "OUTCAR").write_text(CONVERGED_OUTCAR)
            (wd / "CHGCAR").write_text("fake charge density\n")
            for name, content in extra_files.items():
                (wd / name).write_text(content)
            return super().run_vasp(workdir, timeout=timeout)

    return GraphEngine(
        base_dir=tmp_path / "runs",
        tools={"vasp": VASPCalcTool(executor=_MultiExec(), pp_dir=pp)},
        repairers={"vasp_param_bump": VASPParamRepairer()},
        templates_dir=Path(__file__).resolve().parents[1] / "templates",
    )

def test_bands_template_runs(tmp_path):
    vasprun = (
        "<modeling><calculation>"
        "<energy><i name='e_fr_energy'>-10.84</i></energy>"
        "<dos><i name='efermi'>0.0</i></dos>"
        "<eigenvalues><array><set><set>"
        "<set><r>-2.0 1.0</r><r>1.5 0.0</r></set>"
        "</set></set></array></eigenvalues></calculation></modeling>"
    )
    engine = _bands_engine(tmp_path, {"vasprun.xml": vasprun})
    final = engine.run_template("v2_vasp_bands", inputs={"material": "Si"})
    assert final.state.value == "succeeded"
    assert final.nodes["bands"].outputs["band_gap_ev"] == 3.5

def test_dos_template_runs(tmp_path):
    doscar = "\n".join(["1 1 1 1", "h", "h", "h", "h", "0 0 2 0.0 0",
                        "-1.0 0.5 0.5", "1.0 2.0 1.5",
                        "0 0 2 0.0 0", "-1.0 0.1 0.1", "1.0 0.4 0.5"]) + "\n"
    engine = _bands_engine(tmp_path, {"DOSCAR": doscar})
    final = engine.run_template("v2_vasp_dos", inputs={"material": "Si"})
    assert final.state.value == "succeeded"
    assert final.nodes["dos"].outputs["n_energy_points"] == 2
    assert "Si" in final.nodes["pdos"].outputs["elements"]

def test_phonon_template_runs(tmp_path):
    ph_outcar = CONVERGED_OUTCAR + "   1 f  =  10.0 THz\n   2 f  =  10.0 THz\n"
    # phonon reads OUTCAR frequencies; single-node template, no CHGCAR staging
    class _PhExec(FakeExecutor):
        def __init__(self):
            super().__init__(mode="success", stdout=ph_outcar)
        def run_vasp(self, workdir, *, timeout=None):
            (Path(workdir) / "OUTCAR").write_text(ph_outcar)
            return super().run_vasp(workdir, timeout=timeout)
    pp = _fake_pp(tmp_path, "Si")
    from dft_forge.runtime.engine import GraphEngine
    engine = GraphEngine(
        base_dir=tmp_path / "runs",
        tools={"vasp": VASPCalcTool(executor=_PhExec(), pp_dir=pp)},
        templates_dir=Path(__file__).resolve().parents[1] / "templates",
    )
    final = engine.run_template("v3_vasp_phonon", inputs={"material": "Si"})
    assert final.state.value == "succeeded"
    assert final.nodes["phonon"].outputs["n_modes"] == 2


# ── Phonopy finite-displacement dispersion ────────────────────────────────────

# A displaced-supercell OUTCAR carrying a parseable per-atom TOTAL-FORCE block.
FORCE_OUTCAR = (
    CONVERGED_OUTCAR
    + " POSITION                                       TOTAL-FORCE (eV/Angst)\n"
    + " -----------------------------------------------------------------------\n"
    + "   0.0 0.0 0.0     0.001 0.002 0.003\n"
    + "   1.0 1.0 1.0    -0.001 -0.002 -0.003\n"
    + " -----------------------------------------------------------------------\n"
)


def _node(node_id, params):
    return NodeRun(node_id=node_id, params=params)


def test_phonopy_disp_fails_closed_when_missing(tmp_path, monkeypatch):
    """Without phonopy installed, phonopy_disp must fail with an install hint,
    never fabricate force constants."""
    import sys

    monkeypatch.setitem(sys.modules, "phonopy", None)  # force ImportError
    pp = _fake_pp(tmp_path, "Si")
    tool = VASPCalcTool(executor=_OutcarExecutor(FORCE_OUTCAR), pp_dir=pp)
    ctx = ExecutionContext(run_id="r", base_dir=tmp_path / "runs")
    node = _node("phonopy_disp", {"calc": "phonopy_disp", "material": "Si"})
    with pytest.raises(ToolError) as exc:
        tool.execute(node, ctx)
    assert "phonopy" in str(exc.value).lower()
    assert exc.value.category == "config"


def _install_fake_phonopy(monkeypatch, recorder):
    """Inject a minimal fake 'phonopy' package into sys.modules.

    The fake mimics only the surface the bridge touches: displacement
    generation, force ingestion, band-structure/DOS runs, and save/load.
    """
    import sys
    import types

    class _PhonopyAtoms:
        def __init__(self, symbols=None, cell=None, scaled_positions=None):
            self.symbols = symbols or ["Si", "Si"]
            self.cell = cell if cell is not None else [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            self.scaled_positions = (
                scaled_positions if scaled_positions is not None else [[0, 0, 0], [0.25, 0.25, 0.25]]
            )

    class _Phonopy:
        def __init__(self, atoms, supercell_matrix=None):
            self._sc = [_PhonopyAtoms(), _PhonopyAtoms()]  # two displacements
            self.forces = None
            self.produced = False

        def generate_displacements(self, distance=0.01):
            recorder["distance"] = distance

        @property
        def supercells_with_displacements(self):
            return self._sc

        def produce_force_constants(self):
            self.produced = True

        def run_band_structure(self, qpoints, path_connections=None, labels=None):
            recorder["band_labels"] = labels

        def get_band_structure_dict(self):
            return {
                "distances": [[0.0, 0.5, 1.0]],
                "frequencies": [[[1.0, 2.0, 3.0], [1.1, 2.1, 3.1], [1.2, 2.2, 3.2]]],
            }

        def run_mesh(self, mesh):
            recorder["mesh"] = mesh

        def run_total_dos(self):
            pass

        def get_total_dos_dict(self):
            return {"frequency_points": [0.0, 1.0, 2.0], "total_dos": [0.0, 0.5, 0.2]}

        def save(self, path, settings=None):
            Path(path).write_text("fake phonopy params\n")

    def _band_qpoints(band_path, npoints=51):
        return [[0.0, 0.0, 0.0]], [True]

    def _load(path):
        return _Phonopy(None)

    phonopy_mod = types.ModuleType("phonopy")
    phonopy_mod.Phonopy = _Phonopy
    phonopy_mod.load = _load
    atoms_mod = types.ModuleType("phonopy.structure.atoms")
    atoms_mod.PhonopyAtoms = _PhonopyAtoms
    structure_mod = types.ModuleType("phonopy.structure")
    bs_mod = types.ModuleType("phonopy.phonon.band_structure")
    bs_mod.get_band_qpoints_and_path_connections = _band_qpoints
    phonon_mod = types.ModuleType("phonopy.phonon")

    for name, mod in {
        "phonopy": phonopy_mod,
        "phonopy.structure": structure_mod,
        "phonopy.structure.atoms": atoms_mod,
        "phonopy.phonon": phonon_mod,
        "phonopy.phonon.band_structure": bs_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)


def test_phonopy_dispersion_template_runs(tmp_path, monkeypatch):
    """End-to-end: relax -> phonopy_disp -> phonopy_bands with a fake phonopy.

    Verifies displacement fan-out, force collection, and that dispersion,
    DOS, and stability propagate through the graph outputs."""
    recorder: dict = {}
    _install_fake_phonopy(monkeypatch, recorder)
    pp = _fake_pp(tmp_path, "Si")

    from dft_forge.runtime.engine import GraphEngine

    class _PhonExec(FakeExecutor):
        def __init__(self):
            super().__init__(mode="success", stdout=FORCE_OUTCAR)
        def run_vasp(self, workdir, *, timeout=None):
            (Path(workdir) / "OUTCAR").write_text(FORCE_OUTCAR)
            return super().run_vasp(workdir, timeout=timeout)

    engine = GraphEngine(
        base_dir=tmp_path / "runs",
        tools={"vasp": VASPCalcTool(executor=_PhonExec(), pp_dir=pp)},
        repairers={"vasp_param_bump": VASPParamRepairer()},
        templates_dir=Path(__file__).resolve().parents[1] / "templates",
    )
    final = engine.run_template(
        "v4_vasp_phonon_dispersion",
        inputs={"material": "Si", "phonon_supercell": [2, 2, 2]},
    )
    assert final.state.value == "succeeded"
    disp = final.nodes["phonopy_disp"].outputs
    assert disp["n_displacements"] == 2
    assert Path(disp["phonopy_yaml"]).is_file()
    bands = final.nodes["phonopy_bands"].outputs
    assert bands["max_frequency_thz"] == 3.2
    assert bands["min_frequency_thz"] == 1.0
    assert bands["stability"]["dynamically_stable_at_threshold"] is True
    assert bands["phonon_dos"]["total_dos"] == [0.0, 0.5, 0.2]
    assert recorder["distance"] == 0.01  # displacement distance flowed through
