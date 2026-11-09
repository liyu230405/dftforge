"""Tests for the ABINIT engine: writer, parser, tool, and graph integration.

ABINIT is free/open source, so unlike the VASP tests this path is one a user
can reproduce end-to-end (``conda install -c conda-forge abinit``). Here no real
binary runs: a FakeExecutor injects real-format ``.abo`` text and the DDB/DEN
artifacts, exactly as the QE/VASP engine tests inject their tool output.
Pseudopotentials are temporary fakes; nothing is bundled or fabricated.
"""

from pathlib import Path
import re

import pytest

from dft_forge.engines.abinit import parser as ap
from dft_forge.engines.abinit import writer as aw
from dft_forge.engines.abinit.tool import ABINITCalcTool, ABINITParamRepairer
from dft_forge.engines.abinit.writer import ABINITInputError
from dft_forge.executor import FakeExecutor
from dft_forge.runtime.run import ExecutionContext, NodeRun, ToolError

# ── Real-format .abo samples (public, stable ABINIT 9.x text output) ──────────

SCF_ABO = """
 ABINIT 9.10.3

 Cartesian forces (hartree/bohr) at end:
    1     -0.00100000000000    0.00000000000000    0.00000000000000
    2      0.00100000000000    0.00000000000000    0.00000000000000

     iter   Etot(hartree)      deltaE(h)  residm    vres2
 ETOT  1  -7.8500000000000    -7.850E+00 1.2E-02 3.4E-01
 ETOT 12  -7.9012345678000    -1.234E-11 5.6E-13 1.0E-11

 At SCF step   12, etot is converged :
  for the second time, diff in etot=  1.234E-11 < toldfe=  1.000E-10

 Fermi (or HOMO) energy (hartree) =   0.20100   Average Vxc (hartree)

 Unit cell volume ucvol=  2.7010000E+02 bohr^3

    etotal   -7.9012345678E+00
"""

UNCONVERGED_ABO = """
 ABINIT 9.10.3
     iter   Etot(hartree)      deltaE(h)  residm    vres2
 ETOT  1  -7.0000000000000    -7.000E+00 1.2E-02 3.4E-01
    etotal   -7.0000000000000E+00
"""

BANDS_ABO = """
 Eigenvalues (hartree) for nkpt=   2  k points:
 kpt#   1, nband=  4, wtk=  0.5, kpt=  0.0 0.0 0.0
  -0.20  -0.05   0.10   0.30
 kpt#   2, nband=  4, wtk=  0.5, kpt=  0.5 0.0 0.0
  -0.21  -0.04   0.11   0.31

 Fermi (or HOMO) energy (hartree) =   0.00000
"""

METAL_BANDS_ABO = """
 Eigenvalues (hartree) for nkpt=   2  k points:
 kpt#   1, nband=  2, wtk=  0.5, kpt=  0.0 0.0 0.0
  -0.0005   0.0006
 kpt#   2, nband=  2, wtk=  0.5, kpt=  0.5 0.0 0.0
  -0.0004   0.0005

 Fermi (or HOMO) energy (hartree) =   0.00000
"""

ANADDB_OUT = """
 anaddb : phonon calculation

 Phonon frequencies in Hartree :
   1.0E-05  1.1E-05  1.2E-05  2.5E-04  2.5E-04  2.6E-04
"""

ANADDB_IMAG_OUT = """
 Phonon frequencies in Hartree :
  -5.0E-04  1.1E-05  1.2E-05  2.5E-04  2.5E-04  2.6E-04
"""


def _fake_pp(tmp_path: Path, *elements: str) -> Path:
    pp = tmp_path / "pp"
    pp.mkdir(parents=True, exist_ok=True)
    for el in elements:
        (pp / f"{el}.psp8").write_text(f"fake {el} psp8\n")
    return pp


def _ctx(tmp_path):
    return ExecutionContext(base_dir=tmp_path, run_id="r1")


# ── Writer ────────────────────────────────────────────────────────────────────

def test_build_abi_scf_insulator_fixed_occupation():
    atoms, _ = aw.resolve_atoms("Si", None)
    abi = aw.build_abi("scf", {}, atoms, is_metal=False, species=["Si"])
    assert "occopt 1" in abi and "toldfe" in abi
    assert "ionmov" not in abi  # scf does not move ions

def test_build_abi_metal_uses_smearing():
    atoms, _ = aw.resolve_atoms("Al", None)
    abi = aw.build_abi("scf", {}, atoms, is_metal=True, species=["Al"])
    assert "occopt 7" in abi and "tsmear" in abi

def test_build_abi_relax_fixes_cell():
    atoms, _ = aw.resolve_atoms("Si", None)
    abi = aw.build_abi("relax", {}, atoms, is_metal=False, species=["Si"])
    assert "ionmov 2" in abi and "optcell 0" in abi and "tolmxf" in abi

def test_build_abi_vc_relax_relaxes_cell():
    atoms, _ = aw.resolve_atoms("Si", None)
    abi = aw.build_abi("vc-relax", {}, atoms, is_metal=False, species=["Si"])
    assert "optcell 2" in abi and "ecutsm" in abi

def test_build_abi_nscf_reads_density():
    atoms, _ = aw.resolve_atoms("Si", None)
    abi = aw.build_abi("nscf", {}, atoms, is_metal=False, species=["Si"])
    # Each calc is its own abinit run, so the density comes from the upstream
    # node's symlinked file, not from a dataset of this run (`getden 1`).
    assert "iscf -2" in abi
    assert 'getden_filepath "upstream_DEN"' in abi
    assert "getden 1" not in abi

def test_build_abi_dos_prtdos():
    atoms, _ = aw.resolve_atoms("Si", None)
    abi = aw.build_abi("dos", {}, atoms, is_metal=False, species=["Si"])
    assert "prtdos 2" in abi

def test_build_abi_pdos_projected():
    atoms, _ = aw.resolve_atoms("Si", None)
    abi = aw.build_abi("pdos", {}, atoms, is_metal=False, species=["Si"])
    assert "prtdos 3" in abi and "natsph" in abi and "ratsph" in abi

def test_build_abi_dfpt_multidataset():
    atoms, _ = aw.resolve_atoms("Si", None)
    abi = aw.build_abi("dfpt", {"ngqpt": [2, 2, 2]}, atoms, is_metal=False, species=["Si"])
    # One ground-state dataset plus one perturbation dataset per q point of the
    # complete 2x2x2 mesh: 1 + 8. Truncating the mesh would silently drop modes.
    assert "ndtset 9" in abi
    assert "iscf1 7" in abi and "rfphon2 1" in abi
    assert len([l for l in abi.splitlines() if l.startswith("qpt")]) == 8
    # The DDB q-mesh is declared to anaddb as `ngqpt`; `ddb_ngqpt` is not an
    # phonon mesh generator (it is EPH metadata). See the anaddb writer test.
    assert "ddb_ngqpt" not in abi

def test_build_abi_rejects_unknown_calc():
    atoms, _ = aw.resolve_atoms("Si", None)
    with pytest.raises(ABINITInputError):
        aw.build_abi("gw", {}, atoms, is_metal=False, species=["Si"])

def test_geometry_block_is_bohr_and_typat():
    atoms, _ = aw.resolve_atoms("Si", None)
    abi = aw.build_abi("scf", {}, atoms, is_metal=False, species=["Si"])
    assert "acell 1.0 1.0 1.0" in abi and "znucl 14" in abi and "rprim" in abi
    assert "typat 1 1" in abi  # two Si atoms, both type 1

def test_kpoint_block_explicit_grid():
    atoms, _ = aw.resolve_atoms("Si", None)
    abi = aw.build_abi("scf", {"ngkpt": [8, 8, 8]}, atoms, is_metal=False, species=["Si"])
    assert "ngkpt 8 8 8" in abi

def test_kpoint_block_rejects_bad_grid():
    atoms, _ = aw.resolve_atoms("Si", None)
    with pytest.raises(ABINITInputError):
        aw.build_abi("scf", {"ngkpt": [8, 0, 8]}, atoms, is_metal=False, species=["Si"])

def test_band_kpoints_line_mode():
    atoms, _ = aw.resolve_atoms("Si", None)
    abi = aw.build_abi("bands", {"ndivsm": 12}, atoms, is_metal=False, species=["Si"])
    assert "kptbounds" in abi and "ndivsm 12" in abi and "iscf -2" in abi

def test_magnetism_fm_block():
    atoms, _ = aw.resolve_atoms("Fe", None)
    species = aw.species_order(atoms)
    abi = aw.build_abi("scf", {"magnetic_order": "FM"}, atoms, is_metal=True, species=species)
    assert "nsppol 2" in abi and "spinat" in abi

def test_magnetism_afm_alternates_sign():
    from ase import Atoms as _A
    atoms = _A("Fe2", positions=[[0, 0, 0], [1.4, 1.4, 1.4]],
               cell=[2.8, 2.8, 2.8], pbc=True)
    abi = aw.build_abi("scf", {"magnetic_order": "AFM", "initial_moment_muB": 3.0},
                       atoms, is_metal=True, species=["Fe"])
    assert "3.000" in abi and "-3.000" in abi

def test_species_order_first_appearance():
    from ase import Atoms as _A
    atoms = _A("OSiO", positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0]],
               cell=[5, 5, 5], pbc=True)
    assert aw.species_order(atoms) == ["O", "Si"]

def test_pseudo_candidates_common_first():
    cands = aw.pseudo_candidates("Si")
    assert cands[0] == "Si.psp8"

def test_resolve_pseudos_fail_closed_without_dir():
    with pytest.raises(ABINITInputError, match="not bundled"):
        aw.resolve_pseudos(["Si"], None)

def test_resolve_pseudos_fail_closed_missing_element(tmp_path):
    pp = _fake_pp(tmp_path, "Si")
    with pytest.raises(ABINITInputError, match="missing pseudopotential for: O"):
        aw.resolve_pseudos(["Si", "O"], pp)

def test_resolve_pseudos_order_matches_species(tmp_path):
    pp = _fake_pp(tmp_path, "Ga", "As")
    resolved = aw.resolve_pseudos(["Ga", "As"], pp)
    assert [p.name for p in resolved] == ["Ga.psp8", "As.psp8"]

def test_write_inputs_produces_abi(tmp_path):
    pp = _fake_pp(tmp_path, "Si")
    meta = aw.write_inputs("Si", None, "scf", {}, tmp_path / "run", pp_dir=pp)
    assert Path(meta["abi_file"]).is_file()
    assert meta["species"] == ["Si"] and meta["natoms"] == 2
    body = Path(meta["abi_file"]).read_text()
    assert "pseudos" in body and "Si.psp8" in body

def test_write_inputs_supercell_multiplies_atoms(tmp_path):
    pp = _fake_pp(tmp_path, "Si")
    meta = aw.write_inputs("Si", None, "scf", {"supercell": "2x1x1"},
                           tmp_path / "run", pp_dir=pp)
    assert meta["natoms"] == 4 and meta["supercell"] == "2x1x1"

def test_write_anaddb_input_has_ifc_and_dos(tmp_path):
    atoms, _ = aw.resolve_atoms("Si", None)
    path = aw.write_anaddb_input({"ngqpt": [2, 2, 2]}, tmp_path, "Si", atoms)
    text = path.read_text()
    assert "ifcflag 1" in text and "ngqpt 2 2 2" in text and "prtdos 1" in text


# ── Parser ────────────────────────────────────────────────────────────────────

def test_parse_scf_energy_hartree_to_ev():
    p = ap.parse_scf(SCF_ABO)
    assert abs(p.total_energy_ev - (-7.9012345678 * ap.HA_TO_EV)) < 1e-6
    assert p.converged is True
    assert p.n_scf_steps == 12

def test_parse_scf_fermi_and_volume():
    p = ap.parse_scf(SCF_ABO)
    assert abs(p.fermi_energy_ev - 0.201 * ap.HA_TO_EV) < 1e-6
    # ucvol 270.1 bohr^3 -> Å^3
    assert abs(p.final_volume_ang3 - 270.1 / (1.8897261246257702 ** 3)) < 1e-3

def test_parse_scf_forces_hartree_bohr_to_ev_ang():
    p = ap.parse_scf(SCF_ABO)
    # |F| = 0.001 Ha/Bohr -> eV/Å
    assert abs(p.max_force_ev_ang - 0.001 * ap.HA_BOHR_TO_EV_ANG) < 1e-6

def test_parse_scf_unconverged():
    p = ap.parse_scf(UNCONVERGED_ABO)
    assert p.converged is False

def test_parse_result_reads_abo_file(tmp_path):
    (tmp_path / "Si.abo").write_text(SCF_ABO)
    p = ap.parse_result(tmp_path)
    assert abs(p.total_energy_ev - (-7.9012345678 * ap.HA_TO_EV)) < 1e-6

def test_parse_bands_insulator_gap():
    b = ap.parse_bands(".", BANDS_ABO)
    assert b.n_kpoints == 2 and b.n_bands == 4
    assert b.is_metal is False
    # vbm = max occupied (-0.04 Ha), cbm = min empty (0.10 Ha)
    assert abs(b.band_gap_ev - (0.10 - (-0.04)) * ap.HA_TO_EV) < 1e-4

def test_parse_bands_metal():
    b = ap.parse_bands(".", METAL_BANDS_ABO)
    assert b.is_metal is True and b.band_gap_ev == 0.0

def test_parse_dos_total(tmp_path):
    dos_file = tmp_path / "out_DOS"
    dos_file.write_text(
        "# Fermi energy :   0.00000\n"
        "# energy(Ha)  DOS  IDOS\n"
        "  -0.10   0.50   0.10\n"
        "   0.00   2.00   1.00\n"
        "   0.10   0.30   3.00\n"
    )
    dos = ap.parse_dos(tmp_path)
    assert dos.n_energy_points == 3
    assert dos.fermi_energy_ev == 0.0
    assert abs(dos.energies_ev[1] - 0.0) < 1e-9

def test_parse_dos_projected(tmp_path):
    (tmp_path / "out_DOS").write_text(
        "# total\n  -0.10   0.50   0.10\n   0.10   0.30   3.00\n")
    (tmp_path / "Si_DOS_AT0001").write_text(
        "# atom 1\n  -0.10   0.25\n   0.10   0.15\n")
    dos = ap.parse_dos(tmp_path, projected=True)
    assert dos.element_dos is not None and "Si" in dos.element_dos

def test_parse_phonon_stable():
    ph = ap.parse_phonon(ANADDB_OUT)
    assert ph.n_modes == 6 and ph.has_imaginary is False
    assert ph.frequencies_thz[0] > 0

def test_parse_phonon_imaginary():
    ph = ap.parse_phonon(ANADDB_IMAG_OUT)
    assert ph.has_imaginary is True
    assert min(ph.frequencies_thz) < 0

@pytest.mark.parametrize("token,category", [
    ("Dilatmx has been exceeded", "config"),
    ("SCF cycle did not converge", "verification"),
    ("Out of memory", "resource"),
    ("ERROR", "execution"),
])
def test_classify_error(token, category):
    assert ap.classify_error(token).category == category

def test_classify_error_clean():
    assert ap.classify_error("calculation completed") is None


# ── Tool ──────────────────────────────────────────────────────────────────────

class _AboExecutor(FakeExecutor):
    """FakeExecutor that drops a real-format .abo into the workdir on run."""

    def __init__(self, abo: str = SCF_ABO, mode: str = "success", label: str = "Si"):
        super().__init__(mode=mode, stdout=abo)
        self._abo = abo
        self._label = label

    def run_abinit(self, input_file, workdir, *, timeout=None, binary="abinit"):
        if self.mode != "failure":
            (Path(workdir) / f"{self._label}.abo").write_text(self._abo)
            if "gradients are converged" in self._abo:
                _write_fake_gsr(workdir)
        return super().run_abinit(input_file, workdir, timeout=timeout, binary=binary)


def test_tool_scf_success(tmp_path):
    pp = _fake_pp(tmp_path, "Si")
    tool = ABINITCalcTool(executor=_AboExecutor(), pp_dir=pp)
    node = NodeRun(node_id="scf", params={"calc": "scf", "material": "Si"})
    out = tool.execute(node, _ctx(tmp_path))
    assert out["engine"] == "abinit"
    assert out["converged"] is True
    assert abs(out["energy_ev"] - (-7.9012345678 * ap.HA_TO_EV)) < 1e-6
    assert out["method_profile"]["engine"] == "abinit"

def test_tool_missing_pseudo_is_config_error(tmp_path):
    tool = ABINITCalcTool(executor=_AboExecutor(), pp_dir=tmp_path / "empty")
    node = NodeRun(node_id="scf", params={"calc": "scf", "material": "Si"})
    with pytest.raises(ToolError) as exc:
        tool.execute(node, _ctx(tmp_path))
    assert exc.value.category == "config"

def test_tool_execution_failure_raises(tmp_path):
    pp = _fake_pp(tmp_path, "Si")
    tool = ABINITCalcTool(executor=_AboExecutor(mode="failure"), pp_dir=pp)
    node = NodeRun(node_id="scf", params={"calc": "scf", "material": "Si"})
    with pytest.raises(ToolError):
        tool.execute(node, _ctx(tmp_path))

def test_tool_unconverged_raises_verification(tmp_path):
    pp = _fake_pp(tmp_path, "Si")
    tool = ABINITCalcTool(executor=_AboExecutor(abo=UNCONVERGED_ABO), pp_dir=pp)
    node = NodeRun(node_id="scf", params={"calc": "scf", "material": "Si"})
    with pytest.raises(ToolError) as exc:
        tool.execute(node, _ctx(tmp_path))
    assert exc.value.category == "verification"

def test_tool_requires_run_abinit_capable_executor(tmp_path):
    pp = _fake_pp(tmp_path, "Si")
    class NoAbinit(FakeExecutor):
        run_abinit = None
    tool = ABINITCalcTool(executor=NoAbinit(), pp_dir=pp)
    node = NodeRun(node_id="scf", params={"calc": "scf", "material": "Si"})
    with pytest.raises(ToolError, match="run_abinit"):
        tool.execute(node, _ctx(tmp_path))

def test_tool_bands_needs_upstream_density(tmp_path):
    pp = _fake_pp(tmp_path, "Si")
    tool = ABINITCalcTool(executor=_AboExecutor(abo=BANDS_ABO), pp_dir=pp)
    node = NodeRun(node_id="bands", params={"calc": "bands", "material": "Si"})
    with pytest.raises(ToolError) as exc:
        tool.execute(node, _ctx(tmp_path))
    assert exc.value.category == "missing_upstream"

def test_tool_anaddb_needs_upstream_ddb(tmp_path):
    pp = _fake_pp(tmp_path, "Si")
    tool = ABINITCalcTool(executor=_AboExecutor(abo=ANADDB_OUT), pp_dir=pp)
    node = NodeRun(node_id="anaddb", params={"calc": "anaddb", "material": "Si"})
    with pytest.raises(ToolError) as exc:
        tool.execute(node, _ctx(tmp_path))
    assert exc.value.category == "missing_upstream"


# ── Repairer ────────────────────────────────────────────────────────────────

def test_repairer_bumps_ecut_on_verification():
    r = ABINITParamRepairer()
    node = NodeRun(node_id="scf", params={"calc": "scf", "ecut": 15.0})
    assert r.repair(node, ToolError("x", category="verification", repairable=True)) is True
    assert node.params["ecut"] == 20.0

def test_repairer_respects_locked_params():
    r = ABINITParamRepairer()
    node = NodeRun(node_id="scf", params={"calc": "scf", "_parameters_locked": True})
    assert r.repair(node, ToolError("x", category="verification")) is False

def test_repairer_stops_at_max_ecut():
    r = ABINITParamRepairer()
    node = NodeRun(node_id="scf", params={"calc": "scf", "ecut": 50.0})
    assert r.repair(node, ToolError("x", category="verification")) is False

def test_repairer_ignores_unknown_category():
    r = ABINITParamRepairer()
    node = NodeRun(node_id="scf", params={"ecut": 15.0})
    assert r.repair(node, ToolError("x", category="config")) is False


# ── Graph integration ─────────────────────────────────────────────────────────

# A relaxed .abo with forces small enough to pass the relax convergence check
# (|F| = 1e-5 Ha/Bohr well under 3× the default tolmxf threshold).
RELAX_ABO = """
 ABINIT 9.10.3

 Cartesian forces (hartree/bohr) at end:
    1     -0.00001000000000    0.00000000000000    0.00000000000000
    2      0.00001000000000    0.00000000000000    0.00000000000000

     iter   Etot(hartree)      deltaE(h)  residm    vres2
 ETOT 12  -7.9012345678000    -1.234E-11 5.6E-13 1.0E-11

 At SCF step   12, etot is converged :
  for the second time, diff in etot=  1.234E-11 < toldfe=  1.000E-10

 Fermi (or HOMO) energy (hartree) =   0.20100   Average Vxc (hartree)

 Unit cell volume ucvol=  2.7010000E+02 bohr^3

    etotal   -7.9012345678E+00
"""


RELAX_ABO += "\n At Broyd/MD step 2, gradients are converged :\n"


def _write_fake_gsr(workdir):
    import h5py
    import numpy as np
    atoms, _ = aw.resolve_atoms("Si", None)
    with h5py.File(Path(workdir) / "Sio_GSR.nc", "w") as f:
        f["primitive_vectors"] = atoms.cell.array / ap.BOHR_TO_ANG
        f["reduced_atom_positions"] = atoms.get_scaled_positions()
        f["atomic_numbers"] = [14.0]
        f["atom_species"] = [1, 1]
        f["cartesian_forces"] = np.zeros((2, 3))
        f["etotal"] = -7.9012345678


class _MultiBinaryExec(FakeExecutor):
    """Injects the right ABINIT artifacts per binary/calc so a full graph runs.

    - abinit runs: write ``Si.abo`` + a density (_DEN) so downstream nscf/bands
      stage it, and a DDB whenever the input requests DFPT (rfphon2).
    - anaddb runs: return phonon frequencies on stdout for parse_phonon.
    """

    def __init__(self, scf_abo=SCF_ABO, bands_abo=None, dos_file=None):
        super().__init__(mode="success", stdout="")
        self._scf_abo = scf_abo
        self._bands_abo = bands_abo
        self._dos_file = dos_file

    def run_abinit(self, input_file, workdir, *, timeout=None, binary="abinit"):
        wd = Path(workdir)
        if binary == "anaddb":
            self.stdout = ANADDB_OUT
            return super().run_abinit(input_file, workdir, timeout=timeout, binary=binary)
        self.stdout = ""
        abi_text = Path(input_file).read_text() if Path(input_file).is_file() else ""
        # Choose the output the node's finisher will parse.
        if "kptbounds" in abi_text and self._bands_abo:
            (wd / "Si.abo").write_text(self._bands_abo)
        elif "rfphon2" in abi_text:
            (wd / "Si.abo").write_text(RELAX_ABO)
            for index in re.findall(r"^rfphon(\d+) 1$", abi_text, re.M):
                (wd / f"Sio_DS{index}_DDB").write_text("fake DDB\n")
        else:
            (wd / "Si.abo").write_text(self._scf_abo)
            if "ionmov" in abi_text and "gradients are converged" in self._scf_abo:
                _write_fake_gsr(wd)
        # Density for downstream staging.
        (wd / "Si_DEN").write_text("fake density\n")
        if self._dos_file is not None:
            (wd / "out_DOS").write_text(self._dos_file)
            (wd / "Si_DOS_AT0001").write_text(
                "# atom 1\n  -0.10   0.25\n   0.00   0.50\n   0.10   0.15\n")
        return super().run_abinit(input_file, workdir, timeout=timeout, binary=binary)

    def merge_abinit_ddb(self, ddb_files, output_file, workdir, *, timeout=None):
        assert len(ddb_files) == 8
        Path(output_file).write_text("fake merged DDB\n")
        return super().run_abinit(ddb_files[0], workdir, timeout=timeout)


def _engine(tmp_path, executor):
    from dft_forge.runtime.engine import GraphEngine

    pp = _fake_pp(tmp_path, "Si")
    return GraphEngine(
        base_dir=tmp_path / "runs",
        tools={"abinit": ABINITCalcTool(executor=executor, pp_dir=pp)},
        repairers={"abinit_param_bump": ABINITParamRepairer()},
        templates_dir=Path(__file__).resolve().parents[1] / "templates",
    )


def test_scf_template_runs(tmp_path):
    engine = _engine(tmp_path, _AboExecutor())
    final = engine.run_template("a0_abinit_scf", inputs={"material": "Si"})
    assert final.state.value == "succeeded"
    assert abs(final.nodes["scf"].outputs["energy_ev"] - (-7.9012345678 * ap.HA_TO_EV)) < 1e-6

def test_relax_template_runs(tmp_path):
    engine = _engine(tmp_path, _AboExecutor(abo=RELAX_ABO))
    final = engine.run_template("a1_abinit_relax", inputs={"material": "Si"})
    assert final.state.value == "succeeded"
    assert final.nodes["relax"].outputs["converged"] is True

def test_bands_template_runs(tmp_path):
    engine = _engine(tmp_path, _MultiBinaryExec(bands_abo=BANDS_ABO))
    final = engine.run_template("a2_abinit_bands", inputs={"material": "Si"})
    assert final.state.value == "succeeded"
    b = final.nodes["bands"].outputs
    assert b["n_kpoints"] == 2 and b["is_metal"] is False

def test_dos_template_runs(tmp_path):
    dos_file = (
        "# Fermi energy :   0.00000\n"
        "  -0.10   0.50   0.10\n   0.00   2.00   1.00\n   0.10   0.30   3.00\n"
    )
    engine = _engine(tmp_path, _MultiBinaryExec(dos_file=dos_file))
    final = engine.run_template("a2_abinit_dos", inputs={"material": "Si"})
    assert final.state.value == "succeeded"
    assert final.nodes["dos"].outputs["n_energy_points"] == 3
    assert "Si" in final.nodes["pdos"].outputs["elements"]

def test_phonon_dfpt_chain_runs(tmp_path):
    """relax -> dfpt (produces DDB) -> anaddb (dispersion + stability)."""
    engine = _engine(tmp_path, _MultiBinaryExec(scf_abo=RELAX_ABO))
    final = engine.run_template("a3_abinit_phonon", inputs={"material": "Si"})
    assert final.state.value == "succeeded"
    assert final.nodes["dfpt"].outputs["ddb_file"].endswith("_DDB")
    an = final.nodes["anaddb"].outputs
    assert an["n_modes"] == 6
    assert an["has_imaginary_modes"] is False
    assert an["stability"]["dynamically_stable_at_threshold"] is True
