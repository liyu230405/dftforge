"""Regressions discovered with real ABINIT 10.0.3 Si calculations."""
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest

from dft_forge.engines.abinit import parser as ap, writer as aw
from dft_forge.engines.abinit.tool import ABINITCalcTool
from dft_forge.executor import LocalExecutor
from dft_forge.runtime.run import ExecutionContext, NodeRun, ToolError
from dft_forge.tools import graph_tools

FIXTURES = Path(__file__).parent / "fixtures" / "abinit"


def test_real_scf_forces_do_not_include_output_variable_arrays():
    result = ap.parse_scf((FIXTURES / "si_scf_10.0.3.abo").read_text())
    assert result.converged
    assert result.max_force_ev_ang == pytest.approx(0, abs=1e-10)
    assert result.total_energy_ev == pytest.approx(-238.80306473, abs=1e-6)


def test_real_relax_has_positive_ionic_convergence():
    text = (FIXTURES / "si_relax_10.0.3.abo").read_text()
    assert ap.parse_scf(text).ionic_converged
    assert not ap.parse_scf(text.replace("gradients are converged", "gradients have not converged yet")).ionic_converged


def test_real_phonons_stop_before_cm_inverse_and_timing():
    result = ap.parse_phonon((FIXTURES / "si_anaddb_10.0.3.out").read_text())
    assert result.n_modes == 12 * 6
    assert max(result.frequencies_thz) == pytest.approx(17.4823715095)
    assert min(result.frequencies_thz) == pytest.approx(-6.15231107894)


def test_real_phonon_netcdf_is_preferred_when_available(tmp_path):
    with h5py.File(tmp_path / "Si_PHBST.nc", "w") as f:
        f["phfreqs"] = np.array([[0.0, 0.001, -0.002]])
    result = ap.parse_phonon("timing 999999", workdir=tmp_path)
    assert result.frequencies_thz == pytest.approx([0.0, 0.241798924, -0.483597848])


def test_real_phonon_netcdf_and_text_agree():
    text = ap.parse_phonon((FIXTURES / "si_anaddb_10.0.3.out").read_text())
    binary = ap.parse_phonon(workdir=FIXTURES)
    assert binary.frequencies_thz == pytest.approx(text.frequencies_thz, abs=5e-6)


def test_missing_energy_is_unknown_not_zero(tmp_path):
    tool = ABINITCalcTool()
    with pytest.raises(ToolError, match="SCF|scf"):
        tool._finish_scf(tmp_path, "SCF cycle converged", {}, {})


def test_electronic_convergence_alone_cannot_pass_relax(tmp_path):
    text = (FIXTURES / "si_scf_10.0.3.abo").read_text()
    with pytest.raises(ToolError, match="force/stress"):
        ABINITCalcTool()._finish_relax(tmp_path, text, {}, {})


def test_later_scf_failure_overrides_earlier_success():
    text = "At SCF step 12, etot is converged\nnot enough scf cycles to converge"
    assert not ap.parse_scf(text).converged


@pytest.mark.parametrize("classic", [False, True])
def test_final_geometry_converts_bohr_and_uses_final_netcdf(tmp_path, classic):
    arrays = {"primitive_vectors": np.eye(3) * 10,
              "reduced_atom_positions": np.array([[0, 0, 0], [0.25, 0.25, 0.25]]),
              "atomic_numbers": np.array([14.0]), "atom_species": np.array([1, 1])}
    path = tmp_path / "Sio_GSR.nc"
    if classic:
        from scipy.io import netcdf_file
        with netcdf_file(path, "w") as f:
            for key, arr in arrays.items():
                dims = tuple(f"{key}_{i}" for i in range(arr.ndim))
                for dim, n in zip(dims, arr.shape):
                    f.createDimension(dim, n)
                f.createVariable(key, "f8", dims)[:] = arr
    else:
        with h5py.File(path, "w") as f:
            for key, arr in arrays.items():
                f[key] = arr
    atoms = ap.read_final_atoms(tmp_path)
    assert atoms.cell.lengths() == pytest.approx([5.2917721067] * 3)
    assert atoms.get_chemical_symbols() == ["Si", "Si"]


def test_band_gap_uses_occupations_instead_of_zero_energy_reference(tmp_path):
    with h5py.File(tmp_path / "Sio_GSR.nc", "w") as f:
        f["eigenvalues"] = [[[0.2, 0.3, 0.7], [0.1, 0.4, 0.6]]]
        f["occupations"] = [[[2, 2, 0], [2, 2, 0]]]
        f["fermi_energy"] = 0.4
    result = ap.parse_bands(tmp_path)
    assert result.band_gap_ev == pytest.approx(0.2 * ap.HA_TO_EV)
    assert (result.n_kpoints, result.n_bands) == (2, 3)


def test_pdos_sums_angular_channels_and_groups_real_atom_species(tmp_path):
    (tmp_path / "out_DOS_TOTAL").write_text("0.0 10.0 0.0\n0.1 20.0 2.0\n")
    for index in [1, 2, 3]:
        # Five differential channels then five integrated channels.
        (tmp_path / f"out_DOS_AT{index:04}").write_text(
            "0.0 1.0 2.0 3.0 4.0 5.0 100.0 100.0 100.0 100.0 100.0\n"
            "0.1 2.0 2.0 2.0 2.0 2.0 100.0 100.0 100.0 100.0 100.0\n")
    result = ap.parse_dos(tmp_path, projected=True, atom_symbols=["O", "Si", "O"])
    assert result.dos == pytest.approx(np.array([10, 20]) / ap.HA_TO_EV)
    assert result.element_dos["O"] == pytest.approx(np.array([30, 20]) / ap.HA_TO_EV)
    assert result.element_dos["Si"] == pytest.approx(np.array([15, 10]) / ap.HA_TO_EV)


def test_pdos_mismatched_grid_is_rejected(tmp_path):
    (tmp_path / "out_DOS_TOTAL").write_text("0.0 1.0 0.0\n")
    (tmp_path / "out_DOS_AT0001").write_text("0.1 2.0\n")
    with pytest.raises(ValueError, match="grid"):
        ap.parse_dos(tmp_path, projected=True, atom_symbols=["Si"])


@pytest.mark.parametrize("params", [{"ecut": float("nan")}, {"ecut": -1},
    {"ngkpt": [2, 2]}, {"ngkpt": [2.5, 2, 2]}, {"ngkpt": [0, 2, 2]}, {"nstep": "abc"}])
def test_bad_parameters_are_not_silently_replaced(params):
    atoms, _ = aw.resolve_atoms("Si", None)
    with pytest.raises(aw.ABINITInputError):
        aw.build_abi("scf", params, atoms, False, ["Si"])


def test_dfpt_requires_commensurate_grid():
    atoms, _ = aw.resolve_atoms("Si", None)
    with pytest.raises(aw.ABINITInputError, match="divisible"):
        aw.build_abi("dfpt", {"ngkpt": [3, 3, 3], "ngqpt": [2, 2, 2]}, atoms, False, ["Si"])


def test_dfpt_cannot_use_ground_state_ddb_as_complete_mesh(tmp_path):
    (tmp_path / "Sio_DS1_DDB").write_text("ground state")
    with pytest.raises(ToolError, match="missing DFPT"):
        ABINITCalcTool()._merge_ddb(tmp_path, {"ngqpt": [1, 1, 1]}, None)


def test_wfk_does_not_substitute_for_density(tmp_path):
    (tmp_path / "scf").mkdir()
    (tmp_path / "scf" / "Sio_WFK").write_text("wavefunction")
    ctx = ExecutionContext(base_dir=tmp_path, run_id="r", upstream={"scf": {}})
    with pytest.raises(ToolError, match="density"):
        ABINITCalcTool()._stage_density(ctx, tmp_path / "bands")


def test_local_abinit_does_not_require_qe_and_tracks_binary_changes(tmp_path, monkeypatch):
    binary = tmp_path / "abinit"
    binary.write_text("#!/bin/sh\nexit 0\n"); binary.chmod(0o755)
    monkeypatch.setenv("DFT_FORGE_EXECUTOR", "local")
    monkeypatch.setenv("DFT_FORGE_ABINIT_BIN", str(binary))
    monkeypatch.setenv("DFT_FORGE_QE_BIN", str(tmp_path / "no-qe"))
    first, profile, identity = graph_tools._build_execution()
    assert first.qe_bin_dir is None
    assert profile["abinit_executables_sha256"]["abinit"]
    binary.write_text("#!/bin/sh\nexit 1\n")
    assert graph_tools._build_execution()[2] != identity


def test_timeout_preserves_abinit_logs(tmp_path):
    path = tmp_path / "input.py"
    path.write_text("import time\nprint('before timeout', flush=True)\ntime.sleep(5)\n")
    executor = LocalExecutor(abinit_bin=sys.executable)
    result = executor.run_abinit(path, tmp_path, timeout=0.15)
    assert not result.success and "timed out" in result.error_message
    assert "before timeout" in (tmp_path / "input.log").read_text()


def test_mrgddb_uses_stdin_control_file(tmp_path):
    abinit = tmp_path / "abinit"
    abinit.write_text("#!/bin/sh\nexit 0\n"); abinit.chmod(0o755)
    merger = tmp_path / "mrgddb"
    merger.write_text(f"#!{sys.executable}\nimport sys\nfrom pathlib import Path\n"
                     "assert len(sys.argv)==1\nrows=sys.stdin.read().splitlines()\n"
                     "assert int(rows[2])==len(rows[3:])\nPath(rows[0]).write_text('merged')\n")
    merger.chmod(0o755)
    inputs = [tmp_path / "one_DDB", tmp_path / "two_DDB"]
    for path in inputs:
        path.write_text("data")
    result = LocalExecutor(abinit_bin=str(abinit)).merge_abinit_ddb(inputs, tmp_path / "out_DDB", tmp_path)
    assert result.success
    assert (tmp_path / "out_DDB").read_text() == "merged"
