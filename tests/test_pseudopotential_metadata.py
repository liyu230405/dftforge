"""Electron-count regressions for recommendations and the QE compiler."""

import hashlib

import pytest

from dft_forge.compiler import DEFAULT_PSEUDO_DIR, QECompiler
from dft_forge.compiler import pseudopotentials
from dft_forge.compiler.pseudopotentials import read_pseudo_metadata


def test_actual_si_xml_header_and_radial_cutoff():
    path = DEFAULT_PSEUDO_DIR / "Si_r.upf"
    metadata = read_pseudo_metadata(path)
    assert metadata["z_valence"] == 4
    assert metadata["pseudo_type"] == "NC"
    assert metadata["functional"] == "PBE"
    assert metadata["filename"] == path.name
    assert metadata["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert "suggested_ecutwfc" not in metadata
    assert "suggested_ecutrho" not in metadata


def test_old_gbrv_header_and_zero_recommendations():
    metadata = read_pseudo_metadata(DEFAULT_PSEUDO_DIR / "Si.upf")
    assert metadata["z_valence"] == 4
    assert metadata["pseudo_type"] == "US"
    assert "PBE" in metadata["functional"]
    assert "suggested_ecutwfc" not in metadata
    assert "suggested_ecutrho" not in metadata


@pytest.mark.parametrize("xml", [True, False])
def test_fortran_exponents_and_explicit_cutoff_recommendations(tmp_path, xml):
    path = tmp_path / "test.upf"
    if xml:
        path.write_text('<UPF><PP_HEADER z_valence="4.0D+00" pseudo_type="NC" '
                        'wfc_cutoff="5.0d1" rho_cutoff="2.0D2" /></UPF>')
    else:
        path.write_text('<PP_HEADER>\n4.0D+00 Z valence\n'
                        '5.0d1 2.0D2 Suggested cutoff for wfc and rho\n</PP_HEADER>')
    metadata = read_pseudo_metadata(path)
    assert metadata["z_valence"] == 4
    assert metadata["suggested_ecutwfc"] == 50
    assert metadata["suggested_ecutrho"] == 200


@pytest.mark.parametrize("value", ["", "not-a-number", "NaN", "Inf", "-Inf", "0", "-2"])
@pytest.mark.parametrize("xml", [True, False])
def test_invalid_valence_fails_clearly(tmp_path, value, xml):
    path = tmp_path / "invalid.upf"
    path.write_text(f'<PP_HEADER z_valence="{value}"/>' if xml else
                    f"<PP_HEADER>\n{value} Z valence\n</PP_HEADER>")
    with pytest.raises(ValueError, match="invalid.upf.*invalid z_valence"):
        read_pseudo_metadata(path)


def test_unknown_file_is_not_assumed_zero_electrons(tmp_path):
    path = tmp_path / "unknown.upf"
    path.write_text("unrecognized pseudopotential")
    with pytest.raises(ValueError, match="z_valence"):
        read_pseudo_metadata(path)


def test_gth_ionic_charge_is_valence():
    metadata = read_pseudo_metadata(DEFAULT_PSEUDO_DIR / "O-q6.gth")
    assert metadata["z_valence"] == 6
    assert metadata["pseudo_type"] == "GTH"


def test_file_replacement_does_not_reuse_old_electron_count(tmp_path):
    path = tmp_path / "changed.upf"
    path.write_text('<PP_HEADER z_valence="4"/>')
    compiler = QECompiler(tmp_path)
    before = read_pseudo_metadata(path)
    assert compiler._pseudo_z_valence(path.name) == 4
    path.write_text('<PP_HEADER z_valence="6"/>')
    after = read_pseudo_metadata(path)
    assert compiler._pseudo_z_valence(path.name) == 6
    assert before["sha256"] != after["sha256"]


def test_file_mutation_during_read_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / "changing.upf"
    path.write_text('<PP_HEADER z_valence="4"/>')
    original_fstat = pseudopotentials.os.fstat
    calls = 0

    def mutate_before_second_stat(fd):
        nonlocal calls
        calls += 1
        if calls == 2:
            path.write_text('<PP_HEADER z_valence="12"/>')
        return original_fstat(fd)

    monkeypatch.setattr(pseudopotentials.os, "fstat", mutate_before_second_stat)
    with pytest.raises(ValueError, match="changed while reading"):
        read_pseudo_metadata(path)


@pytest.mark.parametrize("kwargs, expected", [
    ({}, 12),
    ({"tot_charge": -4}, 14),
    ({"tot_charge": 2}, 11),
    ({"spin_polarized": True}, 16),
    ({"magnetic_order": "FM", "tot_charge": -2}, 18),
    ({"magnetic_order": "AFM"}, 16),
])
def test_si_automatic_bands_account_for_valence_charge_and_spin(tmp_path, kwargs, expected):
    from dft_forge.compiler.qe_parameters import resolve_qe_parameters

    compiler = QECompiler(DEFAULT_PSEUDO_DIR)
    content = compiler.compile_scf("Si", tmp_path / "input.in", **kwargs)
    assert f"  nbnd = {expected}\n" in content
    preview = resolve_qe_parameters(material="Si", overrides=kwargs)
    assert preview["parameters"]["nbnd"] == expected


@pytest.mark.parametrize("kwargs", [
    {"nbnd": 3},
    {"nbnd": 4, "tot_charge": -2},
    {"nbnd": 7, "spin_polarized": True},
    {"nbnd": 4.5},
    {"nbnd": True},
])
def test_insufficient_or_noninteger_bands_fail_before_writing(tmp_path, kwargs):
    output = tmp_path / "input.in"
    with pytest.raises(ValueError, match="nbnd must be an integer"):
        QECompiler(DEFAULT_PSEUDO_DIR).compile_scf("Si", output, **kwargs)
    assert not output.exists()


@pytest.mark.parametrize("charge", [float("nan"), float("inf"), 8, 9])
def test_unphysical_electron_count_is_rejected(tmp_path, charge):
    with pytest.raises(ValueError, match="tot_charge"):
        QECompiler(DEFAULT_PSEUDO_DIR).compile_scf("Si", tmp_path / "input.in", tot_charge=charge)


def test_mgo_combines_upf_and_gth_valence(tmp_path):
    content = QECompiler(DEFAULT_PSEUDO_DIR).compile_scf("MgO", tmp_path / "input.in")
    assert "  nbnd = 12\n" in content  # Mg 2 + O 6 electrons, four occupied + eight empty.
