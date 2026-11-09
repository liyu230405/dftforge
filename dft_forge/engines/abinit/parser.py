"""Deterministic ABINIT output parser.

Parses the ``.abo`` main output (plain text, format is public and stable) plus
anaddb phonon output. No LLM in the parsing path. ABINIT reports energies in
Hartree and forces in Ha/Bohr; both are converted to eV / eV·Å⁻¹ here so the
engine speaks the same units as QE (eV/atom) and VASP (eV).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# CODATA 2018, shared with dft_forge/parser.py (QE) — keep in sync.
HA_TO_EV = 27.211386245988
HA_BOHR_TO_EV_ANG = HA_TO_EV / 0.52917721067  # Ha/Bohr -> eV/Å
HA_TO_THZ = 6579.683920502  # 1 Hartree -> THz (E/h)
BOHR_TO_ANG = 0.52917721067
NUMBER = r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][+-]?\d+)?"


@dataclass
class ParsedABINIT:
    """Parsed ABINIT SCF / relaxation result (energies in eV)."""

    total_energy_ev: Optional[float] = None
    converged: bool = False
    n_scf_steps: int = 0
    fermi_energy_ev: float = 0.0
    max_force_ev_ang: Optional[float] = None
    final_volume_ang3: float = 0.0
    total_magnetization_muB: Optional[float] = None
    ionic_converged: bool = False


@dataclass
class ParsedABINITBands:
    """Band-structure summary from ABINIT eigenvalues (eV)."""

    n_bands: int = 0
    n_kpoints: int = 0
    band_gap_ev: Optional[float] = None
    is_metal: bool = False
    fermi_energy_ev: Optional[float] = None
    vbm_ev: Optional[float] = None
    cbm_ev: Optional[float] = None


@dataclass
class ParsedABINITDOS:
    """Total (and optionally projected) density of states."""

    n_energy_points: int = 0
    fermi_energy_ev: Optional[float] = None
    dos_at_fermi: Optional[float] = None
    energies_ev: Optional[List[float]] = None
    dos: Optional[List[float]] = None
    element_dos: Optional[dict] = None


@dataclass
class ParsedABINITPhonon:
    """Phonon frequencies from anaddb output, in THz."""

    frequencies_thz: List[float] = field(default_factory=list)
    n_modes: int = 0
    has_imaginary: bool = False


@dataclass
class ParsedABINITError:
    """A recognized ABINIT failure mapped to an actionable category."""

    category: str = ""
    message: str = ""


def _find_abo(workdir: Path) -> Optional[Path]:
    """Locate the main ABINIT text output (.abo / .out / .abo0) in workdir."""
    workdir = Path(workdir)
    for pattern in ("*.abo", "*.abo0", "*.out"):
        matches = sorted(workdir.glob(pattern))
        # Skip files that are clearly anaddb inputs/outputs.
        matches = [m for m in matches if "anaddb" not in m.name]
        if matches:
            return matches[0]
    return None


def parse_scf(text: str) -> ParsedABINIT:
    """Parse total energy / convergence / forces from .abo text.

    ABINIT prints ``etotal`` (Hartree) in the final results block, e.g.::

        etotal   -8.1234567890E+00
    and SCF convergence as ``... converged ...`` / ``after N iterations``.
    """
    result = ParsedABINIT()

    # etotal (Hartree). Prefer the standalone "etotal" line; fall back to
    # ">>>>> ... Etotal= ..." in the SCF cycle report.
    etot = re.findall(r"\betotal\d*\s+(-?\d+\.\d+(?:[eEdD][+-]?\d+)?)", text)
    if not etot:
        etot = re.findall(r"[Ee]total\s*=\s*(-?\d+\.\d+(?:[eEdD][+-]?\d+)?)", text)
    if etot:
        result.total_energy_ev = _f(etot[-1]) * HA_TO_EV

    # Convergence markers ABINIT emits at the end of an SCF cycle.
    result.converged = bool(
        re.search(r"SCF cycle converged", text)
        or re.search(r"converged\s*:.*nres2", text)
        or re.search(r"At SCF step\s+\d+.*converged", text)
        or re.search(r"etotal.*converged", text, re.IGNORECASE)
    )
    failures = list(re.finditer(r"(?:not enough scf cycles to converge|SCF cycle did not converge)", text, re.I))
    successes = list(re.finditer(r"At SCF step[^\n]*converged|SCF cycle converged", text))
    if failures and (not successes or failures[-1].start() > successes[-1].start()):
        result.converged = False
    result.ionic_converged = bool(re.search(r"gradients (?:are|have) converged", text, re.I))
    steps = re.findall(r"ETOT\s+(\d+)\s+", text)
    if steps:
        result.n_scf_steps = max(int(s) for s in steps)

    fermi = re.search(r"Fermi\s*\(?or HOMO\)?\s*energy.*?=\s*(-?\d+\.\d+(?:[eEdD][+-]?\d+)?)", text)
    if not fermi:
        fermi = re.search(r"newocc.*?Fermi energy.*?=\s*(-?\d+\.\d+(?:[eEdD][+-]?\d+)?)", text)
    if fermi:
        result.fermi_energy_ev = _f(fermi.group(1)) * HA_TO_EV

    # Max residual force (Ha/Bohr) from the "Cartesian forces" / fcart block.
    forces = _parse_forces_habohr(text)
    if forces:
        result.max_force_ev_ang = max(
            (fx * fx + fy * fy + fz * fz) ** 0.5 for fx, fy, fz in forces
        ) * HA_BOHR_TO_EV_ANG

    vol = re.findall(r"Unit cell volume ucvol=\s*(-?\d+\.\d+(?:[eEdD][+-]?\d+)?)", text)
    if vol:
        result.final_volume_ang3 = _f(vol[-1]) / (1.8897261246257702 ** 3)

    mag = re.findall(r"Magnetization \(Bohr magneton\)=\s*(-?\d+\.\d+(?:[eEdD][+-]?\d+)?)", text)
    if not mag:
        mag = re.findall(r"magnetization.*?=\s*(-?\d+\.\d+(?:[eEdD][+-]?\d+)?)", text)
    if mag:
        result.total_magnetization_muB = _f(mag[-1])

    return result


def _f(token: str) -> float:
    """Parse a Fortran float that may use 'D' exponents (1.0D-3)."""
    return float(token.replace("D", "e").replace("d", "e"))


def _parse_forces_habohr(text: str) -> List[List[float]]:
    """Extract the last Cartesian force block (Ha/Bohr) from .abo text."""
    # The header line may carry trailing text, e.g.
    # "cartesian forces (hartree/bohr) at end:"; match to end-of-line then rows.
    headers = list(re.finditer(
        r"cartesian forces (?:\(hartree/bohr\)|\(fcart\) \[Ha/bohr\])[^\n]*\n",
        text, re.I))
    if not headers:
        return []
    rows = []
    for line in text[headers[-1].end():].splitlines():
        match = re.fullmatch(rf"\s*\d+\s+({NUMBER})\s+({NUMBER})\s+({NUMBER})\s*", line)
        if not match:
            break
        rows.append([_f(x) for x in match.groups()])
    return rows


def read_netcdf(path: Path, names):
    """Read both HDF5/NetCDF4 and classic NetCDF without a new dependency."""
    import h5py
    import numpy as np
    if h5py.is_hdf5(path):
        with h5py.File(path, "r") as handle:
            return {key: np.array(handle[key]) for key in names if key in handle}
    from scipy.io import netcdf_file
    with netcdf_file(path, "r", mmap=False) as handle:
        return {key: handle.variables[key].data.copy() for key in names if key in handle.variables}


def read_final_atoms(workdir: Path):
    """Read the final GSR geometry; ETSF primitive vectors are in Bohr."""
    from ase import Atoms
    paths = sorted(Path(workdir).glob("*_GSR.nc"))
    if not paths:
        raise ValueError("ABINIT produced no final *_GSR.nc structure")
    data = read_netcdf(paths[-1], ["primitive_vectors", "reduced_atom_positions",
                                 "atomic_numbers", "atom_species"])
    numbers = data["atomic_numbers"].astype(int)[data["atom_species"].astype(int) - 1]
    return Atoms(numbers=numbers, cell=data["primitive_vectors"] * BOHR_TO_ANG,
                 scaled_positions=data["reduced_atom_positions"], pbc=True)


def parse_result(workdir: Path, stdout: str = "") -> ParsedABINIT:
    """Prefer the .abo file, fall back to stdout text."""
    abo = _find_abo(workdir)
    result = parse_scf(abo.read_text(errors="ignore") if abo is not None else stdout)
    paths = sorted(Path(workdir).glob("*_GSR.nc"))
    if paths:
        import numpy as np
        data = read_netcdf(paths[-1], ["etotal", "cartesian_forces", "primitive_vectors", "fermi_energy"])
        if "etotal" in data:
            result.total_energy_ev = float(data["etotal"]) * HA_TO_EV
        if "cartesian_forces" in data:
            result.max_force_ev_ang = float(np.linalg.norm(data["cartesian_forces"], axis=1).max()) * HA_BOHR_TO_EV_ANG
        if "primitive_vectors" in data:
            result.final_volume_ang3 = float(abs(np.linalg.det(data["primitive_vectors"]))) * BOHR_TO_ANG ** 3
        if "fermi_energy" in data:
            result.fermi_energy_ev = float(data["fermi_energy"]) * HA_TO_EV
    return result


def parse_bands(workdir: Path, stdout: str = "") -> ParsedABINITBands:
    """Band structure from ABINIT eigenvalues (Hartree -> eV).

    ABINIT prints per-k-point eigenvalue blocks like::

        Eigenvalues (hartree) for nkpt= 10  k points:
         kpt#   1 ... nband= 8, ...
           -0.20  0.15  0.15  ...
    The fundamental gap is CBM - VBM relative to the Fermi/HOMO level; a gap
    that vanishes (bands crossing E_F) is reported metallic.
    """
    result = ParsedABINITBands()
    paths = sorted(Path(workdir).glob("*_GSR.nc"))
    if paths:
        import numpy as np
        data = read_netcdf(paths[-1], ["eigenvalues", "occupations", "fermi_energy", "number_of_electrons", "occopt"])
        eig = data["eigenvalues"] * HA_TO_EV
        result.n_kpoints, result.n_bands = eig.shape[-2:]
        if "fermi_energy" in data:
            result.fermi_energy_ev = float(data["fermi_energy"]) * HA_TO_EV
        occ = data.get("occupations")
        full = 2.0 if eig.shape[0] == 1 else 1.0
        if occ is not None and occ.shape == eig.shape:
            filled = occ > full - 1e-5
            empty = occ < 1e-5
            if bool(np.any(~(filled | empty))):
                result.is_metal = True
                result.band_gap_ev = 0.0
            elif filled.any() and empty.any():
                result.vbm_ev = float(eig[filled].max())
                result.cbm_ev = float(eig[empty].min())
                gap = result.cbm_ev - result.vbm_ev
                result.is_metal = gap <= 0
                result.band_gap_ev = max(0.0, gap)
        return result
    abo = _find_abo(workdir)
    eigfiles = sorted(Path(workdir).glob("*_EIG"))
    text = eigfiles[-1].read_text(errors="ignore") if eigfiles else (abo.read_text(errors="ignore") if abo is not None else stdout)
    if not text:
        return result

    # Collect eigenvalue rows per k-point from the "(hartree)" eigenvalue block.
    eig_section = re.search(
        r"[Ee]igenvalues \(hartree\).*?(?=\n\s*\n\s*[A-Z]|\Z)", text, re.DOTALL
    )
    eigenvalues: List[List[float]] = []
    if eig_section:
        current: List[float] = []
        for line in eig_section.group(0).splitlines():
            if re.search(r"kpt#|kpt |k points:", line):
                if current:
                    eigenvalues.append(current)
                    current = []
                continue
            nums = re.findall(r"-?\d+\.\d+(?:[eEdD][+-]?\d+)?", line)
            # eigenvalue rows are pure number rows (no '=' assignments)
            if nums and "=" not in line:
                current.extend(_f(n) for n in nums)
        if current:
            eigenvalues.append(current)

    fermi = re.search(r"Fermi\s*\(?or HOMO\)?\s*energy.*?=\s*(-?\d+\.\d+(?:[eEdD][+-]?\d+)?)", text)
    ef_ha = _f(fermi.group(1)) if fermi else None
    if ef_ha is not None:
        result.fermi_energy_ev = ef_ha * HA_TO_EV

    if not eigenvalues:
        return result
    # Convert to eV.
    eigenvalues = [[e * HA_TO_EV for e in row] for row in eigenvalues]
    result.n_kpoints = len(eigenvalues)
    result.n_bands = max(len(row) for row in eigenvalues)
    ef = result.fermi_energy_ev if result.fermi_energy_ev else 0.0
    occupied = [e for row in eigenvalues for e in row if e <= ef + 1e-6]
    empty = [e for row in eigenvalues for e in row if e > ef + 1e-6]
    if occupied and empty:
        result.vbm_ev = max(occupied)
        result.cbm_ev = min(empty)
        gap = result.cbm_ev - result.vbm_ev
        result.is_metal = gap <= 0.05
        result.band_gap_ev = 0.0 if result.is_metal else round(gap, 4)
    else:
        result.is_metal = True
        result.band_gap_ev = 0.0
    return result


def parse_dos(workdir: Path, projected: bool = False, atom_symbols=None) -> ParsedABINITDOS:
    """Total DOS from an ABINIT ``*_DOS`` file (energies in Hartree).

    ABINIT DOS files carry a commented header then rows of
    ``energy  DOS  integrated_DOS`` in Hartree. With prtdos 3, per-atom
    ``*_DOS_AT####`` files hold projected DOS.
    """
    result = ParsedABINITDOS()
    workdir = Path(workdir)
    dos_files = sorted(workdir.glob("*_DOS")) + sorted(workdir.glob("*_DOS_TOTAL")) + sorted(workdir.glob("*_DOS0*"))
    dos_files = [f for f in dos_files if "_AT" not in f.name]
    if not dos_files:
        return result
    energies, dos = [], []
    fermi_ha = None
    for line in dos_files[0].read_text(errors="ignore").splitlines():
        s = line.strip()
        if s.startswith("#"):
            m = re.search(r"Fermi energy.*?:\s*(-?\d+\.\d+(?:[eEdD][+-]?\d+)?)", s)
            if m:
                fermi_ha = _f(m.group(1))
            continue
        nums = re.findall(r"-?\d+\.\d+(?:[eEdD][+-]?\d+)?", s)
        if len(nums) >= 2:
            energies.append(_f(nums[0]) * HA_TO_EV)
            dos.append(_f(nums[1]) / HA_TO_EV)
    if not energies:
        return result
    result.n_energy_points = len(energies)
    result.energies_ev = energies
    result.dos = dos
    if fermi_ha is not None:
        result.fermi_energy_ev = fermi_ha * HA_TO_EV
        import bisect

        j = min(bisect.bisect_left(energies, result.fermi_energy_ev), len(dos) - 1)
        result.dos_at_fermi = dos[j] if dos else None

    if projected:
        result.element_dos = _parse_projected_dos(workdir, atom_symbols, energies)
    return result


def _parse_projected_dos(workdir: Path, atom_symbols=None, energies=None) -> Optional[dict]:
    """Sum per-atom ``*_DOS_AT####`` files into element totals."""
    at_files = sorted(Path(workdir).glob("*_DOS_AT*"))
    if not at_files:
        return None
    # Map atom index -> element via the .abi typat/znucl if available; else
    # label by file. Keep it simple and robust: group all atoms as their
    # symbol parsed from an adjacent structure is unavailable here, so sum
    # everything under the file-derived tag.
    element_dos: dict = {}
    for f in at_files:
        vals: List[float] = []
        grid: List[float] = []
        for line in f.read_text(errors="ignore").splitlines():
            if line.strip().startswith("#"):
                continue
            nums = re.findall(r"-?\d+\.\d+(?:[eEdD][+-]?\d+)?", line)
            if len(nums) >= 2:
                count = (len(nums) - 1) // 2 if len(nums) > 3 else 1
                vals.append(sum(_f(x) for x in nums[1:1 + count]) / HA_TO_EV)
                grid.append(_f(nums[0]) * HA_TO_EV)
        match = re.search(r"_DOS_AT(\d+)", f.name)
        index = int(match.group(1)) - 1 if match else -1
        if atom_symbols is not None:
            if not 0 <= index < len(atom_symbols):
                raise ValueError("PDOS atom index does not match input structure")
            tag = atom_symbols[index]
        else:
            tag = f.name.split("_DOS_AT")[0]
        if energies is not None and (len(grid) != len(energies) or any(abs(a-b) > 1e-7 for a,b in zip(grid, energies))):
            raise ValueError("PDOS energy grid differs from total DOS")
        if vals:
            element_dos[tag] = [a + b for a, b in zip(element_dos.get(tag, [0.0] * len(vals)), vals)]
    return element_dos or None


def parse_phonon(text: str = "", workdir: Optional[Path] = None) -> ParsedABINITPhonon:
    """Parse phonon frequencies from anaddb output.

    anaddb prints phonon frequencies in Hartree, e.g.::

        Phonon frequencies in Hartree :
          -1.0E-06  2.3E-04  2.3E-04 ...
    A negative frequency marks a dynamical instability (imaginary mode).
    """
    result = ParsedABINITPhonon()
    freqs: List[float] = []
    if workdir is not None:
        paths = sorted(Path(workdir).glob("*_PHBST.nc"))
        if paths:
            data = read_netcdf(paths[-1], ["phfreqs"])
            if "phfreqs" in data:
                # PHBST stores energies in eV, unlike the text Hartree block.
                freqs = [float(x) * HA_TO_THZ / HA_TO_EV for x in data["phfreqs"].reshape(-1)]
                result.frequencies_thz = freqs
                result.n_modes = len(freqs)
                result.has_imaginary = any(f < -1e-3 for f in freqs)
                return result
    sections = re.finditer(r"[Pp]honon (?:frequencies|energies) in Hartree[^\n]*\n", text)
    for sec in sections:
        for line in text[sec.end():].splitlines():
            if not re.fullmatch(rf"\s*{NUMBER}(?:\s+{NUMBER})*\s*", line):
                break
            freqs.extend(_f(n) * HA_TO_THZ for n in line.split())
    # Fallback: explicit THz or cm-1 blocks
    if not freqs:
        thz = re.search(r"in Thz\s*:?(.*?)(?=\n\s*\n|\Z)", text, re.DOTALL | re.IGNORECASE)
        if thz:
            freqs = [_f(n) for n in re.findall(r"-?\d+\.\d+(?:[eEdD][+-]?\d+)?", thz.group(1))]
    if freqs:
        result.frequencies_thz = freqs
        result.n_modes = len(freqs)
        result.has_imaginary = any(f < -1e-3 for f in freqs)
    return result


_ERROR_SIGNATURES = [
    ("Cannot find file", "config", "ABINIT could not find an input/pseudo file"),
    ("The perturbation", "verification", "DFPT perturbation setup error"),
    ("Dilatmx has been exceeded", "config", "cell expansion exceeded dilatmx"),
    ("SCF cycle did not converge", "verification", "SCF did not reach tolerance in nstep"),
    ("WARNING", "", ""),  # placeholder, filtered below
    ("Out of memory", "resource", "ABINIT ran out of memory"),
    ("ERROR", "execution", "ABINIT reported a fatal ERROR"),
]


def classify_error(text: str) -> Optional[ParsedABINITError]:
    """Map a known ABINIT failure token to an actionable category."""
    for token, category, message in _ERROR_SIGNATURES:
        if not category:
            continue
        if token in text:
            return ParsedABINITError(category=category, message=message)
    return None
