"""Deterministic VASP output parser.

Prefers vasprun.xml (structured, authoritative) and falls back to OUTCAR
regex. No LLM in the parsing path. Energies are reported in eV, VASP's native
unit, so no Ry conversion is applied.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
from xml.etree import ElementTree as ET


@dataclass
class ParsedVASP:
    """Parsed VASP SCF / relaxation result (energies in eV)."""

    total_energy_ev: float = 0.0
    converged: bool = False
    n_ionic_steps: int = 0
    n_electronic_steps: int = 0
    fermi_energy_ev: float = 0.0
    final_volume_ang3: float = 0.0
    max_force_ev_ang: float = 0.0
    forces_converged: bool = False
    total_magnetization_muB: Optional[float] = None


@dataclass
class ParsedBands:
    """Band-structure summary from EIGENVAL / vasprun.xml eigenvalues."""

    n_bands: int = 0
    n_kpoints: int = 0
    band_gap_ev: Optional[float] = None
    is_metal: bool = False
    fermi_energy_ev: Optional[float] = None
    vbm_ev: Optional[float] = None
    cbm_ev: Optional[float] = None


@dataclass
class ParsedDOS:
    """Total (and optionally element-projected) density of states."""

    n_energy_points: int = 0
    fermi_energy_ev: Optional[float] = None
    dos_at_fermi: Optional[float] = None
    energies_ev: Optional[List[float]] = None
    dos: Optional[List[float]] = None
    element_dos: Optional[dict] = None


@dataclass
class ParsedPhonon:
    """Phonon frequencies from a DFPT (IBRION=8) OUTCAR, in THz."""

    frequencies_thz: List[float] = field(default_factory=list)
    n_modes: int = 0
    has_imaginary: bool = False


@dataclass
class ParsedVASPError:
    """A recognized VASP failure, mapped to an actionable category."""

    category: str = ""
    message: str = ""


def parse_outcar(text: str) -> ParsedVASP:
    """Regex parse of OUTCAR stdout/file text."""
    result = ParsedVASP()

    energies = re.findall(
        r"free  energy   TOTEN\s*=\s*(-?\d+\.\d+)\s*eV", text
    )
    if energies:
        result.total_energy_ev = float(energies[-1])

    # Each "aborting loop because EDIFF is reached" marks a converged
    # electronic step; the count of ionic iterations is separate.
    result.converged = "reached required accuracy" in text or bool(
        re.search(r"aborting loop because EDIFF is reached", text)
    )
    result.n_electronic_steps = len(
        re.findall(r"aborting loop because EDIFF is reached", text)
    ) or len(re.findall(r"Iteration\s+\d+\(\s*\d+\)", text))
    ionic = re.findall(r"Iteration\s+(\d+)\(\s*\d+\)", text)
    if ionic:
        result.n_ionic_steps = max(int(i) for i in ionic)

    fermi = re.search(r"E-fermi\s*:\s*(-?\d+\.\d+)", text)
    if fermi:
        result.fermi_energy_ev = float(fermi.group(1))

    vol = re.findall(r"volume of cell\s*:\s*(\d+\.\d+)", text)
    if vol:
        result.final_volume_ang3 = float(vol[-1])

    # Max residual force from the last "TOTAL-FORCE" block (columns 4-6 are the
    # force components in eV/Angstrom; columns 1-3 are positions).
    force_blocks = list(re.finditer(
        r"TOTAL-FORCE \(eV/Angst\)\s*\n\s*-+\n(.*?)\n\s*-+", text, re.DOTALL
    ))
    if force_blocks:
        max_f = 0.0
        for row in force_blocks[-1].group(1).strip().splitlines():
            nums = row.split()
            if len(nums) >= 6:
                fx, fy, fz = (float(nums[3]), float(nums[4]), float(nums[5]))
                max_f = max(max_f, (fx * fx + fy * fy + fz * fz) ** 0.5)
        result.max_force_ev_ang = max_f

    mag = re.findall(r"number of electron\s+[-\d.]+\s+magnetization\s+(-?\d+\.\d+)", text)
    if mag:
        result.total_magnetization_muB = float(mag[-1])

    return result


def parse_vasprun(path: Path) -> ParsedVASP:
    """Structured parse of vasprun.xml (authoritative when present)."""
    result = ParsedVASP()
    tree = ET.parse(str(path))
    root = tree.getroot()

    energies = root.findall(".//calculation/energy/i[@name='e_fr_energy']")
    if energies and energies[-1].text:
        result.total_energy_ev = float(energies[-1].text)

    calculations = root.findall(".//calculation")
    result.n_ionic_steps = len(calculations)
    if calculations:
        scsteps = calculations[-1].findall("scstep")
        result.n_electronic_steps = len(scsteps)

    fermi = root.find(".//calculation/dos/i[@name='efermi']")
    if fermi is not None and fermi.text:
        result.fermi_energy_ev = float(fermi.text)

    vol = root.findall(".//calculation/structure/crystal/i[@name='volume']")
    if vol and vol[-1].text:
        result.final_volume_ang3 = float(vol[-1].text)

    # vasprun.xml records convergence implicitly: an NELM-capped run leaves an
    # extra scstep beyond the required accuracy. Treat presence of a final
    # calculation energy plus fewer-than-NELM electronic steps as converged.
    nelm_node = root.find(".//parameters//i[@name='NELM']")
    nelm = int(nelm_node.text) if nelm_node is not None and nelm_node.text else None
    if result.total_energy_ev != 0.0:
        result.converged = nelm is None or result.n_electronic_steps < nelm
    return result


def parse_result(workdir: Path, stdout: str = "") -> ParsedVASP:
    """Prefer vasprun.xml, fall back to OUTCAR file, then stdout text."""
    vasprun = Path(workdir) / "vasprun.xml"
    if vasprun.is_file():
        try:
            return parse_vasprun(vasprun)
        except ET.ParseError:
            pass  # truncated/aborted run — fall back to text
    outcar = Path(workdir) / "OUTCAR"
    if outcar.is_file():
        return parse_outcar(outcar.read_text(errors="ignore"))
    return parse_outcar(stdout)


def parse_forces(workdir: Path) -> Optional[List[List[float]]]:
    """Full per-atom force array (eV/Å) from vasprun.xml, else OUTCAR.

    Returns a list of [fx, fy, fz] in POSCAR atom order — the input phonopy
    needs to build force constants. None when no forces are recoverable.
    """
    vasprun = Path(workdir) / "vasprun.xml"
    if vasprun.is_file():
        try:
            root = ET.parse(str(vasprun)).getroot()
            calcs = root.findall(".//calculation")
            if calcs:
                varray = calcs[-1].find("varray[@name='forces']")
                if varray is not None:
                    rows = []
                    for v in varray.findall("v"):
                        if v.text:
                            rows.append([float(x) for x in v.text.split()])
                    if rows:
                        return rows
        except ET.ParseError:
            pass
    outcar = Path(workdir) / "OUTCAR"
    text = outcar.read_text(errors="ignore") if outcar.is_file() else ""
    blocks = list(re.finditer(
        r"TOTAL-FORCE \(eV/Angst\)\s*\n\s*-+\n(.*?)\n\s*-+", text, re.DOTALL
    ))
    if blocks:
        rows = []
        for row in blocks[-1].group(1).strip().splitlines():
            nums = row.split()
            if len(nums) >= 6:
                rows.append([float(nums[3]), float(nums[4]), float(nums[5])])
        if rows:
            return rows
    return None


def parse_bands(workdir: Path) -> ParsedBands:
    """Band structure from vasprun.xml eigenvalues (preferred) or EIGENVAL.

    Computes the fundamental gap as CBM - VBM relative to the Fermi level:
    occupied states sit at/below E_F, empty states above. A negative or zero
    gap (bands crossing E_F) is reported as metallic.
    """
    result = ParsedBands()
    vasprun = Path(workdir) / "vasprun.xml"
    eigenvalues: List[List[float]] = []  # per-kpoint list of band energies
    if vasprun.is_file():
        try:
            root = ET.parse(str(vasprun)).getroot()
            fermi = root.find(".//calculation/dos/i[@name='efermi']")
            if fermi is not None and fermi.text:
                result.fermi_energy_ev = float(fermi.text)
            # eigenvalues live under calculation/eigenvalues/array/set/set(spin)/set(kpoint)
            spin_sets = root.findall(".//calculation/eigenvalues/array/set/set")
            if spin_sets:
                for kset in spin_sets[0].findall("set"):
                    row = []
                    for r in kset.findall("r"):
                        if r.text:
                            row.append(float(r.text.split()[0]))
                    if row:
                        eigenvalues.append(row)
        except ET.ParseError:
            pass
    if not eigenvalues:
        eig = Path(workdir) / "EIGENVAL"
        if eig.is_file():
            eigenvalues, ef = _parse_eigenval(eig)
            if result.fermi_energy_ev is None:
                result.fermi_energy_ev = ef
    if not eigenvalues:
        return result
    result.n_kpoints = len(eigenvalues)
    result.n_bands = max(len(row) for row in eigenvalues)
    ef = result.fermi_energy_ev if result.fermi_energy_ev is not None else 0.0
    occupied = [e for row in eigenvalues for e in row if e <= ef + 1e-6]
    empty = [e for row in eigenvalues for e in row if e > ef + 1e-6]
    if occupied and empty:
        result.vbm_ev = max(occupied)
        result.cbm_ev = min(empty)
        gap = result.cbm_ev - result.vbm_ev
        # A band crossing E_F on some k-point makes VBM≈CBM≈E_F -> metal.
        result.is_metal = gap <= 0.05
        result.band_gap_ev = 0.0 if result.is_metal else round(gap, 4)
    else:
        result.is_metal = True
        result.band_gap_ev = 0.0
    return result


def _parse_eigenval(path: Path) -> tuple:
    """Minimal EIGENVAL reader: returns (per-kpoint band energies, None)."""
    lines = path.read_text(errors="ignore").splitlines()
    if len(lines) < 7:
        return [], None
    header = lines[5].split()
    try:
        nkpts, nbands = int(header[1]), int(header[2])
    except (IndexError, ValueError):
        return [], None
    eigenvalues: List[List[float]] = []
    idx = 6
    for _ in range(nkpts):
        while idx < len(lines) and not lines[idx].strip():
            idx += 1
        idx += 1  # skip the k-point coordinate line
        row = []
        for _ in range(nbands):
            if idx >= len(lines):
                break
            parts = lines[idx].split()
            if len(parts) >= 2:
                row.append(float(parts[1]))
            idx += 1
        if row:
            eigenvalues.append(row)
    return eigenvalues, None


def parse_dos(workdir: Path, projected: bool = False) -> ParsedDOS:
    """Total (and optionally element-projected) DOS from DOSCAR.

    DOSCAR layout: 6 header lines, then NEDOS total-DOS rows
    (energy  dos  integrated_dos [spin-down cols]); if LORBIT=11, per-ion
    projected blocks follow, one block of NEDOS rows per atom.
    """
    result = ParsedDOS()
    doscar = Path(workdir) / "DOSCAR"
    if not doscar.is_file():
        return result
    lines = doscar.read_text(errors="ignore").splitlines()
    if len(lines) < 7:
        return result
    natoms = int(lines[0].split()[0]) if lines[0].split() else 0
    header = lines[5].split()
    try:
        nedos = int(header[2])
        result.fermi_energy_ev = float(header[3])
    except (IndexError, ValueError):
        return result
    energies, dos = [], []
    for i in range(6, min(6 + nedos, len(lines))):
        parts = lines[i].split()
        if len(parts) >= 2:
            energies.append(float(parts[0]))
            dos.append(float(parts[1]))
    result.n_energy_points = len(energies)
    result.energies_ev = energies
    result.dos = dos
    if result.fermi_energy_ev is not None and energies:
        import bisect

        j = min(bisect.bisect_left(energies, result.fermi_energy_ev), len(dos) - 1)
        result.dos_at_fermi = dos[j] if dos else None

    if projected and natoms > 0:
        result.element_dos = _parse_projected_dos(lines, workdir, natoms, nedos)
    return result


def _parse_projected_dos(lines: List[str], workdir: Path, natoms: int, nedos: int) -> Optional[dict]:
    """Sum per-ion projected DOS into element totals, mapping ions via POSCAR."""
    symbols = _poscar_symbols(Path(workdir) / "POSCAR")
    if not symbols or len(symbols) < natoms:
        return None
    element_dos: dict = {}
    cursor = 6 + nedos  # skip header + total-DOS block
    for atom_idx in range(natoms):
        if cursor >= len(lines):
            break
        cursor += 1  # each projected block starts with its own header line
        summed = None
        count = 0
        for i in range(cursor, min(cursor + nedos, len(lines))):
            parts = lines[i].split()
            if len(parts) < 2:
                continue
            orbital_sum = sum(float(x) for x in parts[1:])
            if summed is None:
                summed = []
            if count < len(summed):
                summed[count] += orbital_sum
            else:
                summed.append(orbital_sum)
            count += 1
        cursor += nedos
        el = symbols[atom_idx]
        if summed is not None:
            if el not in element_dos:
                element_dos[el] = summed
            else:
                element_dos[el] = [a + b for a, b in zip(element_dos[el], summed)]
    return element_dos or None


def _poscar_symbols(poscar: Path) -> List[str]:
    """Expand POSCAR line-6 species + line-7 counts into a per-ion symbol list."""
    if not poscar.is_file():
        return []
    lines = poscar.read_text(errors="ignore").splitlines()
    if len(lines) < 7:
        return []
    species = lines[5].split()
    try:
        counts = [int(x) for x in lines[6].split()]
    except ValueError:
        return []
    symbols: List[str] = []
    for el, n in zip(species, counts):
        symbols.extend([el] * n)
    return symbols


def parse_phonon(text: str) -> ParsedPhonon:
    """Parse DFPT phonon frequencies from an IBRION=8 OUTCAR.

    VASP prints lines like ``   1 f  =   12.345678 THz ...`` for real modes and
    ``   4 f/i=    2.100000 THz`` for imaginary ones (f/i marks instability).
    """
    result = ParsedPhonon()
    freqs: List[float] = []
    for m in re.finditer(r"\d+\s+(f\s*=|f/i\s*=)\s+(-?\d+\.\d+)\s+THz", text):
        value = float(m.group(2))
        if "f/i" in m.group(1):
            result.has_imaginary = True
            freqs.append(-value)  # convention: imaginary modes as negative THz
        else:
            freqs.append(value)
    result.frequencies_thz = freqs
    result.n_modes = len(freqs)
    return result


_ERROR_SIGNATURES = [
    ("ZBRENT", "ionic_minimizer", "ZBRENT: fatal error in bracketing — relaxation step failed"),
    ("Sub-Space-Matrix", "electronic_instability", "subspace matrix not Hermitian — electronic instability"),
    ("EDDDAV", "electronic_instability", "Davidson iterative matrix diagonalization failed"),
    ("BRMIX", "charge_mixing", "charge density mixing warning/failure (BRMIX)"),
    ("VERY BAD NEWS", "geometry", "very bad news: reciprocal/real lattice inconsistency"),
    ("POSCAR", "input", "POSCAR parsing error"),
]


def classify_error(text: str) -> Optional[ParsedVASPError]:
    """Map a known VASP failure token to an actionable category."""
    for token, category, message in _ERROR_SIGNATURES:
        if token in text:
            return ParsedVASPError(category=category, message=message)
    return None
