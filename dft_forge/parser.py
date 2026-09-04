"""QE Parser: deterministic extraction of physical quantities from QE output.

Parses stdout (regex) and XML fallback for:
- Total energy (Ry, eV/atom)
- SCF convergence info
- Forces (Ry/Bohr)
- Pressure (kbar)
- Cell parameters (Bohr)
- Fermi energy
- Band structure data (T2)
- DOS data (T2)
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

RY_TO_EV = 13.605693122994
HA_TO_EV = 27.211386245988
HA_TO_RY = HA_TO_EV / RY_TO_EV  # 2.0

logger = logging.getLogger(__name__)


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class ParsedSCF:
    """Parsed SCF convergence information."""
    total_energy_ry: float = 0.0
    total_energy_ev: float = 0.0
    n_iterations: int = 0
    converged: bool = False
    convergence_thr: float = 0.0
    final_delta: float = 0.0
    fermi_energy_ev: float = 0.0
    k_points: int = 0
    n_bands: int = 0


@dataclass
class ParsedCell:
    """Parsed cell parameters."""
    a_bohr: float = 0.0
    b_bohr: float = 0.0
    c_bohr: float = 0.0
    ibrav: int = 0
    volume_bohr3: float = 0.0


@dataclass
class ParsedVCResult:
    """Parsed vc-relax final result."""
    final_energy_ry: float = 0.0
    final_energy_ev_per_atom: float = 0.0
    max_force_ry_bohr: float = 0.0
    pressure_kbar: float = 0.0
    n_iterations: int = 0
    converged: bool = False
    cell: Optional[ParsedCell] = None
    space_group: Optional[str] = None
    natoms: int = 0


@dataclass
class ParsedBands:
    """Parsed band structure data (T2)."""
    n_bands: int = 0
    n_kpoints: int = 0
    fermi_energy_ev: float = 0.0
    band_gap_ev: float = 0.0
    is_metal: bool = False
    k_labels: List[str] = field(default_factory=list)
    # eigenvalues: array of shape (n_kpoints, n_bands)
    eigenvalues: Optional[np.ndarray] = None


@dataclass
class ParsedDOS:
    """Parsed DOS data (T2)."""
    fermi_energy_ev: float = 0.0
    n_energy_points: int = 0
    dos_at_fermi: float = 0.0
    # energy and dos arrays
    energies: Optional[np.ndarray] = None
    dos: Optional[np.ndarray] = None


@dataclass
class ParsedPDOS:
    """Parsed projected DOS from projwfc.x output files."""
    fermi_energy_ev: float = 0.0
    n_energy_points: int = 0
    # element -> summed projected DOS on a common energy grid
    energies: Optional[np.ndarray] = None
    element_dos: Optional[Dict[str, np.ndarray]] = None
    # atom index (1-based, QE order) -> element
    atom_elements: Optional[List[str]] = None


# ── Main parser class ─────────────────────────────────────────────────────────

class QEParser:
    """Deterministic QE output parser.
    
    Parses stdout via regex first, falls back to XML if needed.
    """

    @staticmethod
    def parse_scf(stdout: str) -> ParsedSCF:
        """Parse SCF convergence from stdout."""
        result = ParsedSCF()
        
        # Convergence message
        conv_match = re.search(
            r"convergence has been achieved in\s+(\d+)\s+iterations", stdout
        )
        if conv_match:
            result.n_iterations = int(conv_match.group(1))
            result.converged = True
        
        # Total energy (last occurrence)
        energy_matches = re.findall(
            r"!\s+total energy\s+=\s+(-?\d+\.\d+)\s+Ry", stdout
        )
        if energy_matches:
            result.total_energy_ry = float(energy_matches[-1])
            result.total_energy_ev = result.total_energy_ry * RY_TO_EV
        
        # Convergence threshold
        thr_match = re.search(
            r"convergence threshold\s+=\s+(\d+\.\d+E[+-]?\d+)", stdout
        )
        if thr_match:
            result.convergence_thr = float(thr_match.group(1))
        
        # Fermi energy
        fermi_match = re.search(
            r"the Fermi energy is\s+(-?\d+\.\d+)\s+ev", stdout, re.IGNORECASE
        )
        if fermi_match:
            result.fermi_energy_ev = float(fermi_match.group(1))
        
        # Number of K-points — QE 7.5 prints "number of k points=   63" (no
        # space before =); older/newer variants use " = "
        kpts_match = re.search(r"number of k points\s*=\s*(\d+)", stdout)
        if kpts_match:
            result.k_points = int(kpts_match.group(1))

        # Number of bands — stdout never echoes nbnd; it prints the resolved
        # "number of Kohn-Sham states" line instead
        nbnd_match = re.search(r"number of Kohn-Sham states\s*=\s*(\d+)", stdout)
        if nbnd_match:
            result.n_bands = int(nbnd_match.group(1))
        
        return result

    @staticmethod
    def parse_vc_relax(stdout: str, xml_path: Optional[Path] = None) -> ParsedVCResult:
        """Parse vc-relax output from stdout, with XML fallback."""
        result = ParsedVCResult()
        
        # Bravaiss index
        ibrav_match = re.search(r"bravais-lattice index\s+=\s+(\d+)", stdout)
        ibrav = int(ibrav_match.group(1)) if ibrav_match else 0
        
        # Energy (last SCF step energy)
        energy_matches = re.findall(
            r"!\s+total energy\s+=\s+(-?\d+\.\d+)\s+Ry", stdout
        )
        if energy_matches:
            result.final_energy_ry = float(energy_matches[-1])
        # Electronic SCF convergence occurs at every ionic step and therefore
        # does not prove that the geometry optimization converged.  Require a
        # geometry-specific BFGS/final-coordinate marker.
        geometry_done = bool(
            re.search(
                r"bfgs converged in|End of BFGS Geometry Optimization",
                stdout,
                re.IGNORECASE,
            )
        )
        geometry_failed = bool(
            re.search(
                r"maximum number of steps has been reached|"
                r"geometry optimization did not converge",
                stdout,
                re.IGNORECASE,
            )
        )
        result.converged = geometry_done and not geometry_failed
        
        # Convergence message
        conv_matches = re.findall(
            r"convergence has been achieved in\s+(\d+)\s+iterations", stdout
        )
        if conv_matches:
            result.n_iterations = int(conv_matches[-1])
        
        # Pressure (last occurrence, QE 7.5 format: "P=  37.54")
        pressure_matches = re.findall(r"P=\s+([\d.-]+)", stdout)
        if pressure_matches:
            result.pressure_kbar = float(pressure_matches[-1])
        
        # Forces: only the FINAL configuration matters — the whole-trajectory
        # max includes early steps far from equilibrium and misjudges a
        # converged relax as failed (verifier compares against a threshold)
        final_marker = "Begin final coordinates"
        search_text = stdout.split(final_marker)[-1] if final_marker in stdout else stdout
        force_blocks = list(re.finditer(
            r"(?:^[ \t]*atom[ \t]+\d+.*?force[ \t]*=[ \t]*[-\d.eE+]+[ \t]+[-\d.eE+]+[ \t]+[-\d.eE+]+[ \t]*$\n?)+",
            search_text,
            re.MULTILINE,
        ))
        if force_blocks:
            all_forces = []
            for line in force_blocks[-1].group(0).splitlines():
                m = re.search(
                    r"force[ \t]*=[ \t]*([-\d.eE+]+)[ \t]+([-\d.eE+]+)[ \t]+([-\d.eE+]+)",
                    line,
                )
                if m:
                    components = [float(g) for g in m.groups()]
                    all_forces.append(sum(v * v for v in components) ** 0.5)
            if all_forces:
                result.max_force_ry_bohr = max(all_forces)
        
        # Cell parameters (last block)
        cell_blocks = re.findall(
            r"CELL_PARAMETERS\s+\(alat=\s*([\d.]+)\)\s*\n"
            r"\s*([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s*\n"
            r"\s*([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s*\n"
            r"\s*([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)",
            stdout
        )
        if cell_blocks:
            last = cell_blocks[-1]
            alat = float(last[0])
            a1 = np.array([float(last[1]), float(last[2]), float(last[3])]) * alat
            a2 = np.array([float(last[4]), float(last[5]), float(last[6])]) * alat
            a3 = np.array([float(last[7]), float(last[8]), float(last[9])]) * alat
            a, b, c = QEParser._compute_cell_constants(
                np.array([a1, a2, a3]), ibrav
            )
            volume = abs(np.dot(a1, np.cross(a2, a3)))
            result.cell = ParsedCell(
                a_bohr=a, b_bohr=b, c_bohr=c,
                ibrav=ibrav, volume_bohr3=volume,
            )
        
        # Number of atoms
        nat_match = re.search(r"number of atoms/cell\s+=\s+(\d+)", stdout)
        if nat_match:
            result.natoms = int(nat_match.group(1))
        
        # Energy per atom
        if result.final_energy_ry != 0 and result.natoms > 0:
            result.final_energy_ev_per_atom = result.final_energy_ry * RY_TO_EV / result.natoms
        
        # XML fallback
        if xml_path and xml_path.exists():
            xml_data = QEParser._parse_xml_quick(xml_path)
            if xml_data.get("total_energy_ry") and result.final_energy_ry == 0.0:
                result.final_energy_ry = xml_data["total_energy_ry"]
                result.final_energy_ev_per_atom = (
                    result.final_energy_ry * RY_TO_EV / result.natoms if result.natoms > 0 else 0.0
                )
            if xml_data.get("cell") and result.cell is None:
                result.cell = xml_data["cell"]

        return result

    @staticmethod
    def extract_final_structure(stdout: str):
        """Final relaxed structure from vc-relax stdout.

        QE prints the last CELL_PARAMETERS + ATOMIC_POSITIONS after
        'Begin final coordinates'. Cell vectors are in bohr (alat units);
        positions respect their own declared unit.
        Returns ASE Atoms (Å) or None.
        """
        BOHR = 0.529177210903
        _num = r"-?\d+\.?\d*(?:[eE][+-]?\d+)?"
        cell_blocks = re.findall(
            r"CELL_PARAMETERS\s+\(\w+\s*=\s*([\d.]+)\)\s*\n"
            rf"((?:\s*{_num}\s+{_num}\s+{_num}\s*\n){{3}})",
            stdout,
        )
        pos_blocks = re.findall(
            r"ATOMIC_POSITIONS\s*[({]\s*(\w+)\s*[=)}][^\n]*\n"
            rf"((?:\s*[A-Za-z]{{1,2}}\s+{_num}\s+{_num}\s+{_num}[^\n]*\n)+)",
            stdout,
        )
        if not cell_blocks or not pos_blocks:
            return None

        alat_bohr = float(cell_blocks[-1][0])
        cell_bohr = np.array([
            [float(x) for x in line.split()]
            for line in cell_blocks[-1][1].strip().splitlines()
        ]) * alat_bohr
        cell_ang = cell_bohr * BOHR

        mode = pos_blocks[-1][0].lower()
        symbols, cart_ang = [], []
        for line in pos_blocks[-1][1].strip().splitlines():
            parts = line.split()
            if len(parts) < 4 or parts[0][0].isdigit():
                continue
            symbols.append(parts[0])
            xyz = np.array([float(x) for x in parts[1:4]])
            if mode == "crystal":
                cart_ang.append(xyz @ cell_ang)
            elif mode == "alat":
                cart_ang.append(xyz * alat_bohr * BOHR)
            elif mode in ("bohr", "au"):
                cart_ang.append(xyz * BOHR)
            else:  # angstrom
                cart_ang.append(xyz)

        if not symbols:
            return None

        from ase import Atoms

        return Atoms(symbols, positions=np.array(cart_ang), cell=cell_ang, pbc=True)

    @staticmethod
    def parse_bands(xml_path: Path) -> ParsedBands:
        """Parse band structure from QE XML (works with QE 6.x and 7.x schemas).

        QE 7.x: no nks/nbnd attributes on <band_structure>, eigenvalues and
        fermi_energy in Hartree atomic units.
        """
        result = ParsedBands()
        HA_TO_EV = 27.211386245988

        if not xml_path.exists():
            return result

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            ns = {'qes': 'http://www.quantum-espresso.org/ns/qes/qes-1.0'}

            def find_all(elem, local: str):
                # QE 7.x emits only the root with the qes: prefix; children
                # are namespace-less — try both spellings
                found = elem.findall(f'.//{{{ns["qes"]}}}{local}')
                if found:
                    return found
                return elem.findall(f'.//{local}')

            def find_one(elem, local: str):
                found = find_all(elem, local)
                return found[0] if found else None

            bs_elem = find_one(root, 'band_structure')
            if bs_elem is None:
                return result

            # Parse eigenvalues (Hartree in QE 6.x/7.x data-file-schema)
            eigenvalues = []
            for ks_elem in find_all(bs_elem, 'ks_energies'):
                for e in find_all(ks_elem, 'eigenvalues'):
                    vals = [float(x) for x in e.text.split()] if e.text else []
                    if vals:
                        eigenvalues.append(vals)

            if not eigenvalues:
                return result

            result.eigenvalues = np.array(eigenvalues) * HA_TO_EV
            result.n_kpoints = int(bs_elem.get('nks', 0)) or len(eigenvalues)
            result.n_bands = int(bs_elem.get('nbnd', 0)) or len(eigenvalues[0])

            # Fermi level from XML (Hartree) — fallback: highest occupied
            fermi_elem = find_one(root, 'fermi_energy')
            hoc_elem = find_one(root, 'highestOccupiedLevel')
            fermi_ev = None
            if fermi_elem is not None and fermi_elem.text:
                fermi_ev = float(fermi_elem.text) * HA_TO_EV
            elif hoc_elem is not None and hoc_elem.text:
                fermi_ev = float(hoc_elem.text) * HA_TO_EV

            all_evals = result.eigenvalues.flatten()
            if fermi_ev is None:
                fermi_ev = float(np.max(all_evals) / 2.0)
            result.fermi_energy_ev = fermi_ev

            # Gap via Fermi level: VBM = max eval <= fermi, CBM = min eval > fermi
            occupied = all_evals[all_evals <= fermi_ev + 1e-9]
            empty = all_evals[all_evals > fermi_ev + 1e-9]
            if len(occupied) == 0 or len(empty) == 0:
                result.band_gap_ev = 0.0
                result.is_metal = False
            else:
                vbm = float(np.max(occupied))
                cbm = float(np.min(empty))
                if cbm > vbm:
                    result.band_gap_ev = cbm - vbm
                    # gaps below 2×degauss (0.01 Ry ≈ 0.27 eV) are smaller than
                    # the smearing resolution — the system is effectively metal
                    result.is_metal = result.band_gap_ev < 0.272
                else:
                    result.band_gap_ev = 0.0
                    result.is_metal = True

        except Exception as exc:
            # swallow-and-continue is what let empty results "pass" before:
            # log loudly so a broken XML is visible, verifier will fail it
            logger.warning("parse_bands failed for %s: %s", xml_path, exc)

        return result

    @staticmethod
    def parse_dos(dos_file: Path) -> ParsedDOS:
        """Parse DOS from QE dos.x output file."""
        result = ParsedDOS()
        
        if not dos_file.exists():
            return result
        
        try:
            header = ""
            with open(dos_file) as fh:
                for _ in range(3):
                    line = fh.readline()
                    if not line:
                        break
                    header += line
            data = np.loadtxt(dos_file)
            if data.ndim == 2 and data.shape[1] >= 2:
                # QE dos.x writes the first column in eV (header: "E (eV)").
                # Multiplying by RY_TO_EV here inflates the whole axis ×13.6.
                energies = data[:, 0]
                dos = data[:, 1]
                result.energies = energies
                result.dos = dos
                result.n_energy_points = len(energies)

                # Fermi level is stated in the file header ("EFermi = ... eV");
                # the old midpoint fallback put it at a meaningless energy
                m = re.search(r"EFermi\s*=\s*([-\d.Ee+]+)", header)
                if m:
                    result.fermi_energy_ev = float(m.group(1))
                else:
                    result.fermi_energy_ev = None

                if result.fermi_energy_ev is not None:
                    # DOS at Fermi level
                    result.dos_at_fermi = float(np.interp(result.fermi_energy_ev, energies, dos))
        except Exception as exc:
            logger.warning("parse_dos failed for %s: %s", dos_file, exc)

        return result

    @staticmethod
    def parse_pdos(workdir: Path, prefix: str) -> ParsedPDOS:
        """Parse projwfc.x per-atom pdos files, summed per element.

        Atom->element mapping comes from the QE XML in the .save dir; if the
        XML is unavailable, from the pw stdout 'site n.' table.
        """
        result = ParsedPDOS()
        atm_files = sorted(workdir.glob(f"{prefix}.pdos_atm*"))
        if not atm_files:
            return result

        # atom index -> element: try QE XML (nscf wrote it into a .save dir
        # that projwfc read); fall back to projwfc stdout table
        atom_elements: Dict[int, str] = {}
        xml_candidates = list(workdir.glob("*.xml")) + list(workdir.glob(f"{prefix}.save/*.xml"))
        for xml in xml_candidates:
            atom_elements = QEParser._xml_atom_species(xml)
            if atom_elements:
                result.fermi_energy_ev = QEParser._xml_fermi_ev(xml)
                break
        if not atom_elements:
            for out in workdir.glob("*.out"):
                try:
                    atom_elements = QEParser._stdout_atom_species(out.read_text())
                except OSError:
                    continue
                if atom_elements:
                    break

        energies: Optional[np.ndarray] = None
        elem_dos: Dict[str, np.ndarray] = {}
        for f in atm_files:
            m = re.search(r"atm#(\d+)\(", f.name)
            if not m:
                continue
            atom_idx = int(m.group(1))
            try:
                data = np.loadtxt(f)
            except Exception:
                continue
            if data.ndim != 2 or data.shape[1] < 2:
                continue
            e = data[:, 0]  # pdos_atm files are already in eV ("# E (eV)")
            ldos = data[:, 1]
            if energies is None:
                energies = e
                result.n_energy_points = len(e)
            elif len(e) == len(energies):
                ldos = ldos[: len(energies)]
            else:
                continue
            el = atom_elements.get(atom_idx)
            if el is None:
                continue
            elem_dos[el] = elem_dos.get(el, np.zeros_like(energies)) + ldos

        if energies is not None and elem_dos:
            result.energies = energies
            result.element_dos = elem_dos
            result.atom_elements = [atom_elements.get(i, "?") for i in range(1, max(atom_elements) + 1)] if atom_elements else None
        return result

    @staticmethod
    def _xml_atom_species(xml_path: Path) -> Dict[int, str]:
        """1-based atom index -> element symbol from QE XML."""
        out: Dict[int, str] = {}
        try:
            root = ET.parse(xml_path).getroot()
        except Exception:
            return out
        ns = {"qes": "http://www.quantum-espresso.org/ns/qes/qes-1.0"}

        def find_all(elem, local: str):
            found = elem.findall(f".//{{{ns['qes']}}}{local}")
            return found or elem.findall(f".//{local}")

        atoms = find_all(root, "atom")
        for i, a in enumerate(atoms, start=1):
            name = a.get("name") or a.get("species")
            if name:
                out[i] = name.capitalize() if len(name) > 1 else name.upper()
        return out

    @staticmethod
    def _xml_fermi_ev(xml_path: Path) -> float:
        """Fermi level in eV from a QE XML (Hartree internally)."""
        try:
            root = ET.parse(xml_path).getroot()
            for elem in root.iter("fermi_energy"):
                if elem.text:
                    return float(elem.text) * 27.211386245988
            for elem in root.iter("highestOccupiedLevel"):
                if elem.text:
                    return float(elem.text) * 27.211386245988
        except Exception:
            pass
        return 0.0

    @staticmethod
    def _stdout_atom_species(stdout: str) -> Dict[int, str]:
        """1-based atom index -> element from pw stdout 'site n.' PAW block."""
        out: Dict[int, str] = {}
        # QE prints a table: "site n.     atom      ... " followed by rows
        # like "   1          Na ..."
        m = re.search(
            r"site n\.\s+atom\s+.*?\n((?:\s+\d+\s+[A-Za-z]{1,2}\s.*\n)+)",
            stdout,
        )
        if not m:
            return out
        for line in m.group(1).strip().splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0].isdigit():
                sym = parts[1]
                out[int(parts[0])] = sym.capitalize() if len(sym) > 1 else sym.upper()
        return out

    @staticmethod
    def parse_xml(xml_path: Path) -> Dict[str, Any]:
        """Parse general info from QE XML output."""
        result = {}
        
        if not xml_path.exists():
            return result
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            ns = {'qes': 'http://www.quantum-espresso.org/ns/qes/qes-1.0'}

            def _find(parent, path):
                """QE writes child elements UNPREFIXED under a prefixed root
                (<qes:espresso><output>…), but some tools emit prefixed tags —
                try both or every lookup silently misses and the whole XML
                fallback is dead."""
                hit = parent.find(path)
                if hit is not None:
                    return hit
                return parent.find("qes:" + path.replace("/", "/qes:"), ns)

            output = _find(root, "output")
            if output is None:
                # tolerate a document whose root IS the output element
                output = root if root.tag.split("}")[-1] == "output" else None
            if output is None:
                return result

            # Total energy — QE XML writes etot in Hartree, NOT Ry
            etot_elem = _find(output, "total_energy/etot")
            if etot_elem is not None:
                result["total_energy_ry"] = float(etot_elem.text) * HA_TO_RY

            # Cell parameters
            atomic_struct = _find(output, "atomic_structure")
            if atomic_struct is not None:
                alat = float(atomic_struct.get('alat', 0))
                ibrav = int(atomic_struct.get('bravais_index', 0))

                cell_elem = _find(atomic_struct, "cell")
                if cell_elem is not None and alat > 0:
                    a1 = np.array([float(x) for x in cell_elem[0].text.split()]) * alat
                    a2 = np.array([float(x) for x in cell_elem[1].text.split()]) * alat
                    a3 = np.array([float(x) for x in cell_elem[2].text.split()]) * alat
                    a, b, c = QEParser._compute_cell_constants(
                        np.array([a1, a2, a3]), ibrav
                    )
                    result["cell"] = ParsedCell(
                        a_bohr=a, b_bohr=b, c_bohr=c,
                        ibrav=ibrav,
                        volume_bohr3=abs(np.dot(a1, np.cross(a2, a3))),
                    )
                
                nat = atomic_struct.get('nat')
                if nat:
                    result["natoms"] = int(nat)
        
        except Exception as e:
            result["error"] = str(e)
        
        return result

    @staticmethod
    def _parse_xml_quick(xml_path: Path) -> Dict[str, Any]:
        """Quick XML parser for fallback data."""
        return QEParser.parse_xml(xml_path)

    @staticmethod
    def _compute_cell_constants(
        cell_vectors: np.ndarray, ibrav: int = 0
    ) -> tuple:
        """Compute lattice constants from Cartesian cell vectors.

        For ibrav=0 (free cell) the input is often a primitive fcc/bcc cell;
        detect it from vector angles so `a` matches the conventional cubic
        lattice constant users expect (e.g. NaCl rock-salt ~5.6 A, not 4.0 A).
        """
        if ibrav in (1, 2, 3):  # Cubic
            if ibrav == 2:  # FCC: conventional cubic constant = norm * sqrt(2)
                alat = np.linalg.norm(cell_vectors[0]) * np.sqrt(2)
            elif ibrav == 3:  # BCC: conventional cubic constant = norm * sqrt(4/3)
                alat = np.linalg.norm(cell_vectors[0]) * 2.0 / np.sqrt(3.0)
            else:
                alat = np.linalg.norm(cell_vectors[0])
            return (alat, alat, alat)
        elif ibrav == 4:  # Hexagonal
            a = np.linalg.norm(cell_vectors[0])
            c = np.linalg.norm(cell_vectors[2])
            return (a, a, c)
        elif ibrav == 0:
            angles = QEParser._cell_angles_deg(cell_vectors)
            if angles is not None:
                a_n = np.linalg.norm(cell_vectors[0])
                b_n = np.linalg.norm(cell_vectors[1])
                c_n = np.linalg.norm(cell_vectors[2])
                lengths_equal = (
                    abs(a_n - b_n) / a_n < 0.02 and abs(a_n - c_n) / a_n < 0.02
                )
                if lengths_equal:
                    # fcc primitive: all inter-vector angles 60 deg
                    if all(abs(ang - 60.0) < 2.0 for ang in angles):
                        conv = a_n * np.sqrt(2)
                        return (conv, conv, conv)
                    # bcc primitive: all inter-vector angles ~109.47 deg
                    if all(abs(ang - 109.47) < 2.0 for ang in angles):
                        conv = a_n * 2.0 / np.sqrt(3.0)
                        return (conv, conv, conv)
            a = np.linalg.norm(cell_vectors[0])
            b = np.linalg.norm(cell_vectors[1])
            c = np.linalg.norm(cell_vectors[2])
            return (a, b, c)
        else:
            a = np.linalg.norm(cell_vectors[0])
            b = np.linalg.norm(cell_vectors[1])
            c = np.linalg.norm(cell_vectors[2])
            return (a, b, c)

    @staticmethod
    def _cell_angles_deg(cell_vectors: np.ndarray) -> Optional[tuple]:
        """Inter-vector angles (alpha, beta, gamma) in degrees, or None if degenerate."""
        a1, a2, a3 = cell_vectors
        norms = [np.linalg.norm(v) for v in (a1, a2, a3)]
        if min(norms) < 1e-8:
            return None
        alpha = np.degrees(np.arccos(np.clip(np.dot(a2, a3) / (norms[1] * norms[2]), -1, 1)))
        beta = np.degrees(np.arccos(np.clip(np.dot(a1, a3) / (norms[0] * norms[2]), -1, 1)))
        gamma = np.degrees(np.arccos(np.clip(np.dot(a1, a2) / (norms[0] * norms[1]), -1, 1)))
        return (alpha, beta, gamma)

    @staticmethod
    def save_parsed(result: Any, output_file: Path) -> None:
        """Save parsed result to JSON."""
        import json
        data = {
            "final_energy_ry": getattr(result, 'final_energy_ry', 0.0),
            "final_energy_ev_per_atom": getattr(result, 'final_energy_ev_per_atom', 0.0),
            "max_force_ry_bohr": getattr(result, 'max_force_ry_bohr', 0.0),
            "pressure_kbar": getattr(result, 'pressure_kbar', 0.0),
            "n_iterations": getattr(result, 'n_iterations', 0),
            "converged": getattr(result, 'converged', False),
            "natoms": getattr(result, 'natoms', 0),
        }
        if getattr(result, 'cell', None):
            data["cell"] = {
                "a_bohr": result.cell.a_bohr,
                "b_bohr": result.cell.b_bohr,
                "c_bohr": result.cell.c_bohr,
            }
        output_file.write_text(json.dumps(data, indent=2))
