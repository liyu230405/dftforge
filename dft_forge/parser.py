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

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

RY_TO_EV = 13.605693122994


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
        
        # Number of K-points
        kpts_match = re.search(r"number of k points\s+=\s+(\d+)", stdout)
        if kpts_match:
            result.k_points = int(kpts_match.group(1))
        
        # Number of bands (from &SYSTEM nbnd)
        nbnd_match = re.search(r"nbnd\s+=\s+(\d+)", stdout)
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
            result.converged = True
        
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
        
        # Forces: find all force lines directly; QE may or may not wrap them in
        # a "Forces acting on atoms" header, so we avoid relying on that block.
        force_lines = re.findall(
            r"atom\s+\d+.*?force\s+=\s*([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)",
            stdout
        )
        if force_lines:
            all_forces = []
            for fx, fy, fz in force_lines:
                all_forces.extend([abs(float(fx)), abs(float(fy)), abs(float(fz))])
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
    def parse_bands(xml_path: Path) -> ParsedBands:
        """Parse band structure from QE bands XML output."""
        result = ParsedBands()
        
        if not xml_path.exists():
            return result
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            ns = {'qes': 'http://www.quantum-espresso.org/ns/qes/qes-1.0'}
            
            # Get band structure data
            bs_elem = root.find('.//qes:band_structure', ns)
            if bs_elem is not None:
                # Number of bands and kpoints
                nks = int(bs_elem.get('nks', 0))
                nbnd = int(bs_elem.get('nbnd', 0))
                result.n_kpoints = nks
                result.n_bands = nbnd
                
                # Parse eigenvalues
                eigenvalues = []
                for ks_elem in bs_elem.findall('qes:ks_energies', ns):
                    evals = []
                    for e in ks_elem.findall('qes:eigenvalues', ns):
                        vals = [float(x) for x in e.text.split()] if e.text else []
                        evals.append(vals)
                    if evals:
                        eigenvalues.append(evals[0])
                
                if eigenvalues:
                    result.eigenvalues = np.array(eigenvalues) * RY_TO_EV
                    
                    # Find Fermi level from highest occupied at Gamma
                    # Simple heuristic: find gap or identify metallic
                    if len(eigenvalues) > 0:
                        all_evals = result.eigenvalues.flatten()
                        # Sort and look for gaps
                        sorted_evals = np.sort(all_evals)
                        # Fermi level ≈ highest occupied band
                        result.fermi_energy_ev = sorted_evals[len(sorted_evals) // 2]
                        
                        # Check for band gap (simple check: look for gap > 0.1 eV)
                        diffs = np.diff(sorted_evals)
                        large_gaps = np.where(diffs > 0.1)[0]
                        if len(large_gaps) > 0:
                            result.band_gap_ev = float(diffs[large_gaps[0]])
                        else:
                            result.is_metal = True
                            result.band_gap_ev = 0.0
        
        except Exception:
            pass
        
        return result

    @staticmethod
    def parse_dos(dos_file: Path) -> ParsedDOS:
        """Parse DOS from QE dos.x output file."""
        result = ParsedDOS()
        
        if not dos_file.exists():
            return result
        
        try:
            data = np.loadtxt(dos_file)
            if data.ndim == 2 and data.shape[1] >= 2:
                energies = data[:, 0] * RY_TO_EV  # Convert Ry to eV
                dos = data[:, 1]
                result.energies = energies
                result.dos = dos
                result.n_energy_points = len(energies)
                
                # Find Fermi level (energy where DOS changes behavior)
                # Simple: use middle of energy range for insulators
                mid_idx = len(energies) // 2
                result.fermi_energy_ev = energies[mid_idx]
                
                # DOS at Fermi level
                result.dos_at_fermi = float(np.interp(result.fermi_energy_ev, energies, dos))
        except Exception:
            pass
        
        return result

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
            
            output = root.find('output', ns)
            if output is None:
                return result
            
            # Total energy
            etot_elem = output.find('qes:total_energy/qes:etot', ns)
            if etot_elem is not None:
                result["total_energy_ry"] = float(etot_elem.text)
            
            # Cell parameters
            atomic_struct = output.find('qes:atomic_structure', ns)
            if atomic_struct is not None:
                alat = float(atomic_struct.get('alat', 0))
                ibrav = int(atomic_struct.get('bravais_index', 0))
                
                cell_elem = atomic_struct.find('qes:cell', ns)
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
        """Compute lattice constants from Cartesian cell vectors."""
        if ibrav in (1, 2, 3):  # Cubic
            if ibrav == 2:  # FCC: conventional cubic constant = norm * sqrt(2)
                alat = np.linalg.norm(cell_vectors[0]) * np.sqrt(2)
            else:
                alat = np.linalg.norm(cell_vectors[0])
            return (alat, alat, alat)
        elif ibrav == 4:  # Hexagonal
            a = np.linalg.norm(cell_vectors[0])
            c = np.linalg.norm(cell_vectors[2])
            return (a, a, c)
        else:
            a = np.linalg.norm(cell_vectors[0])
            b = np.linalg.norm(cell_vectors[1])
            c = np.linalg.norm(cell_vectors[2])
            return (a, b, c)

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
