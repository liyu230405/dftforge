"""QE input compiler: deterministic generation of QE input files.

Uses ASE/pymatgen/spglib/seekpath for structure handling, k-points, and paths.
The LLM never writes shell commands or full QE input.
"""

from __future__ import annotations

import hashlib
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from ase import Atoms
from ase.calculators.espresso import Espresso
from ase.io import write as ase_write
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.kpath import KPathSeek


# ── Material database ─────────────────────────────────────────────────────────

MATERIAL_DB: Dict[str, Dict[str, Any]] = {
    "Si": {
        "formula": "Si",
        "structure_type": "diamond",
        "space_group": "Fd-3m",
        "ibrav": 2,
        "lattice_constant_angstrom": 5.43,
        "natoms_primitive": 2,
        "nspecies": 1,
        "species": ["Si"],
        "masses": [28.086],
        "pseudos": {"Si": "Si_r.upf"},
        "occupations": "smearing",
        "smearing": "mp",
        "degauss": 0.005,
        "ecutwfc_default": 25.0,
        "ecutrho_default": 200.0,
        "kpoints_default": (4, 4, 4, 1, 1, 1),
        "is_metal": False,
    },
    "Al": {
        "formula": "Al",
        "structure_type": "fcc",
        "space_group": "Fm-3m",
        "ibrav": 2,
        "lattice_constant_angstrom": 4.05,
        "natoms_primitive": 1,
        "nspecies": 1,
        "species": ["Al"],
        "masses": [26.982],
        "pseudos": {"Al": "Al.pbe-n-rrkjus_psl.1.0.2.UPF"},
        "occupations": "smearing",
        "smearing": "mp",
        "degauss": 0.02,
        "ecutwfc_default": 20.0,
        "ecutrho_default": 160.0,
        "kpoints_default": (8, 8, 8, 1, 1, 1),
        "is_metal": True,
    },
    "MgO": {
        "formula": "MgO",
        "structure_type": "nacl",
        "space_group": "Fm-3m",
        "ibrav": 2,
        "lattice_constant_angstrom": 4.21,
        "natoms_primitive": 2,
        "nspecies": 2,
        "species": ["Mg", "O"],
        "masses": [24.305, 15.999],
        "pseudos": {"Mg": "Mg.pz-n-vbc.UPF", "O": "O-q6.gth"},
        "occupations": "smearing",
        "smearing": "mp",
        "degauss": 0.01,
        "ecutwfc_default": 40.0,
        "ecutrho_default": 320.0,
        "kpoints_default": (4, 4, 4, 1, 1, 1),
        "is_metal": False,
        "input_dft": "pz",
    },
}


# ── Structure builder ─────────────────────────────────────────────────────────

def build_atoms(material: str) -> Atoms:
    """Build ASE Atoms object for a known material.

    Returns primitive cell structures suitable for QE ibrav=0 input.
    For FCC-based materials (Si, Al, MgO), the primitive cell vectors are:
        a1 = [0, a/2, a/2], a2 = [a/2, 0, a/2], a3 = [a/2, a/2, 0]
    where a is the conventional cubic lattice constant.
    """
    if material not in MATERIAL_DB:
        raise ValueError(f"Unknown material: {material}. Known: {list(MATERIAL_DB.keys())}")

    db = MATERIAL_DB[material]
    a_angstrom = db["lattice_constant_angstrom"]
    a_bohr = a_angstrom * 1.8897261246

    if material == "Si":
        # Diamond structure: primitive 2-atom cell
        # Lattice vectors for FCC primitive cell
        cell = [
            [0, a_bohr/2, a_bohr/2],
            [a_bohr/2, 0, a_bohr/2],
            [a_bohr/2, a_bohr/2, 0]
        ]
        # Positions in crystal coordinates of primitive cell
        pos = [
            [0, 0, 0],
            [0.25, 0.25, 0.25]
        ]
        atoms = Atoms("Si2", scaled_positions=pos, cell=cell, pbc=True)
        return atoms

    elif material == "Al":
        # FCC primitive cell: 1 atom
        cell = [
            [0, a_bohr/2, a_bohr/2],
            [a_bohr/2, 0, a_bohr/2],
            [a_bohr/2, a_bohr/2, 0]
        ]
        pos = [[0, 0, 0]]
        atoms = Atoms("Al", scaled_positions=pos, cell=cell, pbc=True)
        return atoms

    elif material == "MgO":
        # NaCl structure: primitive 2-atom cell (FCC lattice with 2-atom basis)
        cell = [
            [0, a_bohr/2, a_bohr/2],
            [a_bohr/2, 0, a_bohr/2],
            [a_bohr/2, a_bohr/2, 0]
        ]
        # Mg at (0,0,0), O at (0.5,0.5,0.5) in primitive coordinates
        pos = [
            [0, 0, 0],
            [0.5, 0.5, 0.5]
        ]
        atoms = Atoms("MgO", scaled_positions=pos, cell=cell, pbc=True)
        return atoms

    else:
        raise ValueError(f"Material '{material}' structure not implemented")


# ── K-point generation ────────────────────────────────────────────────────────

def get_kpoints(material: str, kpoints_override: Optional[Tuple] = None) -> Tuple:
    """Get k-point mesh for a material.
    
    Returns (nx, ny, nz, sx, sy, sz) tuple for QE K_POINTS automatic.
    """
    if kpoints_override:
        return kpoints_override
    
    db = MATERIAL_DB.get(material, {})
    return db.get("kpoints_default", (4, 4, 4, 1, 1, 1))


def get_high_symmetry_path(material: str) -> Tuple[np.ndarray, List[str]]:
    """Get high-symmetry k-path for band structure calculations (T2).
    
    Returns (kpoints_array, labels) where kpoints_array is (N, 3) in
    fractional coordinates of the primitive cell.
    """
    db = MATERIAL_DB.get(material)
    if not db:
        raise ValueError(f"Unknown material: {material}")
    
    a = db["lattice_constant_angstrom"]
    sg = db["space_group"]
    
    # Build primitive structure
    struct = build_atoms(material)
    
    # Use seekpath for high-symmetry path
    kpath = KPathSeek(struct).kpath
    
    kpoints = kpath["kpoints"]
    labels = kpath["path"][0]  # First path segment
    
    return kpoints, labels


# ── QE input generator ────────────────────────────────────────────────────────

class QECompiler:
    """Deterministic QE input file compiler.
    
    Takes a TaskSpec and material profile, produces a valid .in file.
    No LLM involved in input generation.
    """

    def __init__(self, pseudo_dir: Path):
        self.pseudo_dir = Path(pseudo_dir)
        self._validate_pseudo_dir()

    def _validate_pseudo_dir(self) -> None:
        if not self.pseudo_dir.exists():
            raise FileNotFoundError(f"pseudo_dir not found: {self.pseudo_dir}")

    def _check_pseudos(self, pseudos: Dict[str, str]) -> List[str]:
        """Check that all required pseudopotentials exist."""
        missing = []
        for element, filename in pseudos.items():
            if not (self.pseudo_dir / filename).exists():
                missing.append(f"{element}: {filename}")
        return missing

    def compile_t1(
        self,
        material: str,
        output_path: Path,
        *,
        prefix: Optional[str] = None,
        ecutwfc: Optional[float] = None,
        ecutrho: Optional[float] = None,
        kpoints: Optional[Tuple] = None,
        force_threshold: float = 1.0e-3,
        pressure_threshold: float = 0.5,
        conv_thr: float = 1.0e-8,
        cell_dynamics: str = "bfgs",
        nstep: int = 200,
        cell_optimization: bool = True,
    ) -> str:
        """Compile a T1 vc-relax input file.
        
        Args:
            material: Material key in MATERIAL_DB
            output_path: Where to write the .in file
            prefix: QE prefix (defaults to material lowercased)
            ecutwfc: Wavefunction cutoff in Ry
            ecutrho: Charge density cutoff in Ry
            kpoints: Override k-points tuple (nx, ny, nz, sx, sy, sz)
            force_threshold: Force convergence threshold Ry/Bohr
            pressure_threshold: Pressure threshold kbar
            conv_thr: SCF convergence threshold
            cell_dynamics: Cell dynamics algorithm
        
        Returns:
            The generated input file content as string
        """
        if material not in MATERIAL_DB:
            raise ValueError(f"Unknown material: {material}")
        
        db = MATERIAL_DB[material]
        atoms = build_atoms(material)
        
        prefix = prefix or material.lower()
        ecutwfc = ecutwfc if ecutwfc is not None else db["ecutwfc_default"]
        ecutrho = ecutrho if ecutrho is not None else db["ecutrho_default"]
        kpoints = get_kpoints(material, kpoints)
        
        # Check pseudos
        missing = self._check_pseudos(db["pseudos"])
        if missing:
            raise FileNotFoundError(f"Missing pseudopotentials: {missing}")
        
        # Build QE input
        lines = []
        calc_type = "vc-relax" if cell_optimization else "relax"
        lines.append("&CONTROL")
        lines.append(f'  calculation = "{calc_type}"')
        lines.append(f'  prefix = "{prefix}"')
        lines.append(f'  outdir = "./"')
        lines.append(f'  pseudo_dir = "{self.pseudo_dir}"')
        lines.append(f"  nstep = {nstep}")
        lines.append("/")
        lines.append("")
        lines.append("&SYSTEM")
        lines.append(f"  ibrav = {db['ibrav']}")
        alat_bohr = db["lattice_constant_angstrom"] * 1.8897261246
        lines.append(f"  celldm(1) = {alat_bohr:.6f}")
        lines.append(f"  nat = {len(atoms)}")
        lines.append(f"  ntyp = {db['nspecies']}")
        lines.append(f"  ecutwfc = {ecutwfc}")
        lines.append(f"  ecutrho = {ecutrho}")
        lines.append(f'  occupations = "{db["occupations"]}"')
        lines.append(f'  smearing = "{db["smearing"]}"')
        lines.append(f"  degauss = {db['degauss']}")
        if db.get("input_dft"):
            lines.append(f'  input_dft = "{db["input_dft"]}"')
        lines.append("/")
        lines.append("")
        lines.append("&ELECTRONS")
        lines.append(f"  conv_thr = {conv_thr:.1e}")
        lines.append("  mixing_beta = 0.7")
        lines.append("/")
        lines.append("")
        lines.append("&IONS")
        lines.append('  ion_dynamics = "bfgs"')
        lines.append("/")
        lines.append("")
        
        # CELL section: only for vc-relax (cell optimization)
        if cell_optimization:
            lines.append("&CELL")
            lines.append(f'  cell_dynamics = "{cell_dynamics}"')
            lines.append("  press = 0.0")
            lines.append("  wmass = 0.01")
            lines.append("/")
            lines.append("")
        
        # ATOMIC_SPECIES
        lines.append("ATOMIC_SPECIES")
        for elem, mass in zip(db["species"], db["masses"]):
            pseudo_file = db["pseudos"][elem]
            lines.append(f"  {elem}  {mass:.3f}  {pseudo_file}")
        lines.append("")
        
        # ATOMIC_POSITIONS (crystal) — primitive cell
        lines.append("ATOMIC_POSITIONS (crystal)")
        for sym, pos in zip(atoms.symbols, atoms.get_scaled_positions()):
            lines.append(f"  {sym}  {pos[0]:.8f}  {pos[1]:.8f}  {pos[2]:.8f}")
        lines.append("")
        
        # CELL_PARAMETERS: only for ibrav=0 (free cell)
        # For ibrav != 0, QE derives the cell from celldm(1) and bravais-lattice index
        if db["ibrav"] == 0:
            # atoms.cell is in Angstrom, so divide by lattice_constant_angstrom to get alat units
            lines.append("CELL_PARAMETERS (alat=1.0)")
            a_angstrom = db["lattice_constant_angstrom"]
            for vec in atoms.cell:
                lines.append(f"  {vec[0]/a_angstrom:.8f}  {vec[1]/a_angstrom:.8f}  {vec[2]/a_angstrom:.8f}")
            lines.append("")
        
        # K_POINTS
        lines.append("K_POINTS automatic")
        lines.append(f"  {kpoints[0]} {kpoints[1]} {kpoints[2]}  {kpoints[3]} {kpoints[4]} {kpoints[5]}")
        lines.append("")
        
        content = "\n".join(lines)
        
        # Write to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content)
        
        return content

    def compile_scf(
        self,
        material: str,
        output_path: Path,
        *,
        prefix: Optional[str] = None,
        calculation: str = "scf",
        ecutwfc: Optional[float] = None,
        ecutrho: Optional[float] = None,
        kpoints: Optional[Tuple] = None,
        conv_thr: float = 1.0e-8,
        nbnd: Optional[int] = None,
    ) -> str:
        """Compile a generic SCF or NSCF input file.
        
        Used for T2 bands/DOS steps.
        """
        if material not in MATERIAL_DB:
            raise ValueError(f"Unknown material: {material}")
        
        db = MATERIAL_DB[material]
        atoms = build_atoms(material)
        
        prefix = prefix or material.lower()
        ecutwfc = ecutwfc if ecutwfc is not None else db["ecutwfc_default"]
        ecutrho = ecutrho if ecutrho is not None else db["ecutrho_default"]
        kpoints = get_kpoints(material, kpoints)
        
        # Check pseudos
        missing = self._check_pseudos(db["pseudos"])
        if missing:
            raise FileNotFoundError(f"Missing pseudopotentials: {missing}")
        
        # Determine nbnd for metals
        if nbnd is None:
            if db["is_metal"]:
                # For metals, use more bands than valence electrons
                # Simple heuristic: 1.5x the number of occupied bands at Γ
                # For now, use a generous default
                nbnd = len(atoms) * 8
            else:
                nbnd = len(atoms) * 4
        
        alat_bohr = db["lattice_constant_angstrom"] * 1.8897261246
        
        lines = []
        lines.append("&CONTROL")
        lines.append(f'  calculation = "{calculation}"')
        lines.append(f'  prefix = "{prefix}"')
        lines.append(f'  outdir = "./"')
        lines.append(f'  pseudo_dir = "{self.pseudo_dir}"')
        if calculation in ("bands", "nscf"):
            lines.append('  restart_mode = "from_scratch"')
        lines.append("/")
        lines.append("")
        lines.append("&SYSTEM")
        lines.append(f"  ibrav = {db['ibrav']}")
        lines.append(f"  celldm(1) = {alat_bohr:.6f}")
        lines.append(f"  nat = {len(atoms)}")
        lines.append(f"  ntyp = {db['nspecies']}")
        lines.append(f"  ecutwfc = {ecutwfc}")
        lines.append(f"  ecutrho = {ecutrho}")
        lines.append(f'  occupations = "{db["occupations"]}"')
        lines.append(f'  smearing = "{db["smearing"]}"')
        lines.append(f"  degauss = {db['degauss']}")
        lines.append(f"  nbnd = {nbnd}")
        lines.append("/")
        lines.append("")
        lines.append("&ELECTRONS")
        lines.append(f"  conv_thr = {conv_thr:.1e}")
        lines.append("  mixing_beta = 0.7")
        lines.append("/")
        lines.append("")
        
        # ATOMIC_SPECIES
        lines.append("ATOMIC_SPECIES")
        for elem, mass in zip(db["species"], db["masses"]):
            pseudo_file = db["pseudos"][elem]
            lines.append(f"  {elem}  {mass:.3f}  {pseudo_file}")
        lines.append("")
        
        # ATOMIC_POSITIONS
        lines.append("ATOMIC_POSITIONS (crystal)")
        for sym, pos in zip(atoms.symbols, atoms.get_scaled_positions()):
            lines.append(f"  {sym}  {pos[0]:.8f}  {pos[1]:.8f}  {pos[2]:.8f}")
        lines.append("")
        
        # CELL_PARAMETERS
        lines.append("CELL_PARAMETERS (alat=1.0)")
        for vec in atoms.cell:
            lines.append(f"  {vec[0]/alat_bohr:.8f}  {vec[1]/alat_bohr:.8f}  {vec[2]/alat_bohr:.8f}")
        lines.append("")
        
        # K_POINTS — for NSCF/bands, this is overridden by the caller
        lines.append("K_POINTS automatic")
        lines.append(f"  {kpoints[0]} {kpoints[1]} {kpoints[2]}  {kpoints[3]} {kpoints[4]} {kpoints[5]}")
        lines.append("")
        
        content = "\n".join(lines)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content)
        return content

    def compile_bands_input(
        self,
        material: str,
        output_path: Path,
        *,
        prefix: Optional[str] = None,
        kpath: Optional[Tuple] = None,
        klabels: Optional[List[str]] = None,
        nkpoints: int = 100,
    ) -> str:
        """Compile a bands.x post-processing input file.

        Generates a high-symmetry k-path input suitable for bands.x.

        Args:
            material: Material key in MATERIAL_DB
            output_path: Where to write the .in file
            prefix: QE prefix (must match SCF/NSCF prefix)
            kpath: Override k-points as (N, 3) array in crystal coords
            klabels: Override k-point labels (one per k-point)
            nkpoints: Number of interpolated points between high-symmetry points

        Returns:
            The generated input file content as string
        """
        if material not in MATERIAL_DB:
            raise ValueError(f"Unknown material: {material}")

        db = MATERIAL_DB[material]
        prefix = prefix or material.lower()

        # Get high-symmetry path from seekpath if not provided
        if kpath is None or klabels is None:
            try:
                kpath_arr, klabels_list = get_high_symmetry_path(material)
                if kpath is None:
                    kpath = kpath_arr
                if klabels is None:
                    klabels = klabels_list
            except Exception:
                # Fallback: simple Gamma -> X -> M -> Gamma path for cubic
                kpath = [
                    [0.0, 0.0, 0.0],
                    [0.5, 0.0, 0.5],
                    [0.5, 0.5, 0.5],
                    [0.0, 0.0, 0.0],
                ]
                klabels = ["GAMMA", "X", "M", "GAMMA"]

        lines = []
        lines.append("&bands")
        lines.append(f'  prefix = "{prefix}"')
        lines.append('  outdir = "./"')
        lines.append(f"  lsym = .true.")
        lines.append(f"  no_overlap = .true.")
        lines.append("/")
        lines.append("")
        lines.append(f"{len(kpath)}")
        for i, (kp, label) in enumerate(zip(kpath, klabels)):
            lines.append(
                f"  {kp[0]:.8f}  {kp[1]:.8f}  {kp[2]:.8f}  {label}"
            )
        lines.append("")
        lines.append(f"{nkpoints}")
        lines.append("")

        content = "\n".join(lines)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content)
        return content

    def compile_dos_input(
        self,
        material: str,
        output_path: Path,
        *,
        prefix: Optional[str] = None,
        emin: Optional[float] = None,
        emax: Optional[float] = None,
        deltae: float = 0.01,
        fwhm: float = 0.05,
        ngauss: int = 1,
    ) -> str:
        """Compile a dos.x post-processing input file.

        Args:
            material: Material key in MATERIAL_DB
            output_path: Where to write the .in file
            prefix: QE prefix (must match SCF/NSCF prefix)
            emin: Minimum energy in eV (None = auto)
            emax: Maximum energy in eV (None = auto)
            deltae: Energy grid step in eV
            fwhm: Broadening width in eV
            ngauss: Broadening type (1=Gaussian, 0=Methfessel-Paxton)

        Returns:
            The generated input file content as string
        """
        if material not in MATERIAL_DB:
            raise ValueError(f"Unknown material: {material}")

        db = MATERIAL_DB[material]
        prefix = prefix or material.lower()

        lines = []
        lines.append("&dos")
        lines.append(f'  prefix = "{prefix}"')
        lines.append('  outdir = "./"')
        lines.append(f"  ngauss = {ngauss}")
        lines.append(f"  degauss = {fwhm}")
        lines.append(f"  DeltaE = {deltae}")
        if emin is not None:
            lines.append(f"  Emin = {emin}")
        if emax is not None:
            lines.append(f"  Emax = {emax}")
        lines.append("/")

        content = "\n".join(lines)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content)
        return content

    def compute_input_hash(self, content: str) -> str:
        """Compute SHA-256 hash of input file content for reproducibility."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]
