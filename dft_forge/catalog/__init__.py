"""Material catalog: deterministic material profiles and task definitions.

No LLM involved.  These are the "ground truth" specifications for the
deterministic compiler and verifier.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from dft_forge.protocol.schemas import MaterialProfile, TaskSpec

from dft_forge.catalog.library import LIBRARY_MATERIALS

# ── Built-in materials ───────────────────────────────────────────────────────

MATERIALS: Dict[str, Dict[str, Any]] = {
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
        "charge": 0,
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
        "charge": 0,
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
        "charge": 0,
        "is_metal": False,
    },
}


def get_material_profile(material: str) -> MaterialProfile:
    """Get a MaterialProfile for a known or library material."""
    db = None
    if material in MATERIALS:
        db = MATERIALS[material]
    elif material in LIBRARY_MATERIALS:
        db = LIBRARY_MATERIALS[material]
    if db is None:
        raise ValueError(
            f"Unknown material: {material}. Known: {sorted(set(MATERIALS) | set(LIBRARY_MATERIALS))[:20]} ..."
        )

    if "cellpar" in db:
        a_angstrom = db["cellpar"][0]
        natoms = len(db["sites"])
    else:
        a_angstrom = db["lattice_constant_angstrom"]
        natoms = db["natoms_primitive"]
    return MaterialProfile(
        formula=db["formula"],
        structure_type=db["structure_type"],
        space_group=db["space_group"],
        ibrav=db["ibrav"],
        lattice_constant_bohr=a_angstrom * 1.8897261246,
        natoms=natoms,
        nspecies=db["nspecies"],
        species=db["species"],
        masses=db["masses"],
        pseudos=db["pseudos"],
        occupations=db["occupations"],
        smearing=db["smearing"],
        degauss=db["degauss"],
        charge=db["charge"] if "charge" in db else 0,
        is_metal=db["is_metal"],
    )


# ── Task specs ───────────────────────────────────────────────────────────────

TASKS: Dict[str, Dict[str, Any]] = {
    "T1_Si_vcrelax": {
        "task_id": "T1_Si_vcrelax",
        "task_type": "T1",
        "material": "Si",
        "description": "Si diamond structure vc-relax optimization",
        "parameters": {
            "ecutwfc": 25.0,
            "ecutrho": 200.0,
            "kpoints": [4, 4, 4, 1, 1, 1],
            "conv_thr": 1.0e-8,
            "cell_optimization": True,
            "nstep": 200,
        },
    },
    "T1_Al_vcrelax": {
        "task_id": "T1_Al_vcrelax",
        "task_type": "T1",
        "material": "Al",
        "description": "Al FCC structure vc-relax optimization",
        "parameters": {
            "ecutwfc": 20.0,
            "ecutrho": 160.0,
            "kpoints": [8, 8, 8, 1, 1, 1],
            "conv_thr": 1.0e-8,
            "cell_optimization": True,
            "nstep": 200,
        },
    },
    "T1_MgO_vcrelax": {
        "task_id": "T1_MgO_vcrelax",
        "task_type": "T1",
        "material": "MgO",
        "description": "MgO NaCl structure vc-relax optimization",
        "parameters": {
            "ecutwfc": 40.0,
            "ecutrho": 320.0,
            "kpoints": [4, 4, 4, 1, 1, 1],
            "conv_thr": 1.0e-8,
            "cell_optimization": True,
            "nstep": 200,
        },
    },
    "T2_Si_bands": {
        "task_id": "T2_Si_bands",
        "task_type": "T2",
        "material": "Si",
        "description": "Si diamond structure band structure (SCF + NSCF + bands.x)",
        "parameters": {
            "subtype": "bands",
            "ecutwfc": 25.0,
            "ecutrho": 200.0,
            "kpoints_scf": [8, 8, 8, 1, 1, 1],
            "kpoints_nscf": [8, 8, 8, 1, 1, 1],
            "conv_thr": 1.0e-8,
            "nbnd": 16,
            "nkpoints_bands": 100,
        },
    },
    "T2_Al_bands": {
        "task_id": "T2_Al_bands",
        "task_type": "T2",
        "material": "Al",
        "description": "Al FCC structure band structure (SCF + NSCF + bands.x)",
        "parameters": {
            "subtype": "bands",
            "ecutwfc": 20.0,
            "ecutrho": 160.0,
            "kpoints_scf": [12, 12, 12, 1, 1, 1],
            "kpoints_nscf": [12, 12, 12, 1, 1, 1],
            "conv_thr": 1.0e-8,
            "nbnd": 12,
            "nkpoints_bands": 100,
        },
    },
    "T2_MgO_bands": {
        "task_id": "T2_MgO_bands",
        "task_type": "T2",
        "material": "MgO",
        "description": "MgO NaCl structure band structure (SCF + NSCF + bands.x)",
        "parameters": {
            "subtype": "bands",
            "ecutwfc": 40.0,
            "ecutrho": 320.0,
            "kpoints_scf": [8, 8, 8, 1, 1, 1],
            "kpoints_nscf": [8, 8, 8, 1, 1, 1],
            "conv_thr": 1.0e-8,
            "nbnd": 24,
            "nkpoints_bands": 100,
        },
    },
    "T2_Si_dos": {
        "task_id": "T2_Si_dos",
        "task_type": "T2",
        "material": "Si",
        "description": "Si diamond structure DOS (SCF + NSCF + dos.x)",
        "parameters": {
            "subtype": "dos",
            "ecutwfc": 25.0,
            "ecutrho": 200.0,
            "kpoints_scf": [8, 8, 8, 1, 1, 1],
            "kpoints_nscf": [8, 8, 8, 1, 1, 1],
            "conv_thr": 1.0e-8,
            "nbnd": 16,
            "dos_deltae": 0.01,
            "dos_fwhm": 0.05,
        },
    },
    "T2_Al_dos": {
        "task_id": "T2_Al_dos",
        "task_type": "T2",
        "material": "Al",
        "description": "Al FCC structure DOS (SCF + NSCF + dos.x)",
        "parameters": {
            "subtype": "dos",
            "ecutwfc": 20.0,
            "ecutrho": 160.0,
            "kpoints_scf": [12, 12, 12, 1, 1, 1],
            "kpoints_nscf": [12, 12, 12, 1, 1, 1],
            "conv_thr": 1.0e-8,
            "nbnd": 12,
            "dos_deltae": 0.01,
            "dos_fwhm": 0.05,
        },
    },
    "T2_MgO_dos": {
        "task_id": "T2_MgO_dos",
        "task_type": "T2",
        "material": "MgO",
        "description": "MgO NaCl structure DOS (SCF + NSCF + dos.x)",
        "parameters": {
            "subtype": "dos",
            "ecutwfc": 40.0,
            "ecutrho": 320.0,
            "kpoints_scf": [8, 8, 8, 1, 1, 1],
            "kpoints_nscf": [8, 8, 8, 1, 1, 1],
            "conv_thr": 1.0e-8,
            "nbnd": 24,
            "dos_deltae": 0.01,
            "dos_fwhm": 0.05,
        },
    },
}


def get_task_spec(task_id: str) -> TaskSpec:
    """Get a TaskSpec for a known task."""
    if task_id not in TASKS:
        raise ValueError(f"Unknown task: {task_id}. Known: {list(TASKS.keys())}")
    
    db = TASKS[task_id]
    material_profile = get_material_profile(db["material"])
    
    return TaskSpec(
        task_id=db["task_id"],
        task_type=db["task_type"],
        material=db["material"],
        description=db["description"],
        material_profile=material_profile,
        parameters=db["parameters"],
    )


def list_tasks(task_type: Optional[str] = None) -> list:
    """List all available task IDs, optionally filtered by task_type."""
    if task_type is None:
        return list(TASKS.keys())
    return [tid for tid, tspec in TASKS.items() if tspec["task_type"] == task_type]


# ── Dynamic profiles from user structures ─────────────────────────────────────

def register_dynamic_material(
    material_id: str,
    profile: MaterialProfile,
) -> None:
    """Register a user-provided material profile in the in-memory catalog."""
    MATERIALS[material_id] = {
        "formula": profile.formula,
        "structure_type": "user_imported",
        "space_group": profile.space_group or "P1",
        "ibrav": 0,
        "lattice_constant_angstrom": float(profile.lattice_constant_bohr / 1.8897261246),
        "natoms_primitive": profile.natoms,
        "nspecies": profile.nspecies,
        "species": list(profile.species),
        "masses": list(profile.masses),
        "pseudos": dict(profile.pseudos),
        "occupations": profile.occupations,
        "smearing": profile.smearing,
        "degauss": profile.degauss,
        "charge": profile.charge,
        "is_metal": profile.is_metal,
    }


def build_dynamic_material_profile(
    report: Any,
    *,
    pseudos: Optional[Dict[str, str]] = None,
    occupations: str = "smearing",
    smearing: str = "mp",
    degauss: float = 0.02,
    charge: int = 0,
    is_metal: bool = False,
) -> MaterialProfile:
    """Build a MaterialProfile from a successful StructureImporter ValidationReport.

    The caller should supply pseudopotential mapping because the importer
    does not know which pseudos the user has on disk.
    """
    if report.atoms is None or not report.success:
        raise ValueError("Cannot build profile from failed validation report")

    atoms = report.atoms
    species = report.species or sorted({sym for sym in atoms.get_chemical_symbols()})
    mass_map = {
        "H": 1.008, "He": 4.0026, "Li": 6.941, "Be": 9.0122, "B": 10.81, "C": 12.011,
        "N": 14.007, "O": 15.999, "F": 18.998, "Na": 22.99, "Mg": 24.305, "Al": 26.982,
        "Si": 28.086, "P": 30.974, "S": 32.065, "Cl": 35.453, "K": 39.098, "Ca": 40.078,
        "Ti": 47.867, "V": 50.942, "Cr": 51.996, "Mn": 54.938, "Fe": 55.845, "Co": 58.933,
        "Ni": 58.693, "Cu": 63.546, "Zn": 65.38, "Ga": 69.723, "Ge": 72.63, "As": 74.922,
        "Se": 78.971, "Br": 79.904, "Rb": 85.468, "Sr": 87.62, "Y": 88.906, "Zr": 91.224,
        "Nb": 92.906, "Mo": 95.95, "Tc": 98.0, "Ru": 101.07, "Rh": 102.906, "Pd": 106.42,
        "Ag": 107.868, "Cd": 112.414, "In": 114.818, "Sn": 118.71, "Sb": 121.76, "Te": 127.6,
        "I": 126.904, "Cs": 132.905, "Ba": 137.327, "La": 138.905, "Ce": 140.116, "Pr": 140.908,
        "Nd": 144.242, "Pm": 145.0, "Sm": 150.36, "Eu": 151.964, "Gd": 157.25, "Tb": 158.925,
        "Dy": 162.5, "Ho": 164.93, "Er": 167.259, "Tm": 168.934, "Yb": 173.054, "Lu": 174.967,
        "Hf": 178.49, "Ta": 180.948, "W": 183.84, "Re": 186.207, "Os": 190.23, "Ir": 192.217,
        "Pt": 195.084, "Au": 196.967, "Hg": 200.592, "Tl": 204.383, "Pb": 207.2, "Bi": 208.98,
    }
    masses = [float(mass_map.get(sym, 0.0)) for sym in species]
    if any(m == 0.0 for m in masses):
        unknown = [sym for sym, m in zip(species, masses) if m == 0.0]
        raise ValueError(f"Unknown element(s) without mass lookup: {unknown}")

    cell = atoms.cell
    a_angstrom = float(np.linalg.norm(cell[0]) / 1.8897261246)
    return MaterialProfile(
        formula=report.formula or atoms.get_chemical_formula(),
        structure_type="user_imported",
        space_group=report.space_group or "P1",
        ibrav=0,
        lattice_constant_bohr=float(a_angstrom * 1.8897261246),
        natoms=report.natoms,
        nspecies=report.nspecies,
        species=species,
        masses=masses,
        pseudos=pseudos or {},
        occupations=occupations,
        smearing=smearing,
        degauss=degauss,
        charge=charge,
        is_metal=is_metal,
    )


def build_dynamic_task_spec(
    task_id: str,
    profile: MaterialProfile,
    *,
    task_type: str = "T1",
    parameters: Optional[Dict[str, Any]] = None,
    description: str = "",
) -> TaskSpec:
    """Build a TaskSpec for a dynamically imported structure."""
    return TaskSpec(
        task_id=task_id,
        task_type=task_type,
        material=profile.formula,
        description=description or f"{profile.formula} user-imported structure",
        material_profile=profile,
        parameters=parameters or {},
        metadata={"dynamic": True},
    )


def build_library_task_spec(
    material: str,
    *,
    task_type: str = "T1",
    subtype: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a TASKS-style dict for a library material on demand.

    Parameters follow the built-in task conventions with heuristics from the
    material's library entry (ecut, kpoints, metal smearing).
    """
    if material not in LIBRARY_MATERIALS and material not in MATERIALS:
        raise ValueError(f"Unknown material: {material}")
    db = MATERIALS.get(material) or LIBRARY_MATERIALS[material]
    is_metal = db["is_metal"]
    ecutwfc = db.get("ecutwfc_default", 45.0)
    ecutrho = db.get("ecutrho_default", ecutwfc * 8.0)
    kpoints = db.get("kpoints_default", (6, 6, 6, 1, 1, 1))
    natoms = len(db["sites"]) if "sites" in db else db["natoms_primitive"]
    nbnd = natoms * 8 if is_metal else natoms * 4

    if task_type == "T1":
        return {
            "task_id": f"T1_{material}_vcrelax",
            "task_type": "T1",
            "material": material,
            "description": f"{material} ({db.get('structure_type', '')}) vc-relax optimization",
            "parameters": {
                "ecutwfc": ecutwfc,
                "ecutrho": ecutrho,
                "kpoints": list(kpoints),
                "conv_thr": 1.0e-8,
                "cell_optimization": True,
                "nstep": 200,
            },
        }

    if task_type == "T2":
        params = {
            "subtype": subtype or "bands",
            "ecutwfc": ecutwfc,
            "ecutrho": ecutrho,
            "kpoints_scf": list(kpoints),
            "kpoints_nscf": list(kpoints),
            "conv_thr": 1.0e-8,
            "nbnd": nbnd,
        }
        if (subtype or "bands") == "bands":
            params["nkpoints_bands"] = 100
        else:
            params["dos_deltae"] = 0.01
            params["dos_fwhm"] = 0.05
        return {
            "task_id": f"T2_{material}_{subtype or 'bands'}",
            "task_type": "T2",
            "material": material,
            "description": f"{material} {subtype or 'bands'} (SCF + NSCF)",
            "parameters": params,
        }

    raise ValueError(f"Unsupported task_type: {task_type}")


def list_materials() -> List[str]:
    """List all registered material IDs, including library and dynamic ones."""
    combined = set(MATERIALS.keys()) | set(LIBRARY_MATERIALS.keys())
    return sorted(combined)
