"""Atomic adsorption site scan runner."""

from __future__ import annotations

import math
import numbers
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from dft_forge.runtime.graph import GraphTemplate, NodeSpec

from .common import (
    WorkflowError,
    _base_params,
    _finite_real,
    _jsonable,
    _matching_atom_count,
    _require_method_compatibility,
    _runtime_controls,
    _workflow_id,
    _workflow_workdir,
)
from .energetics import adsorption_energy


def _run_graph(*args, **kwargs):
    """Resolve via the package namespace so tests can monkeypatch it."""
    import dft_forge.workflows as _pkg
    return _pkg._run_graph(*args, **kwargs)


def _write_structure(*args, **kwargs):
    """Resolve via the package namespace so tests can monkeypatch it."""
    import dft_forge.workflows as _pkg
    return _pkg._write_structure(*args, **kwargs)


_ATOMIC_ADSORBATE_SPINS = {
    "H": 1.0, "C": 2.0, "N": 3.0, "O": 2.0, "F": 1.0, "Cl": 1.0,
}
_ATOMIC_REFERENCE_MOLECULES = {
    "H": "h2", "N": "n2", "O": "o2", "F": "f2", "Cl": "cl2",
}


def _normalise_adsorption_sites(args: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_sites = args.get("sites")
    if raw_sites in (None, ""):
        raw_sites = [{"site": args.get("site", "top"), "site_index": args.get("site_index", 0)}]
    if not isinstance(raw_sites, (list, tuple)) or not raw_sites:
        raise WorkflowError("sites must be a non-empty array")
    max_sites = args.get("max_sites", 8)
    if isinstance(max_sites, bool) or not isinstance(max_sites, numbers.Integral) or not 1 <= int(max_sites) <= 32:
        raise WorkflowError("max_sites must be an integer between 1 and 32")
    if len(raw_sites) > int(max_sites):
        raise WorkflowError(f"requested {len(raw_sites)} sites exceeds max_sites={int(max_sites)}")
    normalised: List[Dict[str, Any]] = []
    identities = set()
    for item in raw_sites:
        if isinstance(item, str):
            item = {"site": item}
        if not isinstance(item, dict):
            raise WorkflowError("each adsorption site must be a string or object")
        site = str(item.get("site", "top")).strip().lower()
        if site not in {"top", "bridge", "hollow"}:
            raise WorkflowError("adsorption site must be top, bridge, or hollow")
        index = item.get("site_index", 0)
        if isinstance(index, bool) or not isinstance(index, numbers.Integral) or int(index) < 0:
            raise WorkflowError("site_index must be a non-negative integer")
        height = item.get("height", args.get("height", 1.5))
        if isinstance(height, bool) or not isinstance(height, (int, float)) or not math.isfinite(height) or height <= 0:
            raise WorkflowError("adsorption height must be a finite positive number")
        identity = (site, int(index))
        if identity in identities:
            raise WorkflowError(f"duplicate adsorption site {site} with site_index={int(index)}")
        identities.add(identity)
        normalised.append({
            "site": site, "site_index": int(index), "height": float(height),
            "requested_site_id": item.get("site_id"),
        })
    return normalised


def _spin_params(value: Any, *, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    config = dict(default or {})
    if value is not None:
        if not isinstance(value, dict):
            raise WorkflowError("spin settings must be an object")
        config.update({key: val for key, val in value.items() if val is not None})
    return {key: config[key] for key in (
        "spin_polarized", "magnetic_order", "magnetic_element", "initial_moment_muB",
        "total_magnetization_muB", "occupations", "smearing", "degauss",
    ) if key in config}


def _adsorption_geometry_validation(
    *,
    initial_atoms: Any,
    surface_atoms: Any,
    adsorbate_element: str,
    final_structure_cif: Optional[str],
    final_structure_extxyz: Optional[str],
    require_final_structure: bool,
    max_distance_angstrom: float,
) -> Dict[str, Any]:
    """Verify that the relaxed adsorbate remains associated with the slab.

    QE preserves atom order through a relaxation, so the surface atoms occupy
    the same leading indices and the one appended adsorbate is last. Missing
    or malformed final geometry fails closed instead of allowing an apparently
    converged but desorbed configuration to rank.
    """
    result: Dict[str, Any] = {
        "verified": False,
        "adsorbate_element": adsorbate_element,
        "adsorbate_atom_index": len(surface_atoms),
        "nearest_surface_atom_index": None,
        "nearest_surface_element": None,
        "nearest_distance_angstrom": None,
        "maximum_distance_angstrom": float(max_distance_angstrom),
        "structure_source": "final_relaxed_structure" if require_final_structure else "initial_structure",
    }
    try:
        if require_final_structure:
            structure_file = final_structure_extxyz or final_structure_cif
            path = Path(str(structure_file or ""))
            if not structure_file or not path.is_file():
                raise WorkflowError("relaxed adsorption structure is missing")
            from ase.io import read as ase_read

            atoms = ase_read(str(path), index=-1)
            result["structure_source"] = (
                "final_relaxed_extxyz" if final_structure_extxyz else "final_relaxed_cif"
            )
        else:
            atoms = initial_atoms.copy()

        surface_symbols = list(surface_atoms.get_chemical_symbols())
        symbols = list(atoms.get_chemical_symbols())
        if len(symbols) != len(surface_symbols) + 1:
            raise WorkflowError(
                "adsorption geometry has an unexpected atom count "
                f"({len(symbols)} != {len(surface_symbols) + 1})"
            )
        if symbols[:-1] != surface_symbols or symbols[-1] != adsorbate_element:
            raise WorkflowError("adsorption geometry no longer matches the surface-plus-adsorbate atom order")

        if require_final_structure:
            for row, expected_row in zip(atoms.cell, surface_atoms.cell):
                if any(
                    not math.isclose(float(actual), float(expected), rel_tol=1e-6, abs_tol=1e-4)
                    for actual, expected in zip(row, expected_row)
                ):
                    raise WorkflowError("relaxed adsorption cell does not match the fixed input slab cell")
            if final_structure_extxyz and list(atoms.pbc) != list(surface_atoms.pbc):
                raise WorkflowError("relaxed adsorption extxyz PBC does not match the input slab")
        # CIF does not reliably preserve non-periodic axes. After checking
        # cell equality, restore the fixed-cell input's authoritative PBC.
        atoms.cell = surface_atoms.cell
        atoms.pbc = surface_atoms.pbc
        adsorbate_index = len(surface_symbols)
        distances = atoms.get_distances(
            adsorbate_index, list(range(len(surface_symbols))), mic=True,
        )
        finite = [
            (index, float(distance))
            for index, distance in enumerate(distances)
            if _finite_real(distance) and float(distance) > 0
        ]
        if not finite:
            raise WorkflowError("adsorption geometry has no finite adsorbate-surface distance")
        nearest_index, nearest_distance = min(finite, key=lambda item: item[1])
        result.update({
            "nearest_surface_atom_index": nearest_index,
            "nearest_surface_element": surface_symbols[nearest_index],
            "nearest_distance_angstrom": nearest_distance,
            "verified": nearest_distance <= float(max_distance_angstrom),
        })
        if not result["verified"]:
            result["reason"] = (
                f"nearest adsorbate-surface distance {nearest_distance:.6g} angstrom exceeds "
                f"the {float(max_distance_angstrom):.6g} angstrom limit"
            )
    except Exception as exc:
        result["reason"] = str(exc)
    return result


def run_adsorption_scan(args: Dict[str, Any]) -> Dict[str, Any]:
    """Relax and rank atomic adsorption sites with shared surface/gas references."""
    kind = str(args.get("kind") or "").strip().lower()
    element = str(args.get("adsorbate_element") or args.get("element") or "").strip()
    element = element[:1].upper() + element[1:].lower() if element else ""
    if not kind or not element or not re.fullmatch(r"[A-Z][a-z]?", element):
        raise WorkflowError("adsorption_scan requires kind and one atomic adsorbate_element")
    sites = _normalise_adsorption_sites(args)
    supercell = str(args.get("supercell", "3x3"))
    vacuum = float(args.get("vacuum", 15.0))
    if not math.isfinite(vacuum) or vacuum <= 0:
        raise WorkflowError("vacuum must be a finite positive number")
    reference_kind = str(
        args.get("reference_molecule") or _ATOMIC_REFERENCE_MOLECULES.get(element) or ""
    ).lower()
    if not reference_kind:
        raise WorkflowError(
            f"no molecular reference is defined for {element}; provide reference_molecule explicitly"
        )
    reference_magnetization_tolerance = args.get(
        "reference_magnetization_tolerance_muB", 0.35
    )
    adsorbed_magnetization_tolerance = args.get(
        "adsorbed_magnetization_tolerance_muB", 0.35
    )
    max_adsorbate_surface_distance = args.get(
        "max_adsorbate_surface_distance_angstrom", 3.0
    )
    for label, value in (
        ("reference_magnetization_tolerance_muB", reference_magnetization_tolerance),
        ("adsorbed_magnetization_tolerance_muB", adsorbed_magnetization_tolerance),
    ):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0):
            raise WorkflowError(f"{label} must be a finite non-negative number")
    reference_magnetization_tolerance = float(reference_magnetization_tolerance)
    adsorbed_magnetization_tolerance = float(adsorbed_magnetization_tolerance)
    if (not _finite_real(max_adsorbate_surface_distance)
            or float(max_adsorbate_surface_distance) <= 0):
        raise WorkflowError(
            "max_adsorbate_surface_distance_angstrom must be a finite positive number"
        )
    max_adsorbate_surface_distance = float(max_adsorbate_surface_distance)
    from dft_forge.structure_builder import build_2d, build_molecule

    clean = build_2d(kind, supercell=supercell, vacuum=vacuum)
    molecule = build_molecule(reference_kind, box=float(args.get("molecule_box", 10.0)))
    molecule_symbols = molecule.atoms.get_chemical_symbols()
    reference_atoms = molecule_symbols.count(element)
    if reference_atoms < 1 or set(molecule_symbols) != {element}:
        raise WorkflowError(
            "atomic adsorption energy requires a homonuclear molecular reference containing the adsorbate element"
        )
    workdir = _workflow_workdir(args, "adsorption_scan")
    clean_path = _write_structure(clean, workdir / "clean_surface.cif")
    molecule_path = _write_structure(molecule, workdir / f"reference_{reference_kind}.cif")
    structures: List[Tuple[Dict[str, Any], Any, str]] = []
    for ordinal, request in enumerate(sites, start=1):
        built = build_2d(
            kind, supercell=supercell, vacuum=vacuum,
            adsorbate={
                "element": element, "site": request["site"],
                "site_index": request["site_index"], "height": request["height"],
            },
        )
        metadata = dict(built.selected_site or {})
        stable_id = str(request.get("requested_site_id") or metadata.get("site_id") or f"{request['site']}_{request['site_index']}")
        metadata.update({
            "site_id": stable_id, "initial_site_id": stable_id,
            "site": request["site"], "site_index": request["site_index"],
            "height": request["height"],
        })
        path = _write_structure(built, workdir / f"site_{ordinal}_{request['site']}_{request['site_index']}.cif")
        structures.append((metadata, built, path))

    base = _base_params(args)
    base.pop("structure", None)
    base.pop("material", None)
    base.setdefault("conv_thr", 1.0e-8 if args.get("conv_thr") in (None, "") else args["conv_thr"])
    do_relax = bool(args.get("relax", True))
    relax_extra = {
        "force_threshold": float(args.get("force_threshold", 1.0e-3)),
        "nstep": int(args.get("nstep", 200)),
    }
    surface_spin = _spin_params(args.get("surface_spin"), default={"spin_polarized": False})
    ads_default: Dict[str, Any] = {"spin_polarized": False}
    if element in _ATOMIC_ADSORBATE_SPINS:
        ads_default = {
            "spin_polarized": True, "magnetic_order": "FM", "magnetic_element": element,
            "initial_moment_muB": _ATOMIC_ADSORBATE_SPINS[element],
        }
    ads_spin = _spin_params(args.get("adsorbed_spin"), default=ads_default)
    molecule_default: Dict[str, Any] = {"spin_polarized": False, "occupations": "fixed"}
    expected_reference_magnetization = None
    if reference_kind == "o2":
        molecule_default = {
            "spin_polarized": True, "magnetic_order": "FM", "magnetic_element": "O",
            "initial_moment_muB": 1.0, "total_magnetization_muB": 2.0,
            "occupations": "fixed",
        }
        expected_reference_magnetization = 2.0
    molecule_spin = _spin_params(args.get("reference_spin"), default=molecule_default)
    molecule_base = {**base, "kpoints": [1, 1, 1, 0, 0, 0], **molecule_spin}

    nodes: List[NodeSpec] = []

    def add_pair(prefix: str, structure_path: str, params: Dict[str, Any]) -> None:
        if do_relax:
            nodes.append(NodeSpec(
                id=f"{prefix}_relax", tool="qe",
                params={"calc": "relax", "prefix": prefix, "structure": structure_path, **base, **params, **relax_extra},
                repair="", max_repair_attempts=0,
            ))
            nodes.append(NodeSpec(
                id=f"{prefix}_scf", tool="qe",
                params={
                    "calc": "scf", "prefix": f"{prefix}_scf",
                    "structure": f"${{nodes.{prefix}_relax.outputs.final_structure_cif}}",
                    **base, **params,
                },
                depends_on=[f"{prefix}_relax"],
                repair="", max_repair_attempts=0,
            ))
        else:
            nodes.append(NodeSpec(
                id=f"{prefix}_scf", tool="qe",
                params={"calc": "scf", "prefix": f"{prefix}_scf", "structure": structure_path, **base, **params},
                repair="", max_repair_attempts=0,
            ))

    add_pair("clean", clean_path, surface_spin)
    # The gas reference intentionally uses Gamma-only sampling.
    if do_relax:
        nodes.append(NodeSpec(
            id="reference_relax", tool="qe",
            params={"calc": "relax", "prefix": "reference", "structure": molecule_path, **molecule_base, **relax_extra},
            repair="", max_repair_attempts=0,
        ))
        nodes.append(NodeSpec(
            id="reference_scf", tool="qe",
            params={
                "calc": "scf", "prefix": "reference_scf",
                "structure": "${nodes.reference_relax.outputs.final_structure_cif}", **molecule_base,
            },
            depends_on=["reference_relax"],
            repair="", max_repair_attempts=0,
        ))
    else:
        nodes.append(NodeSpec(
            id="reference_scf", tool="qe",
            params={"calc": "scf", "prefix": "reference_scf", "structure": molecule_path, **molecule_base},
            repair="", max_repair_attempts=0,
        ))
    for ordinal, (_metadata, _built, path) in enumerate(structures, start=1):
        add_pair(f"site_{ordinal}", path, ads_spin)

    template = GraphTemplate(
        template_id=_workflow_id("adsorption_scan", {
            "kind": kind, "element": element, "supercell": supercell, "sites": sites,
            "reference": reference_kind, "relax": do_relax,
        }),
        description="shared-reference multi-site atomic adsorption workflow",
        nodes=nodes,
    )
    max_jobs = args.get("max_jobs", 24)
    if isinstance(max_jobs, bool) or not isinstance(max_jobs, numbers.Integral) or int(max_jobs) < 1:
        raise WorkflowError("max_jobs must be a positive integer")
    if len(nodes) > int(max_jobs):
        raise WorkflowError(f"adsorption DAG has {len(nodes)} jobs, above max_jobs={int(max_jobs)}")
    payload = _run_graph(template, workdir / "graphs", **_runtime_controls(args))
    payload.update({
        "workflow_type": "adsorption_scan", "kind": kind, "adsorbate_element": element,
        "supercell": supercell, "reference_molecule": reference_kind,
        "reference_atoms": reference_atoms, "requested_sites": [meta for meta, _built, _path in structures],
        "spin_policy": {
            "surface": surface_spin, "adsorbed": ads_spin, "reference": molecule_spin,
        },
        "geometry_policy": {
            "maximum_adsorbate_surface_distance_angstrom": max_adsorbate_surface_distance,
            "failure_behavior": "exclude_from_ranking",
        },
    })
    clean_node = payload["nodes"].get("clean_scf", {})
    reference_node = payload["nodes"].get("reference_scf", {})
    clean_out = clean_node.get("outputs") or {}
    reference_out = reference_node.get("outputs") or {}
    blocking: List[str] = []
    if (clean_node.get("state") != "succeeded" or reference_node.get("state") != "succeeded"
            or not _finite_real(clean_out.get("energy_ry"))
            or not _finite_real(reference_out.get("energy_ry"))
            or not _matching_atom_count(clean_out.get("natoms"), len(clean.atoms))
            or not _matching_atom_count(reference_out.get("natoms"), len(molecule.atoms))):
        blocking.append("clean surface or molecular reference lacks successful finite energy/atom-count evidence")
    reference_mag = reference_out.get("total_magnetization_muB")
    if expected_reference_magnetization is not None:
        if (not isinstance(reference_mag, (int, float)) or not math.isfinite(reference_mag)
                or abs(float(reference_mag) - expected_reference_magnetization) > reference_magnetization_tolerance):
            blocking.append(
                "O2 reference did not verify the constrained triplet magnetization near 2 muB"
            )
    payload["reference_validation"] = {
        "expected_total_magnetization_muB": expected_reference_magnetization,
        "observed_total_magnetization_muB": reference_mag,
        "tolerance_muB": reference_magnetization_tolerance if expected_reference_magnetization is not None else None,
        "verified": not any("O2 reference" in reason for reason in blocking),
    }
    site_results: List[Dict[str, Any]] = []
    for ordinal, (metadata, built, _path) in enumerate(structures, start=1):
        scf_node = payload["nodes"].get(f"site_{ordinal}_scf", {})
        relax_node = payload["nodes"].get(f"site_{ordinal}_relax", {}) if do_relax else {}
        out = scf_node.get("outputs") or {}
        row: Dict[str, Any] = {
            **metadata, "node_id": f"site_{ordinal}_scf", "state": scf_node.get("state"),
            "error": scf_node.get("error"),
            "final_structure_cif": (relax_node.get("outputs") or {}).get("final_structure_cif"),
            "final_structure_extxyz": (relax_node.get("outputs") or {}).get("final_structure_extxyz"),
            "max_force_ry_bohr": (relax_node.get("outputs") or {}).get("max_force_ry_bohr"),
            "total_magnetization_muB": out.get("total_magnetization_muB"),
            "absolute_magnetization_muB": out.get("absolute_magnetization_muB"),
            "atom_magnetizations_muB": out.get("atom_magnetizations_muB"),
        }
        row_errors = list(blocking)
        if (scf_node.get("state") != "succeeded" or not _finite_real(out.get("energy_ry"))
                or not _matching_atom_count(out.get("natoms"), len(built.atoms))):
            row_errors.append("site SCF lacks successful finite energy/atom-count evidence")
        geometry_validation = _adsorption_geometry_validation(
            initial_atoms=built.atoms,
            surface_atoms=clean.atoms,
            adsorbate_element=element,
            final_structure_cif=row["final_structure_cif"],
            final_structure_extxyz=row["final_structure_extxyz"],
            require_final_structure=do_relax,
            max_distance_angstrom=max_adsorbate_surface_distance,
        )
        row["geometry_validation"] = geometry_validation
        if not geometry_validation["verified"]:
            row_errors.append(
                "site final adsorption geometry failed validation: "
                + str(geometry_validation.get("reason") or "unknown geometry error")
            )
        expected_site_magnetization = ads_spin.get("total_magnetization_muB")
        observed_site_magnetization = out.get("total_magnetization_muB")
        site_spin_verified = None
        if expected_site_magnetization is not None:
            site_spin_verified = (
                isinstance(observed_site_magnetization, (int, float))
                and not isinstance(observed_site_magnetization, bool)
                and math.isfinite(float(observed_site_magnetization))
                and abs(float(observed_site_magnetization) - float(expected_site_magnetization))
                <= adsorbed_magnetization_tolerance
            )
            if not site_spin_verified:
                row_errors.append("site did not verify the requested total magnetization")
        row["spin_validation"] = {
            "expected_total_magnetization_muB": expected_site_magnetization,
            "observed_total_magnetization_muB": observed_site_magnetization,
            "tolerance_muB": (
                adsorbed_magnetization_tolerance
                if expected_site_magnetization is not None else None
            ),
            "verified": site_spin_verified,
        }
        if not row_errors:
            try:
                _require_method_compatibility(
                    clean_out, out, compare_kpoints=True,
                    common_elements=set(clean.atoms.get_chemical_symbols()),
                )
                _require_method_compatibility(
                    reference_out, out, compare_kpoints=False, common_elements={element},
                )
            except WorkflowError as exc:
                row_errors.append(str(exc))
        if row_errors:
            row["blocking_reasons"] = list(dict.fromkeys(row_errors))
        else:
            row.update(adsorption_energy(
                adsorbed_energy_ry=float(out["energy_ry"]),
                surface_energy_ry=float(clean_out["energy_ry"]),
                molecule_energy_ry=float(reference_out["energy_ry"]),
                adsorbate_atoms=1, reference_atoms=reference_atoms,
            ))
            row["adsorbed_energy_ry"] = float(out["energy_ry"])
        site_results.append(row)
    payload["site_results"] = site_results
    successful = [row for row in site_results if isinstance(row.get("e_ads_ev"), (int, float))]
    if successful:
        best = min(successful, key=lambda row: row["e_ads_ev"])
        payload["best_site_among_successful"] = {
            "site_id": best["site_id"], "e_ads_ev": best["e_ads_ev"],
        }
        for row in successful:
            row["relative_to_lowest_successful_ev"] = row["e_ads_ev"] - best["e_ads_ev"]
    if len(successful) == len(site_results):
        payload["status"] = "complete"
        payload["best_site"] = payload.get("best_site_among_successful")
    else:
        payload["status"] = "partial"
        payload["blocking_reasons"] = list(dict.fromkeys(
            blocking + [reason for row in site_results for reason in row.get("blocking_reasons", [])]
        ))
    payload["scope"] = (
        "ranking is limited to the requested initial sites and tested spin initializations; "
        "it is not a global adsorption-structure search"
    )
    return payload
