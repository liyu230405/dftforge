"""Neutral and charged vacancy formation runners."""

from __future__ import annotations

import math
import numbers
from typing import Any, Dict, List

from dft_forge.parser import RY_TO_EV
from dft_forge.runtime.graph import GraphTemplate, NodeSpec

from .common import (
    WorkflowError,
    _base_params,
    _charge_node_id,
    _charge_value,
    _finite_real,
    _jsonable,
    _matching_atom_count,
    _require_method_compatibility,
    _runtime_controls,
    _workflow_id,
    _workflow_workdir,
)
from .energetics import (
    charged_vacancy_formation_energy,
    makov_payne_correction_ev,
    vacancy_formation_energy,
)


def _run_graph(*args, **kwargs):
    """Resolve via the package namespace so tests can monkeypatch it."""
    import dft_forge.workflows as _pkg

    return _pkg._run_graph(*args, **kwargs)


def _write_structure(*args, **kwargs):
    """Resolve via the package namespace so tests can monkeypatch it."""
    import dft_forge.workflows as _pkg

    return _pkg._write_structure(*args, **kwargs)


def run_neutral_vacancy(args: Dict[str, Any]) -> Dict[str, Any]:
    if int(args.get("charge", 0) or 0) != 0:
        raise WorkflowError("neutral_vacancy only accepts charge=0; charged defects need a charged-DFT/correction workflow")
    source = str(args.get("source") or "")
    element = str(args.get("element") or "")
    if not source or not element:
        raise WorkflowError("neutral_vacancy requires source and element")
    reference_mode = args.get("reference_mode", "explicit")
    if reference_mode not in {"explicit", "same_method_bulk_per_atom"}:
        raise WorkflowError("unknown vacancy reference_mode")
    mu_fields = [key for key in ("chemical_potential_ry", "chemical_potential_ev") if args.get(key) is not None]
    if len(mu_fields) > 1 or (reference_mode == "same_method_bulk_per_atom" and mu_fields):
        raise WorkflowError("choose either one explicit chemical potential or same_method_bulk_per_atom, not both")
    workdir = _workflow_workdir(args, "neutral_vacancy")
    from dft_forge.structure_builder import supercell_3d, vacancy_3d

    pristine = supercell_3d(source, supercell=str(args.get("supercell", "2x2x2")))
    defect = vacancy_3d(source, element, supercell=str(args.get("supercell", "2x2x2")), index=args.get("index"))
    if reference_mode == "same_method_bulk_per_atom":
        if (set(pristine.atoms.get_chemical_symbols()) != {element}
                or set(defect.atoms.get_chemical_symbols()) != {element}
                or len(defect.atoms) != len(pristine.atoms) - 1):
            raise WorkflowError("same_method_bulk_per_atom requires an elemental bulk and exactly one removed host atom")
    pristine_path = _write_structure(pristine, workdir / "pristine.cif")
    defect_path = _write_structure(defect, workdir / "vacancy.cif")
    base = _base_params(args)
    base.pop("structure", None)
    base["material"] = source
    base.setdefault("conv_thr", 1e-8 if args.get("conv_thr") in (None, "") else args["conv_thr"])
    nodes = [
        NodeSpec(
            id="pristine_scf", tool="qe",
            params={"calc": "scf", "prefix": "pristine", "structure": pristine_path, **base},
        ),
    ]
    if bool(args.get("relax", True)):
        nodes.append(NodeSpec(
            id="defect_relax", tool="qe",
            params={"calc": "relax", "prefix": "defect", "structure": defect_path, **base},
        ))
        defect_structure = "${nodes.defect_relax.outputs.final_structure_cif}"
        defect_dep = ["defect_relax"]
    else:
        defect_structure = defect_path
        defect_dep = []
    nodes.append(NodeSpec(
        id="defect_scf", tool="qe",
        params={"calc": "scf", "prefix": "defect_scf", "structure": defect_structure, **base},
        depends_on=defect_dep,
    ))
    template = GraphTemplate(
        template_id=_workflow_id("neutral_vacancy", {"source": source, "element": element, "supercell": args.get("supercell"), "relax": bool(args.get("relax", True)), "reference_mode": reference_mode}),
        description="neutral bulk vacancy formation workflow",
        nodes=nodes,
    )
    payload = _run_graph(template, workdir / "graphs", **_runtime_controls(args))
    p_out = payload["nodes"]["pristine_scf"].get("outputs") or {}
    d_out = payload["nodes"]["defect_scf"].get("outputs") or {}
    payload["workflow_type"] = "neutral_vacancy"
    payload["structure"] = {
        "pristine": _jsonable(pristine.to_dict()),
        "defect": _jsonable(defect.to_dict()),
    }
    mu_ry = args.get("chemical_potential_ry")
    if mu_ry is None and args.get("chemical_potential_ev") is not None:
        mu_ry = float(args["chemical_potential_ev"]) / RY_TO_EV
    if reference_mode == "same_method_bulk_per_atom":
        energy = p_out.get("energy_ry")
        natoms = len(pristine.atoms)
        if (payload["nodes"]["pristine_scf"].get("state") != "succeeded"
                or not isinstance(energy, (int, float)) or isinstance(energy, bool)
                or not math.isfinite(energy)
                or p_out.get("natoms", natoms) != natoms):
            payload["status"] = "partial"
            payload["blocking_reasons"] = ["cannot derive chemical potential: pristine energy/atom-count evidence is invalid"]
            return payload
        mu_ry = energy / natoms
        payload["chemical_potential_reference"] = {
            "mode": reference_mode, "source_node": "pristine_scf", "element": element,
            "bulk_energy_ry": energy, "natoms": natoms, "chemical_potential_ry": mu_ry,
            "run_id": payload.get("run_id"), "execution": (payload.get("metadata") or {}).get("execution"),
        }
    if mu_ry is None:
        payload["status"] = "partial"
        payload["blocking_reasons"] = ["chemical_potential_ry/ev is required; refuse to guess the removed-atom reference"]
    elif (all(payload["nodes"][node].get("state") == "succeeded" for node in ("pristine_scf", "defect_scf"))
          and all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
                  for value in (p_out.get("energy_ry"), d_out.get("energy_ry"), mu_ry))):
        payload["formation"] = vacancy_formation_energy(
            defect_energy_ry=d_out["energy_ry"],
            pristine_energy_ry=p_out["energy_ry"],
            chemical_potential_ry=float(mu_ry),
        )
        payload["formation"]["reference_mode"] = reference_mode
    return payload


def run_charged_vacancy(args: Dict[str, Any]) -> Dict[str, Any]:
    """Matched charged-vacancy calculations plus explicit finite-size terms.

    The workflow deliberately does not pretend to infer dielectric screening
    or far-field potential alignment.  Those observations must be supplied;
    the image-charge term may either be explicit or computed with the cubic
    first-order Makov-Payne model.
    """
    source = str(args.get("source") or "")
    element = str(args.get("element") or "")
    if not source or not element:
        raise WorkflowError("charged_vacancy requires source and element")
    raw_charges = args.get("charges")
    if not isinstance(raw_charges, (list, tuple)) or not raw_charges:
        raise WorkflowError("charges must be a non-empty integer array")
    if any(isinstance(value, bool) or not isinstance(value, numbers.Integral) for value in raw_charges):
        raise WorkflowError("charges must contain only integers")
    charges = [int(value) for value in raw_charges]
    if len(set(charges)) != len(charges) or any(abs(value) > 4 for value in charges):
        raise WorkflowError("charges must be unique and within -4..4")
    if not any(charge != 0 for charge in charges):
        raise WorkflowError("charged_vacancy requires at least one non-zero charge state")
    fermi_levels = args.get("fermi_levels_ev", [0.0])
    if (not isinstance(fermi_levels, (list, tuple)) or not fermi_levels
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) or value < 0 for value in fermi_levels)):
        raise WorkflowError("fermi_levels_ev must be a non-empty array of finite values >= 0")
    fermi_levels = [float(value) for value in fermi_levels]
    band_gap = args.get("band_gap_ev")
    if band_gap is not None:
        band_gap = float(band_gap)
        if not math.isfinite(band_gap) or band_gap <= 0 or any(value > band_gap + 1e-9 for value in fermi_levels):
            raise WorkflowError("band_gap_ev must be positive and contain every requested Fermi level")
    magnetization_tolerance = args.get("magnetization_tolerance_muB", 0.35)
    if (isinstance(magnetization_tolerance, bool)
            or not isinstance(magnetization_tolerance, (int, float))
            or not math.isfinite(magnetization_tolerance)
            or magnetization_tolerance < 0):
        raise WorkflowError("magnetization_tolerance_muB must be a finite non-negative number")
    magnetization_tolerance = float(magnetization_tolerance)

    reference_mode = str(args.get("reference_mode", "explicit"))
    if reference_mode not in {"explicit", "same_method_bulk_per_atom"}:
        raise WorkflowError("unknown charged-vacancy reference_mode")
    mu_fields = [key for key in ("chemical_potential_ry", "chemical_potential_ev") if args.get(key) is not None]
    if len(mu_fields) > 1 or (reference_mode == "same_method_bulk_per_atom" and mu_fields):
        raise WorkflowError("choose one explicit chemical potential or same_method_bulk_per_atom")
    correction_mode = str(args.get("correction_mode", "makov_payne_cubic"))
    if correction_mode not in {"makov_payne_cubic", "explicit"}:
        raise WorkflowError("correction_mode must be makov_payne_cubic or explicit")
    alignments = args.get("potential_alignment_ev_by_charge")
    corrections = args.get("electrostatic_correction_ev_by_charge")
    spin_by_charge = args.get("spin_by_charge") or {}
    normalized_spin: Dict[int, Dict[str, Any]] = {}
    for charge in charges:
        if charge == 0:
            config = _charge_value(spin_by_charge, charge, "spin_by_charge", default={})
        else:
            config = _charge_value(spin_by_charge, charge, "spin_by_charge")
        if not isinstance(config, dict) or "spin_polarized" not in config:
            if charge != 0:
                raise WorkflowError(
                    f"spin_by_charge[{charge:+d}] must explicitly select spin_polarized for a charged state"
                )
            config = {"spin_polarized": False, **(config if isinstance(config, dict) else {})}
        normalized_spin[charge] = {
            key: config[key] for key in (
                "spin_polarized", "magnetic_order", "magnetic_element", "initial_moment_muB",
                "total_magnetization_muB", "occupations", "smearing", "degauss",
            ) if config.get(key) is not None
        }
        spin_choice = normalized_spin[charge].get("spin_polarized")
        if not isinstance(spin_choice, bool):
            raise WorkflowError(f"spin_by_charge[{charge:+d}].spin_polarized must be boolean")
        order = str(normalized_spin[charge].get("magnetic_order") or "").lower()
        if not spin_choice and order in {"fm", "afm"}:
            raise WorkflowError(f"spin_by_charge[{charge:+d}] cannot request {order.upper()} with spin_polarized=false")
        if spin_choice and not (
            normalized_spin[charge].get("total_magnetization_muB") is not None
            or (order in {"fm", "afm"} and normalized_spin[charge].get("initial_moment_muB") is not None)
        ):
            raise WorkflowError(
                f"spin_by_charge[{charge:+d}] must provide total_magnetization_muB or an explicit magnetic_order/initial_moment_muB seed"
            )
        for moment_key in ("initial_moment_muB", "total_magnetization_muB"):
            moment = normalized_spin[charge].get(moment_key)
            if moment is not None and (
                isinstance(moment, bool) or not isinstance(moment, (int, float)) or not math.isfinite(moment)
            ):
                raise WorkflowError(f"spin_by_charge[{charge:+d}].{moment_key} must be finite")
        if charge != 0:
            alignment = _charge_value(alignments, charge, "potential_alignment_ev_by_charge")
            if isinstance(alignment, bool) or not isinstance(alignment, (int, float)) or not math.isfinite(alignment):
                raise WorkflowError("potential alignments must be finite numbers")
            if correction_mode == "explicit":
                correction = _charge_value(corrections, charge, "electrostatic_correction_ev_by_charge")
                if isinstance(correction, bool) or not isinstance(correction, (int, float)) or not math.isfinite(correction):
                    raise WorkflowError("explicit electrostatic corrections must be finite numbers")

    workdir = _workflow_workdir(args, "charged_vacancy")
    from dft_forge.structure_builder import supercell_3d, vacancy_3d

    pristine = supercell_3d(source, supercell=str(args.get("supercell", "2x2x2")))
    defect = vacancy_3d(source, element, supercell=str(args.get("supercell", "2x2x2")), index=args.get("index"))
    if len(defect.atoms) != len(pristine.atoms) - 1:
        raise WorkflowError("charged_vacancy requires exactly one removed atom")
    if reference_mode == "same_method_bulk_per_atom" and (
        set(pristine.atoms.get_chemical_symbols()) != {element}
        or set(defect.atoms.get_chemical_symbols()) != {element}
    ):
        raise WorkflowError("same_method_bulk_per_atom requires an elemental host")
    correction_details: Dict[int, Dict[str, Any]] = {}
    for charge in charges:
        if charge == 0:
            correction_details[charge] = {"mode": correction_mode, "electrostatic_correction_ev": 0.0}
        elif correction_mode == "makov_payne_cubic":
            correction_details[charge] = {
                "mode": correction_mode,
                **makov_payne_correction_ev(
                    defect.atoms.cell, charge=charge,
                    dielectric_constant=args.get("dielectric_constant"),
                    madelung_alpha=float(args.get("madelung_alpha", 2.837297)),
                    cubic_tolerance=float(args.get("cubic_tolerance", 0.02)),
                ),
            }
        else:
            correction_details[charge] = {
                "mode": correction_mode,
                "electrostatic_correction_ev": float(_charge_value(
                    corrections, charge, "electrostatic_correction_ev_by_charge"
                )),
            }

    pristine_path = _write_structure(pristine, workdir / "pristine.cif")
    defect_path = _write_structure(defect, workdir / "vacancy.cif")
    base = _base_params(args)
    base.pop("structure", None)
    base["material"] = source
    base.setdefault("conv_thr", 1e-8 if args.get("conv_thr") in (None, "") else args["conv_thr"])
    nodes = [NodeSpec(
        id="pristine_scf", tool="qe",
        params={"calc": "scf", "prefix": "pristine", "structure": pristine_path, **base},
        repair="", max_repair_attempts=0,
    )]
    do_relax = bool(args.get("relax", True))
    for charge in charges:
        node_id = _charge_node_id(charge)
        charge_params = {"tot_charge": charge, **normalized_spin[charge]}
        if do_relax:
            nodes.append(NodeSpec(
                id=f"{node_id}_relax", tool="qe",
                params={
                    "calc": "relax", "prefix": f"defect_{node_id}", "structure": defect_path,
                    "force_threshold": float(args.get("force_threshold", 1.0e-3)),
                    "nstep": int(args.get("nstep", 200)),
                    **base, **charge_params,
                },
                repair="", max_repair_attempts=0,
            ))
            structure = f"${{nodes.{node_id}_relax.outputs.final_structure_cif}}"
            depends = [f"{node_id}_relax"]
        else:
            structure, depends = defect_path, []
        nodes.append(NodeSpec(
            id=f"{node_id}_scf", tool="qe",
            params={
                "calc": "scf", "prefix": f"defect_{node_id}_scf", "structure": structure,
                **base, **charge_params,
            },
            depends_on=depends,
            repair="", max_repair_attempts=0,
        ))
    template = GraphTemplate(
        template_id=_workflow_id("charged_vacancy", {
            "source": source, "element": element, "supercell": args.get("supercell"),
            "charges": charges, "correction_mode": correction_mode, "relax": do_relax,
        }),
        description="matched charged-vacancy formation and correction workflow",
        nodes=nodes,
    )
    payload = _run_graph(template, workdir / "graphs", **_runtime_controls(args))
    payload.update({
        "workflow_type": "charged_vacancy",
        "charges": charges,
        "fermi_levels_ev": fermi_levels,
        "band_gap_ev": band_gap,
        "correction_mode": correction_mode,
        "scope": (
            "finite-size treatment is limited to the supplied explicit correction or the "
            "first-order near-cubic Makov-Payne model; potential alignment and dielectric "
            "screening are external evidence, not inferred by this workflow"
        ),
        "structure": {"pristine": _jsonable(pristine.to_dict()), "defect": _jsonable(defect.to_dict())},
    })
    p_out = payload["nodes"]["pristine_scf"].get("outputs") or {}
    if (payload["nodes"]["pristine_scf"].get("state") != "succeeded"
            or not _finite_real(p_out.get("energy_ry"))
            or not _matching_atom_count(p_out.get("natoms"), len(pristine.atoms))):
        payload["status"] = "partial"
        payload["formation_lines"] = []
        payload["blocking_reasons"] = [
            "pristine SCF lacks successful finite energy and matched atom-count evidence"
        ]
        return payload
    mu_ry = args.get("chemical_potential_ry")
    if mu_ry is None and args.get("chemical_potential_ev") is not None:
        mu_ry = float(args["chemical_potential_ev"]) / RY_TO_EV
    if reference_mode == "same_method_bulk_per_atom":
        if (payload["nodes"]["pristine_scf"].get("state") != "succeeded"
                or not isinstance(p_out.get("energy_ry"), (int, float))
                or p_out.get("natoms") != len(pristine.atoms)):
            payload["status"] = "partial"
            payload["blocking_reasons"] = ["cannot derive elemental chemical potential from pristine evidence"]
            return payload
        mu_ry = p_out["energy_ry"] / len(pristine.atoms)
        payload["chemical_potential_reference"] = {
            "mode": reference_mode, "element": element, "source_node": "pristine_scf",
            "bulk_energy_ry": p_out["energy_ry"], "natoms": len(pristine.atoms),
            "chemical_potential_ry": mu_ry,
        }
    if mu_ry is None or not math.isfinite(float(mu_ry)):
        payload["status"] = "partial"
        payload["blocking_reasons"] = ["an absolute chemical potential is required"]
        return payload

    formation_lines = []
    charge_validations: Dict[str, Dict[str, Any]] = {}
    validation_failures: List[str] = []
    for charge in charges:
        node = payload["nodes"][f"{_charge_node_id(charge)}_scf"]
        out = node.get("outputs") or {}
        if (node.get("state") != "succeeded" or not _finite_real(out.get("energy_ry"))
                or not _matching_atom_count(out.get("natoms"), len(defect.atoms))):
            continue
        expected_magnetization = normalized_spin[charge].get("total_magnetization_muB")
        observed_magnetization = out.get("total_magnetization_muB")
        spin_verified = None
        if expected_magnetization is not None:
            spin_verified = (
                isinstance(observed_magnetization, (int, float))
                and not isinstance(observed_magnetization, bool)
                and math.isfinite(float(observed_magnetization))
                and abs(float(observed_magnetization) - float(expected_magnetization))
                <= magnetization_tolerance
            )
        spin_validation = {
            "expected_total_magnetization_muB": expected_magnetization,
            "observed_total_magnetization_muB": observed_magnetization,
            "tolerance_muB": magnetization_tolerance if expected_magnetization is not None else None,
            "verified": spin_verified,
        }
        charge_validations[str(charge)] = spin_validation
        if spin_verified is False:
            validation_failures.append(
                f"charge {charge:+d} did not verify the requested total magnetization"
            )
            continue
        try:
            _require_method_compatibility(
                p_out, out, compare_kpoints=True,
                common_elements=set(pristine.atoms.get_chemical_symbols()) & set(defect.atoms.get_chemical_symbols()),
            )
        except WorkflowError as exc:
            payload["status"] = "partial"
            payload["formation_lines"] = []
            payload["blocking_reasons"] = [str(exc)]
            return payload
        alignment = 0.0 if charge == 0 else float(_charge_value(
            alignments, charge, "potential_alignment_ev_by_charge"
        ))
        correction = correction_details[charge]["electrostatic_correction_ev"]
        points = [charged_vacancy_formation_energy(
            defect_energy_ry=out["energy_ry"], pristine_energy_ry=p_out["energy_ry"],
            chemical_potential_ry=float(mu_ry), charge=charge,
            fermi_level_from_vbm_ev=fermi, potential_alignment_ev=alignment,
            electrostatic_correction_ev=correction,
        ) for fermi in fermi_levels]
        formation_lines.append({
            "charge": charge,
            "node_id": f"{_charge_node_id(charge)}_scf",
            "defect_energy_ry": out["energy_ry"],
            "intercept_ev_at_vbm": points[0]["formation_energy_ev"] - charge * fermi_levels[0],
            "slope_ev_per_ev": charge,
            "potential_alignment_ev": alignment,
            "correction": correction_details[charge],
            "spin_validation": spin_validation,
            "points": points,
        })
    payload["formation_lines"] = formation_lines
    payload["charge_validations"] = charge_validations
    if len(formation_lines) != len(charges):
        payload["status"] = "partial"
        payload["blocking_reasons"] = validation_failures or [
            "one or more charged-defect SCF nodes lack verified energy/atom-count evidence"
        ]
        return payload
    transitions = []
    for i, left in enumerate(formation_lines):
        for right in formation_lines[i + 1:]:
            if left["charge"] == right["charge"]:
                continue
            level = (left["intercept_ev_at_vbm"] - right["intercept_ev_at_vbm"]) / (
                right["charge"] - left["charge"]
            )
            transitions.append({
                "charges": [left["charge"], right["charge"]],
                "fermi_level_from_vbm_ev": level,
                "within_supplied_gap": None if band_gap is None else 0 <= level <= band_gap,
            })
    payload["transition_levels"] = transitions
    payload["status"] = "complete"
    return payload
