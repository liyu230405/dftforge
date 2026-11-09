from __future__ import annotations

from typing import Dict

from ._helpers import _error_response, _json_dumps


RY_TO_EV = 13.605693122994


def cmd_thermo_eads(args: argparse.Namespace) -> int:
    """thermo eads: E_ads = E(adsorbed) − E(surface) − (n_ads/n_mol)·E(molecule)."""
    try:
        e_a, e_s, e_m = args.ry_a, args.ry_s, args.ry_m
    except TypeError:
        return _error_response("eads requires --ry-a, --ry-s, --ry-m energies in Ry")
    n_ads = int(getattr(args, "nat_a", 0) or 0)   # adsorbed atoms on the surface
    n_mol = int(getattr(args, "nat_mol", 0) or 0)  # atoms in the gas molecule
    if n_ads > 0 and n_mol > 0:
        # one O atom adsorbed but the reference gas is O2 → subtract E(O2)/2
        ratio = n_ads / n_mol
    else:
        ratio = 1.0
    e_ads = e_a - e_s - ratio * e_m
    data = {
        "command": "thermo.eads",
        "ok": True,
        "e_ads_ev": round(e_ads * RY_TO_EV, 4),
        "e_ads_ry": round(e_ads, 6),
        "e_ads_ev_per_adsorbate": round(e_ads * RY_TO_EV / n_ads, 4) if n_ads else None,
        "molecule_ratio": round(ratio, 4),
        "formula_note": (
            f"E_ads = E(surf+ads) − E(surf) − {ratio:g}·E(mol) "
            "(molecule energy scaled by adsorbed-atom count); negative = exothermic"
        ),
    }
    print(_json_dumps(data))
    return 0



def cmd_thermo_formation(args: argparse.Namespace) -> int:
    """thermo formation: E_form from compound + elemental reference energies."""
    from dft_forge.compiler import formula_atoms

    try:
        atoms, _proto = formula_atoms(args.formula)
    except Exception as exc:
        return _error_response(f"cannot parse formula {args.formula!r}: {exc}")

    counts: Dict[str, int] = {}
    for s in atoms.get_chemical_symbols():
        counts[s] = counts.get(s, 0) + 1

    refs = dict(zip(args.ref_element or [], args.ref_energy or []))
    refs = {el: float(e) for el, e in refs.items()}
    ref_n = dict(zip(args.ref_element or [], args.ref_natoms or []))
    ref_n = {el: int(n) for el, n in ref_n.items()}

    missing = [el for el in counts if el not in refs]
    if missing:
        return _error_response(
            f"missing reference energy for: {', '.join(missing)} — run structure.reference + t0_scf for each"
        )

    e_comp = float(args.compound_energy)
    e_form = e_comp - sum(n * (refs[el] / ref_n.get(el, 1)) for el, n in counts.items())
    n_atoms = sum(counts.values())
    data = {
        "command": "thermo.formation",
        "ok": True,
        "formula": args.formula,
        "formation_energy_ev_per_atom": round(e_form * RY_TO_EV / n_atoms, 4),
        "formation_energy_ev_per_formula": round(e_form * RY_TO_EV, 4),
        "composition": counts,
        "n_atoms": n_atoms,
        "formula_note": "E_form = E(compound) − Σ n_i·e_i(element per atom); negative = thermodynamically stable vs elements",
    }
    print(_json_dumps(data))
    return 0
