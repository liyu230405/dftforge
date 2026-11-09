"""Numeric field extraction: run record -> the `field` names used by checks.

Each field in QUESTIONS.md / the generated JSONs has exactly one extractor
here. Extractors are deliberately permissive about *where* the number lives
(tool payload, node payload, parsed result) and strict about *what* counts:
they never parse the agent's Chinese reply for a number, because a fabricated
number in prose must not be able to earn a numeric point. Missing -> None,
which the scorer treats as 0 for that check (never as a pass).
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

Number = Optional[float]

RY_TO_EV = 13.605693122994

# ---------------------------------------------------------------- walkers


def _iter_payloads(record: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """Every dict reachable from results[].json and chain[].nodes[]."""
    stack: List[Any] = []
    for entry in record.get("results") or []:
        stack.append(entry.get("json"))
    for node in record.get("chain") or []:
        stack.append(node)
    while stack:
        obj = stack.pop()
        if isinstance(obj, dict):
            yield obj
            stack.extend(obj.values())
        elif isinstance(obj, list):
            stack.extend(obj)


def _first(record: Dict[str, Any], *keys: str) -> Any:
    """First non-None value stored under any of `keys`, anywhere in payloads."""
    for payload in _iter_payloads(record):
        for key in keys:
            if key in payload and payload[key] is not None:
                return payload[key]
    return None


def _num(value: Any) -> Number:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return None if math.isnan(float(value)) else float(value)
    if isinstance(value, str):
        m = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eEdD][-+]?\d+)?", value.replace("D", "E"))
        if m:
            try:
                return float(m.group(0))
            except ValueError:
                return None
    return None


def _num_field(record: Dict[str, Any], *keys: str) -> Number:
    return _num(_first(record, *keys))


def _floats(value: Any) -> List[float]:
    out: List[float] = []
    if isinstance(value, (list, tuple)):
        for item in value:
            n = _num(item)
            if n is not None:
                out.append(n)
    return out


def _iter_with_path(record: Dict[str, Any]) -> Iterable[tuple]:
    """(dotted key path, dict) for every dict, depth-first in document order.

    The path is relative to each results[].json / chain node, so a workflow's
    own tool id never leaks into it: node names like ``defect_relax`` or
    ``structure.pristine`` are what label a payload.
    """
    def walk(obj: Any, path: str):
        if isinstance(obj, dict):
            yield path, obj
            for k, v in obj.items():
                yield from walk(v, f"{path}.{k}")
        elif isinstance(obj, list):
            for item in obj:
                yield from walk(item, path)
    for entry in record.get("results") or []:
        yield from walk(entry.get("json"), "")
    for node in record.get("chain") or []:
        yield from walk(node, "")


# Derived cells from symmetry analysis (conventional/primitive) are not the
# structure that was computed; their atom counts must not be picked up.
_DERIVED_CELL_RE = re.compile(r"\.(symmetry|conventional_cell|primitive_cell)\b")


def _structures_with_path(record: Dict[str, Any]) -> List[tuple]:
    out = []
    for path, payload in _iter_with_path(record):
        if _DERIVED_CELL_RE.search(path):
            continue
        if any(k in payload for k in ("n_atoms", "natoms", "num_atoms")):
            out.append((path, payload))
    return out


def _structures(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Payloads that look like a structure (carry an atom count)."""
    return [payload for _, payload in _structures_with_path(record)]


def _atom_count(payload: Dict[str, Any]) -> Number:
    return _num(payload.get("n_atoms") or payload.get("natoms") or payload.get("num_atoms"))


def _label(payload: Dict[str, Any]) -> str:
    bits = [str(payload.get(k) or "") for k in
            ("label", "name", "tag", "step_id", "formula", "path", "prefix", "structure_id",
             "command")]
    return " ".join(bits).lower()


def _atoms_matching(record: Dict[str, Any], *needles: str) -> Number:
    for path, payload in _structures_with_path(record):
        lab = f"{path.lower()} {_label(payload)}"
        if any(n.lower() in lab for n in needles):
            return _atom_count(payload)
    return None


# ---------------------------------------------------------------- extractors


def _a_conventional(record: Dict[str, Any]) -> Number:
    v = _num_field(record, "a_conventional_angstrom", "a_conventional", "lattice_a_angstrom")
    if v is not None:
        return v
    # celldm(1) in bohr -> conventional cubic a
    alat = _num_field(record, "alat_bohr", "celldm1", "celldm_1")
    if alat is not None:
        return alat * 0.529177210903
    return _num_field(record, "a_angstrom", "lattice_constant_angstrom")


def _pressure_abs(record: Dict[str, Any]) -> Number:
    v = _num_field(record, "pressure_kbar", "pressure", "final_pressure_kbar")
    return None if v is None else abs(v)


def _max_force(record: Dict[str, Any]) -> Number:
    return _num_field(record, "max_force_ry_bohr", "max_force", "total_force")


def _band_gap(record: Dict[str, Any]) -> Number:
    return _num_field(record, "band_gap_ev", "band_gap", "gap_ev", "indirect_gap_ev")


def _e_fermi(record: Dict[str, Any]) -> Number:
    return _num_field(record, "e_fermi_ev", "fermi_energy_ev", "fermi_ev", "efermi", "fermi_energy")


def _dos_at_fermi(record: Dict[str, Any]) -> Number:
    return _num_field(record, "dos_at_fermi", "dos_at_ef", "dos_ef")


def _pdos_cu_at_fermi(record: Dict[str, Any]) -> Number:
    v = _num_field(record, "pdos_cu_at_fermi", "pdos_at_fermi")
    if v is not None:
        return v
    by_element = _first(record, "element_dos_at_fermi", "pdos_by_element_at_fermi")
    if isinstance(by_element, dict):
        for key, val in by_element.items():
            if str(key).strip().lower() == "cu":
                return _num(val)
    for payload in _iter_payloads(record):
        lab = _label(payload)
        if "cu" in lab and any(k in payload for k in ("dos_at_fermi", "dos_at_ef")):
            return _num(payload.get("dos_at_fermi") or payload.get("dos_at_ef"))
    return None


_MOMENT_KEYS = ("total_magnetization_muB", "total_magnetization",
                "total_magnetization_bohr_mag")


def _order_of(path: str, payload: Dict[str, Any]) -> Optional[str]:
    """Magnetic order a payload belongs to: realized > requested > node name."""
    for key in ("realized_order", "requested_order", "magnetic_order"):
        v = payload.get(key)
        if isinstance(v, str) and v.strip().lower() in ("nm", "fm", "afm"):
            return v.strip().lower()
    m = re.search(r"\.(nm|fm|afm)(?:\.|$)", path.lower())
    return m.group(1) if m else None


def _fm_moment(record: Dict[str, Any]) -> Number:
    v = _num_field(record, "fm_total_moment_muB")
    if v is not None:
        return abs(v)
    # Prefer the moment that belongs to the FM state: in a NM/FM/AFM sweep a
    # bare total_magnetization lookup lands on whichever state comes first.
    for path, payload in _iter_with_path(record):
        if _order_of(path, payload) != "fm":
            continue
        for key in _MOMENT_KEYS:
            n = _num(payload.get(key))
            if n is not None:
                return abs(n)
    v = _num_field(record, *_MOMENT_KEYS)
    return None if v is None else abs(v)


def _formation(record: Dict[str, Any]) -> Number:
    return _num_field(record, "formation_energy_ev", "formation_energy",
                      "e_formation_ev", "defect_formation_energy_ev")


def _formation_per_atom(record: Dict[str, Any]) -> Number:
    v = _num_field(record, "formation_energy_ev_per_atom", "formation_energy_per_atom")
    if v is not None:
        return v
    tot = _formation(record)
    n = _num_field(record, "n_atoms_formula", "n_atoms")
    if tot is not None and n:
        return tot / n
    return None


def _e_ads(record: Dict[str, Any]) -> Number:
    return _num_field(record, "e_ads_ev", "adsorption_energy_ev", "e_ads", "eads_ev")


def _phonon_freqs(record: Dict[str, Any]) -> Optional[List[float]]:
    for key in ("phonon_freqs_thz", "frequencies_thz", "freqs_thz", "omega_thz"):
        v = _first(record, key)
        f = _floats(v)
        if f:
            return sorted(f)
    return None


def _optical_mode(record: Dict[str, Any]) -> Number:
    v = _num_field(record, "optical_mode_thz", "to_mode_thz")
    if v is not None:
        return v
    freqs = _phonon_freqs(record)
    if freqs and len(freqs) > 3:
        return freqs[3]           # first mode above the 3 acoustic ones
    return None


def _magnetic_orders(record: Dict[str, Any]) -> Optional[List[str]]:
    v = _first(record, "magnetic_orders", "orders", "magnetic_configurations")
    if isinstance(v, list) and v:
        return [str(x).lower() for x in v]
    found = []
    for payload in _iter_payloads(record):
        lab = _label(payload)
        for order in ("nm", "fm", "afm"):
            if re.search(rf"\b{order}\b", lab) and order not in found:
                found.append(order)
    return found or None


def _energy_by_order(record: Dict[str, Any]) -> Dict[str, float]:
    """order-name -> total energy, for the magnetic-ordering comparison."""
    out: Dict[str, float] = {}
    table = _first(record, "energies_by_order", "order_energies")
    if isinstance(table, dict):
        for k, v in table.items():
            n = _num(v)
            if n is not None:
                out[str(k).lower()] = n
        if out:
            return out
    for path, payload in _iter_with_path(record):
        e = _num(payload.get("total_energy_ev") or payload.get("total_energy")
                 or payload.get("energy_ev"))
        if e is None:
            ry = _num(payload.get("energy_ry") or payload.get("total_energy_ry"))
            e = None if ry is None else ry * RY_TO_EV
        if e is None:
            continue
        order = _order_of(path, payload)
        if order is None:
            lab = _label(payload)
            order = next((o for o in ("afm", "fm", "nm")
                          if re.search(rf"\b{o}\b", lab)), None)
        if order and order not in out:
            out[order] = e
    return out


def _lowest_energy_order(record: Dict[str, Any]) -> Optional[str]:
    v = _first(record, "lowest_energy_order", "ground_state_order")
    if isinstance(v, str):
        return v.lower()
    # Decide from the energies themselves when they are there; the product's
    # own comparison verdict is only a fallback.
    table = _energy_by_order(record)
    if len(table) >= 2:
        return min(table.items(), key=lambda kv: kv[1])[0]
    v = _first(record, "ground_state")
    if isinstance(v, str) and v.strip().lower() in ("nm", "fm", "afm"):
        return v.strip().lower()
    return None


def _convergence_selection(record: Dict[str, Any]) -> Number:
    v = _num_field(record, "converged_ecutwfc", "selected_ecutwfc",
                   "recommended_ecutwfc", "convergence_selection")
    if v is not None:
        return v
    sel = _first(record, "selection")
    if isinstance(sel, dict) and isinstance(sel.get("selected"), dict):
        return _num(sel["selected"].get("ecutwfc"))
    return None


def convergence_reference(record: Dict[str, Any]) -> Number:
    """ecutwfc of the point the selection was judged against, if reported."""
    sel = _first(record, "selection")
    if isinstance(sel, dict) and isinstance(sel.get("reference"), dict):
        return _num(sel["reference"].get("ecutwfc"))
    return None


def _kpoints_kz(record: Dict[str, Any]) -> Number:
    for payload in _iter_payloads(record):
        kp = payload.get("kpoints") or payload.get("kpts") or payload.get("k_grid")
        f = _floats(kp)
        if len(f) >= 3:
            return f[2]
    return None


def _cif_fract_z(path: str) -> List[float]:
    """fract_z column of a CIF's atom_site loop (pymatgen/ase style)."""
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    cols: List[str] = []
    out: List[float] = []
    in_loop = False
    for line in lines:
        t = line.strip()
        if t.startswith("loop_"):
            cols, in_loop = [], True
            continue
        if in_loop and t.startswith("_"):
            cols.append(t.split()[0])
            continue
        if in_loop and cols and "_atom_site_fract_z" in cols and t and not t.startswith("_"):
            parts = t.split()
            if len(parts) >= len(cols):
                n = _num(parts[cols.index("_atom_site_fract_z")])
                if n is not None:
                    out.append(n % 1.0)
            continue
        if in_loop and cols and not t:
            in_loop = False
    return out


def _vacuum(record: Dict[str, Any]) -> Number:
    v = _num_field(record, "vacuum_angstrom", "vacuum", "vacuum_thickness_angstrom")
    if v is not None:
        return v
    # Derive it from the built slab: c-axis length minus the atoms' z extent.
    # Needs a c-axis along z and the structure file the tool wrote.
    for payload in _iter_payloads(record):
        cell, out = payload.get("cell"), payload.get("output")
        if not (isinstance(cell, list) and len(cell) == 3 and isinstance(out, str)
                and out.lower().endswith(".cif")):
            continue
        c = _floats(cell[2])
        if len(c) != 3 or abs(c[0]) > 1e-6 or abs(c[1]) > 1e-6 or c[2] <= 0:
            continue
        zs = _cif_fract_z(out)
        if not zs:
            continue
        # Atoms may wrap across the periodic boundary: the vacuum is the
        # largest gap between consecutive fractional z values.
        zs = sorted(zs)
        gaps = [b - a for a, b in zip(zs, zs[1:])] + [zs[0] + 1.0 - zs[-1]]
        return round(max(gaps) * c[2], 4)
    return None


def _n_atoms(record: Dict[str, Any]) -> Number:
    counts = [_atom_count(p) for p in _structures(record)]
    counts = [c for c in counts if c is not None]
    return max(counts) if counts else None


def _n_atoms_pristine(record: Dict[str, Any]) -> Number:
    v = _atoms_matching(record, "pristine", "perfect", "bulk", "host", "reference", "完美", "本征")
    if v is not None:
        return v
    return _n_atoms(record)


def _n_atoms_defect(record: Dict[str, Any]) -> Number:
    return _atoms_matching(record, "vacancy", "defect", "vac", "空位")


def _n_atoms_doped(record: Dict[str, Any]) -> Number:
    return _atoms_matching(record, "doped", "dope", "substitut", "掺杂")


def _n_atoms_N(record: Dict[str, Any]) -> Number:
    """Count of N atoms in the N-doped graphene cell."""
    v = _num_field(record, "n_atoms_N", "n_N", "count_N")
    if v is not None:
        return v
    for payload in _iter_payloads(record):
        comp = payload.get("composition") or payload.get("species_counts")
        if isinstance(comp, dict):
            for key, val in comp.items():
                if str(key).strip().lower() == "n":
                    return _num(val)
    for payload in _iter_payloads(record):
        symbols = payload.get("symbols") or payload.get("species")
        if isinstance(symbols, list) and symbols:
            n = sum(1 for s in symbols if str(s).strip().capitalize() == "N")
            if n:
                return float(n)
    return None


EXTRACTORS: Dict[str, Callable[[Dict[str, Any]], Any]] = {
    "a_conventional_angstrom": _a_conventional,
    "pressure_kbar_abs": _pressure_abs,
    "max_force_ry_bohr": _max_force,
    "band_gap_ev": _band_gap,
    "e_fermi_ev": _e_fermi,
    "dos_at_fermi": _dos_at_fermi,
    "pdos_cu_at_fermi": _pdos_cu_at_fermi,
    "fm_total_moment_muB": _fm_moment,
    "formation_energy_ev": _formation,
    "formation_energy_ev_per_atom": _formation_per_atom,
    "e_ads_ev": _e_ads,
    "phonon_freqs_thz": _phonon_freqs,
    "optical_mode_thz": _optical_mode,
    "magnetic_orders": _magnetic_orders,
    "lowest_energy_order": _lowest_energy_order,
    "convergence_selection": _convergence_selection,
    "kpoints_kz": _kpoints_kz,
    "vacuum_angstrom": _vacuum,
    "n_atoms": _n_atoms,
    "n_atoms_pristine": _n_atoms_pristine,
    "n_atoms_defect": _n_atoms_defect,
    "n_atoms_doped": _n_atoms_doped,
    "n_atoms_N": _n_atoms_N,
}


def extract(record: Dict[str, Any], field: str) -> Any:
    fn = EXTRACTORS.get(field)
    if fn is None:
        raise KeyError(f"no extractor for numeric field {field!r}")
    try:
        return fn(record)
    except Exception:
        return None


def extract_all(record: Dict[str, Any]) -> Dict[str, Any]:
    return {name: extract(record, name) for name in EXTRACTORS}
