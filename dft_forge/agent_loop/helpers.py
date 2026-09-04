"""Shared constants and pure helpers for the agent loop package."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_CHAT_SKIP_TOOLS = {"help"}

_ELEMENTS = {
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S",
    "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga",
    "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd",
    "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm",
    "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os",
    "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa",
    "U", "Np", "Pu",
}

# Chinese aliases for quick material lookup in chat text
_ZH_MATERIAL = {"硅": "Si", "铝": "Al", "氯化钠": "NaCl", "食盐": "NaCl", "氧化镁": "MgO"}

# host elements per 2D kind — used to reject meaningless self-doping
_2D_HOST_ELEMENTS = {
    "graphene": {"C"},
    "bn": {"B", "N"},
    "mos2": {"Mo", "S"}, "ws2": {"W", "S"}, "mose2": {"Mo", "Se"},
    "wse2": {"W", "Se"}, "mote2": {"Mo", "Te"}, "wte2": {"W", "Te"},
}
_2D_KIND_NAMES = {
    "graphene": "石墨烯", "bn": "h-BN", "mos2": "MoS2", "ws2": "WS2",
    "mose2": "MoSe2", "wse2": "WSe2", "mote2": "MoTe2", "wte2": "WTe2",
}


def extract_formula(message: str) -> Optional[str]:
    """Pull a chemical formula (e.g. CaTiO3, GaAs, NaCl) out of chat text.

    Accepts any casing (catio3 → CaTiO3) as long as tokens map to real
    element symbols. Longest match wins so GaAs is preferred over Ga/As.
    """
    text = message
    for zh, en in _ZH_MATERIAL.items():
        if zh in text:
            return en
    tokens = re.findall(r"[A-Za-z]{1,2}\d{0,3}(?:\s?[A-Za-z]{1,2}\d{0,3})*", text)
    best = None
    for tok in tokens:
        parts = re.findall(r"[A-Za-z]{1,2}\d{0,3}", tok)
        syms = []
        ok = True
        for p in parts:
            m = re.match(r"([A-Za-z]{1,2})(\d*)", p)
            sym, num = m.group(1), m.group(2)
            cand = None
            for probe in (sym.capitalize(), sym.upper()):
                if probe in _ELEMENTS:
                    cand = probe
                    break
            if cand is None:
                ok = False
                break
            syms.append(cand + num)
        if not ok or not syms or len(syms) > 4:
            continue
        formula = "".join(syms)
        if best is None or len(formula) > len(best):
            best = formula
    return best


def _read_text(path: Any) -> Optional[str]:
    try:
        return Path(str(path)).read_text()
    except OSError:
        return None


def _viewer_payload_from_file(path: Any) -> Optional[Dict[str, Any]]:
    """CIF viewer payload from a structure file path."""
    cif = _read_text(path)
    if not cif:
        return None
    natoms = None
    formula = None
    try:
        from ase.io import read as ase_read

        atoms = ase_read(str(path))
        natoms = len(atoms)
        formula = atoms.get_chemical_formula()
    except Exception:
        pass
    return {
        "formula": formula or Path(str(path)).stem,
        "cif": cif,
        "natoms": natoms,
        "source": str(path),
    }


def _viewer_payload_for_material(material: Any) -> Optional[Dict[str, Any]]:
    """CIF viewer payload for a material key or raw formula (e.g. CaTiO3)."""
    if not material or not str(material).strip():
        return None
    try:
        import tempfile

        from ase.io import write as ase_write

        from dft_forge.compiler import MATERIAL_DB, build_atoms, formula_atoms

        name = str(material).strip()
        atoms = build_atoms(name) if name in MATERIAL_DB else formula_atoms(name)[0]
        with tempfile.NamedTemporaryFile(suffix=".cif", mode="w+", delete=False) as tf:
            ase_write(tf.name, atoms, format="cif")
            cif = Path(tf.name).read_text()
        Path(tf.name).unlink(missing_ok=True)
        return {
            "formula": atoms.get_chemical_formula(),
            "cif": cif,
            "natoms": len(atoms),
            "source": name,
        }
    except Exception:
        return None


def _extract_charts(nodes: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Pull plottable curves + headline numbers from graph node outputs."""
    chart: Dict[str, Any] = {}
    for info in nodes.values():
        outs = info.get("outputs") or {}
        if outs.get("eigenvalues_ev"):
            bands = {
                "kind": "bands",
                "eigenvalues_ev": outs["eigenvalues_ev"],
                "fermi_ev": outs.get("fermi_ev"),
                "band_gap_ev": outs.get("band_gap_ev"),
                "is_metal": outs.get("is_metal"),
                "n_bands": outs.get("n_bands"),
                "n_kpoints": outs.get("n_kpoints"),
            }
            for key in ("k_axis", "k_ticks", "k_labels"):
                if outs.get(key):
                    bands[key] = outs[key]
            chart["bands"] = bands
        if outs.get("dos_curve"):
            chart["dos"] = {"kind": "dos", **outs["dos_curve"], "fermi_ev": outs.get("fermi_ev")}
        if outs.get("pdos_curve"):
            chart["pdos"] = {"kind": "pdos", **outs["pdos_curve"], "fermi_ev": outs.get("fermi_ev")}
        for k in ("energy_ry", "a_angstrom", "pressure_kbar", "max_force_ev_ang"):
            v = outs.get(k)
            if v is None:
                continue
            metrics = chart.setdefault("metrics", {})
            if k not in metrics or (not metrics[k] and v):  # nonzero wins over nscf placeholder zeros
                metrics[k] = v
    return chart or None


def _dedupe_node_log(log: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collapse a node event stream to one entry per node (latest state, first-seen order)."""
    latest: Dict[str, Any] = {}
    order: List[str] = []
    for e in log:
        n = e.get("node")
        if n not in latest:
            order.append(n)
        latest[n] = e.get("state")
    return [{"node": n, "state": latest[n]} for n in order]


def _load_env_file(path: Path) -> None:
    """Load 'export KEY=VALUE' lines from .env without overwriting real env."""
    if not path.exists():
        return
    import os
    import re

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$", line)
        if not m:
            continue
        key, val = m.group(1), m.group(2).strip().strip("'\"")
        if key not in os.environ:
            os.environ[key] = val
