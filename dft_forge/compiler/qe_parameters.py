"""Read-only, reproducible QE parameter recommendations for the product layer.

The returned numbers are initial choices, not a convergence certificate.  File
metadata is read again when a method is confirmed, so a changed pseudo changes
the method digest too. No engine, generated file, or model is needed here.
"""

from __future__ import annotations

import math
import hashlib
import json
from pathlib import Path

import numpy as np

from dft_forge.compiler import (
    DEFAULT_PSEUDO_DIR, MATERIAL_DB, build_atoms, formula_atoms,
    profile_from_atoms, read_structure, resolve_band_count,
)
from dft_forge.compiler.pseudopotentials import read_pseudo_metadata

PARAMETERS = ("ecutwfc", "ecutrho", "kpoints", "nbnd", "nkpoints_bands")


def _positive(value, name, *, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须是有效数字")
    if not math.isfinite(value) or value <= 0 or (integer and not isinstance(value, int)):
        raise ValueError(f"{name} 必须是正{'整数' if integer else '数'}")
    return value


def resolve_qe_parameters(*, material=None, structure=None, accuracy="balanced",
                          overrides=None, task="t2_bands", pseudo_dir=None):
    """Resolve actual values using the same material/profile as QECompiler."""
    pseudo_dir = Path(pseudo_dir or DEFAULT_PSEUDO_DIR)
    overrides = {k: v for k, v in (overrides or {}).items() if v is not None}
    if structure:
        atoms = read_structure(Path(structure))
        profile = profile_from_atoms(atoms, pseudo_dir)
        profile_source = "上传/生成结构的材料初值"
    elif material in MATERIAL_DB:
        atoms = build_atoms(material)
        profile = MATERIAL_DB[material]
        profile_source = f"材料库 {material} 的初值"
    elif material:
        atoms, _ = formula_atoms(material)
        profile = profile_from_atoms(atoms, pseudo_dir)
        profile_source = "已登记晶体原型的初值"
    else:
        raise ValueError("自动定参需要明确材料或已生成的结构")
    if not len(atoms) or atoms.cell.rank != 3:
        raise ValueError("自动定参需要非空原子结构和完整晶胞")
    pseudo_map = {el: read_pseudo_metadata(pseudo_dir / name) for el, name in profile["pseudos"].items()}
    pseudos = [{"element": el, **meta} for el, meta in sorted(pseudo_map.items())]
    spin = bool(overrides.get("spin_polarized") or
                str(overrides.get("magnetic_order", "")).lower().replace("-", "") in {"fm", "afm"})
    bands = resolve_band_count(
        valence_electrons=sum(pseudo_map[str(el)]["z_valence"] for el in atoms.symbols),
        natoms=len(atoms), is_metal=bool(profile["is_metal"]),
        nbnd=overrides.get("nbnd"), tot_charge=overrides.get("tot_charge", 0.0),
        spin_polarized=spin, extra_bands=12 if profile["is_metal"] or task == "t2_dos" else 8,
    )
    nelec, nocc = bands["nelec"], bands["minimum_bands"]
    reasons, notes = {}, []

    # Use a real header recommendation where present; otherwise explicitly
    # identify the material/profile fallback as an untested starting value.
    suggested_wfc = [p["suggested_ecutwfc"] for p in pseudos if p.get("suggested_ecutwfc")]
    base_wfc = max([float(profile["ecutwfc_default"]), *suggested_wfc])
    ecut = base_wfc
    reasons["ecutwfc"] = ("赝势文件推荐值与材料初值中的较大值" if suggested_wfc else
                          profile_source + "；赝势未提供可用的推荐截断能")
    ratio = max(4.0, float(profile["ecutrho_default"]) / float(profile["ecutwfc_default"]))
    if accuracy == "precise":
        ecut = float(math.ceil(ecut * 1.25 / 5) * 5)
        reasons["ecutwfc"] += "；精细策略提高 25% 并向上取整"
    if "ecutwfc" in overrides:
        ecut = float(_positive(overrides["ecutwfc"], "ecutwfc"))
        reasons["ecutwfc"] = "用户自定义值"
    rho = max([ecut * ratio, *[p["suggested_ecutrho"] for p in pseudos if p.get("suggested_ecutrho")]])
    reasons["ecutrho"] = f"按材料初值比例 {ratio:g} × ecutwfc，并兼顾赝势推荐值"
    if "ecutrho" in overrides:
        rho = float(_positive(overrides["ecutrho"], "ecutrho"))
        reasons["ecutrho"] = "用户自定义值"
    if rho < 4 * ecut:
        raise ValueError("ecutrho 不能小于 4 × ecutwfc")

    pbc = [bool(x) for x in atoms.pbc]
    if structure or material not in MATERIAL_DB:
        spacing = {"quick": 0.50, "precise": 0.25}.get(accuracy, 0.35)
        lengths = np.linalg.norm(np.asarray(atoms.cell.reciprocal()), axis=1) * 2 * math.pi
        mesh = [max(1, int(math.ceil(length / spacing))) if periodic else 1
                for length, periodic in zip(lengths, pbc)]
        grid = mesh + [0 if not periodic or n == 1 else 1 for n, periodic in zip(mesh, pbc)]
        reasons["kpoints"] = f"按倒空间间距 ≤ {spacing:g} Å⁻¹ 生成；非周期方向取 1"
    else:
        grid = list(profile["kpoints_default"])
        reasons["kpoints"] = profile_source + "的 SCF 网格"
        if accuracy == "precise":
            grid[:3] = [max(n, int(math.ceil(n * 1.5))) if periodic else 1 for n, periodic in zip(grid[:3], pbc)]
            reasons["kpoints"] += "；精细策略沿周期方向加密 1.5 倍"
    if "kpoints" in overrides:
        grid = list(overrides["kpoints"]) if isinstance(overrides["kpoints"], (list, tuple)) else []
        if len(grid) != 6 or any(type(n) is not int or n < 1 for n in grid[:3]) or any(type(n) is not int or n not in (0, 1) for n in grid[3:]):
            raise ValueError("kpoints 需要 3 个正整数网格与 3 个 0/1 偏移")
        if any(not periodic and (grid[i] != 1 or grid[i + 3] != 0) for i, periodic in enumerate(pbc)):
            raise ValueError("非周期方向的 k 网格必须为 1，偏移必须为 0")
        reasons["kpoints"] = "用户自定义 SCF 网格"
    extra, nbnd = bands["extra_bands"], bands["nbnd"]
    occupation = "共线自旋计算采用保守占据上界" if spin else "非自旋计算占据带至少"
    reasons["nbnd"] = f"计入电荷后共 {nelec:g} 个电子；{occupation} {nocc}，另留 {extra} 条空带，且总数 ≥ 4 × 原子数"
    if "nbnd" in overrides:
        reasons["nbnd"] = "用户自定义能带总数（已检查占据带下限）"
    band_points = 60 if accuracy == "quick" else (160 if accuracy == "precise" else 100)
    if accuracy not in {"quick", "precise"} and structure and len(atoms) > 4:
        band_points = 60 if len(atoms) <= 12 else 40
    reasons["nkpoints_bands"] = "按精度策略和结构大小选择路径采样数；高对称路径由 ASE 从晶格生成"
    if "nkpoints_bands" in overrides:
        band_points = _positive(overrides["nkpoints_bands"], "nkpoints_bands", integer=True)
        if band_points < 10:
            raise ValueError("能带路径点数至少为 10")
        reasons["nkpoints_bands"] = "用户自定义路径采样数"
    path = None
    if task == "t2_bands":
        path = atoms.cell.bandpath(npoints=band_points, pbc=pbc).path
    if structure:
        notes.append("上传结构的金属性尚未独立判定；沿用编译器的展宽策略")
    if task == "t2_dos":
        notes.append("DOS 的 NSCF 均匀网格在上述 SCF 网格基础上加倍，非周期方向保持 1")
    notes.append("这些是可复现的初始参数，尚未通过收敛测试；能带路径采样不等同于带隙收敛")
    result = {
        "status": "recommended", "resolver": "qe-parameters-v1",
        "parameters": {"ecutwfc": ecut, "ecutrho": rho, "kpoints": grid,
                       "nbnd": nbnd, "nkpoints_bands": band_points},
        "reasons": reasons, "pseudopotentials": pseudos,
        "structure": {"formula": atoms.get_chemical_formula(), "natoms": len(atoms),
                      "pbc": pbc, "valence_electrons": sum(pseudo_map[str(el)]["z_valence"] for el in atoms.symbols),
                      "electron_count": nelec, "tot_charge": bands["tot_charge"] if "tot_charge" in bands else bands["charge"],
                      "spin_polarized": spin},
        "band_path": path, "validation": "not_tested", "notes": notes,
        "occupations": profile["occupations"], "smearing": profile["smearing"],
        "degauss": profile["degauss"],
        "sources": {key: "user" if key in overrides else "deterministic" for key in PARAMETERS},
        "overrides_applied": {key: overrides[key] for key in PARAMETERS if key in overrides},
        "density_cutoff_ratio": ratio,
    }
    result["resolution_digest"] = hashlib.sha256(json.dumps({
        "recommendation": result,
        "cell": atoms.cell.tolist(), "positions": atoms.positions.tolist(),
        "symbols": list(atoms.get_chemical_symbols()),
    }, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    return result


def convergence_proposal(recommendation):
    """A bounded 3×3 sensitivity scan; final choices require an interior plateau."""
    values = recommendation["parameters"]
    base = values["ecutwfc"]
    cutoffs = [round(base * scale, 6) for scale in (1.0, 1.25, 1.5)]
    grids = []
    for delta in (0, 2, 4):
        grid = [n + delta if periodic else 1 for n, periodic in zip(values["kpoints"][:3], recommendation["structure"]["pbc"])] + values["kpoints"][3:]
        if grid not in grids:
            grids.append(grid)
    return {"ecutwfc_values": cutoffs, "kpoints_values": grids,
            "ecutrho_ratio": recommendation["density_cutoff_ratio"],
            "smearing_values": [recommendation["smearing"]],
            "degauss_values": [recommendation["degauss"]],
            "occupations": recommendation["occupations"],
            "threshold_mev_per_atom": 1.0, "job_count": len(cutoffs) * len(grids),
            "max_jobs": 9, "scope": "bounded_total_energy_sensitivity_not_band_gap_validation"}


def planner_parameter_context(message, ctx):
    """Compact scientific context for a configured LLM; no model/engine call."""
    import re

    config = ctx.get("run_config") or {}
    result = {"method_preferences": {key: config[key] for key in
              ("accuracy", "auto_parameters", "converge_first", *PARAMETERS) if key in config}}
    source = ctx.get("last_structure_source")
    candidates = ([None] if source else [name for name in sorted(MATERIAL_DB)
                  if re.search(r"(?<![A-Za-z0-9])" + re.escape(name) + r"(?![A-Za-z0-9])", message)][:3])
    resolved = []
    for material in candidates:
        try:
            rec = resolve_qe_parameters(material=material, structure=source,
                                        accuracy=config.get("accuracy", "balanced"),
                                        overrides={key: config[key] for key in PARAMETERS if config.get(key) is not None},
                                        task="t0_scf")
            resolved.append({"material": material, **{key: rec[key] for key in
                            ("parameters", "structure", "reasons", "validation", "pseudopotentials")}})
        except (ValueError, OSError, KeyError, TypeError):
            resolved.append({"material": material, "status": "requires_structure_or_pseudopotential_review"})
    result["method_candidates"] = resolved
    return result


def validate_source_lock(lock, structure, pseudo_dir):
    """Check the exact confirmed source bytes at each QE node boundary."""
    if not lock:
        return
    for pseudo in lock.get("pseudopotentials", []):
        name = pseudo["filename"]
        if Path(name).name != name:
            raise ValueError("无效赝势文件名")
        if read_pseudo_metadata(Path(pseudo_dir) / name)["sha256"] != pseudo["sha256"]:
            raise ValueError(f"赝势 {name} 已变化，请重新确认方法")
    expected = lock.get("structure_sha256")
    if expected:
        if not structure:
            raise ValueError("已确认的结构文件缺失，请重新确认方法")
        path = Path(structure)
        before = path.stat()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        after = path.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns) or digest != expected:
            raise ValueError("结构文件已变化，请重新确认方法")
