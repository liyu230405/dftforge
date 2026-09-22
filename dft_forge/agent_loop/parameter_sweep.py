"""Execute a confirmed, bounded SCF scan before the requested target graph."""

from __future__ import annotations

import copy
import hashlib
import math
from numbers import Integral, Real
from pathlib import Path
import uuid

from dft_forge.compiler import (
    DEFAULT_PSEUDO_DIR, MATERIAL_DB, build_atoms, formula_atoms,
    profile_from_atoms, read_structure,
)
from dft_forge.parser import RY_TO_EV
from dft_forge.runtime.graph import GraphTemplate, NodeSpec
from dft_forge.runtime.run import ToolError


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"收敛扫描 {name} 必须为有限正数")
    return float(value)


def _validated_grid(plan, pbc):
    if plan.get("selection_policy") != "interior_plateau_v1":
        raise ValueError("不支持的收敛选择策略，请重新确认方法")
    cutoffs = [_positive(value, "ecutwfc") for value in plan.get("ecutwfc_values", [])]
    if len(cutoffs) < 2 or any(a >= b for a, b in zip(cutoffs, cutoffs[1:])):
        raise ValueError("收敛扫描至少需要两个严格递增的截断能")
    grids = plan.get("kpoints_values") or []
    if not isinstance(grids, list) or not grids:
        raise ValueError("收敛扫描需要非空 k 网格")
    for grid in grids:
        if not isinstance(grid, (list, tuple)) or len(grid) != 6:
            raise ValueError("收敛扫描 k 网格需要三个正整数及三个偏移")
        if any(isinstance(n, bool) or not isinstance(n, Integral) or n < 1 for n in grid[:3]):
            raise ValueError("收敛扫描 k 网格需要严格正整数")
        if any(type(n) is not int or n not in (0, 1) for n in grid[3:]):
            raise ValueError("收敛扫描 k 网格偏移必须为 0 或 1")
        if any(not periodic and (grid[i] != 1 or grid[i + 3] != 0) for i, periodic in enumerate(pbc)):
            raise ValueError("非周期方向只能使用无偏移的单 k 点")
    if any(a[3:] != b[3:] or any(x > y for x, y in zip(a[:3], b[:3])) or a[:3] == b[:3]
           for a, b in zip(grids, grids[1:])):
        raise ValueError("收敛扫描 k 网格必须保持偏移并逐层加密")
    if len(grids) == 1 and any(pbc):
        raise ValueError("周期结构必须扫描至少两个 k 网格")
    count = len(cutoffs) * len(grids)
    if type(plan.get("max_jobs")) is not int or not 1 <= plan["max_jobs"] <= 9:
        raise ValueError("自动收敛扫描最多允许 9 个任务")
    if type(plan.get("job_count")) is not int or plan["job_count"] != count or count > plan["max_jobs"]:
        raise ValueError("收敛扫描任务数与已确认范围不一致")
    rho = _positive(plan.get("fixed_ecutrho"), "固定 ecutrho")
    if rho < 4 * max(cutoffs):
        raise ValueError("固定 ecutrho 不足以覆盖扫描截断能上界")
    threshold = _positive(plan.get("threshold_mev_per_atom"), "threshold_mev_per_atom")
    if len(plan.get("smearing_values", [])) != 1 or len(plan.get("degauss_values", [])) != 1:
        raise ValueError("自动收敛扫描必须固定展宽方法和展宽宽度")
    if not isinstance(plan.get("occupations"), str) or not plan["occupations"]:
        raise ValueError("收敛扫描需要明确 occupations")
    smearing = plan["smearing_values"][0]
    if not isinstance(smearing, str) or not smearing:
        raise ValueError("收敛扫描需要明确 smearing")
    degauss = _positive(plan["degauss_values"][0], "degauss")
    return cutoffs, [list(grid) for grid in grids], rho, threshold, smearing, degauss


def _source_guard(inputs, step, pseudo_dir):
    """Bind every point to the confirmed structure/profile and pseudo bytes."""
    recommendation = step.get("recommendation") or {}
    expected_pseudos = recommendation.get("pseudopotentials") or []
    structure = inputs.get("structure")
    material = inputs.get("material")
    if structure:
        atoms = read_structure(structure)
        profile = profile_from_atoms(atoms, pseudo_dir)
    elif material in MATERIAL_DB:
        atoms, profile = build_atoms(material), MATERIAL_DB[material]
    elif material:
        atoms, _ = formula_atoms(material)
        profile = profile_from_atoms(atoms, pseudo_dir)
    else:
        raise ValueError("收敛扫描需要明确结构或材料")
    natoms = recommendation.get("structure", {}).get("natoms")
    if type(natoms) is not int or natoms <= 0 or len(atoms) != natoms:
        raise ValueError("结构原子数与已确认推荐不一致")
    pbc = [bool(x) for x in atoms.pbc]
    if recommendation.get("structure", {}).get("pbc") != pbc:
        raise ValueError("结构周期性与已确认推荐不一致")
    declared = {row.get("element"): row.get("filename") for row in expected_pseudos}
    if len(declared) != len(expected_pseudos) or declared != profile["pseudos"]:
        raise ValueError("赝势选择与已确认推荐不一致")
    identities = []
    for row in expected_pseudos:
        name = row["filename"]
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("无效的赝势文件名")
        identities.append((Path(pseudo_dir) / name, row.get("sha256")))
    if structure:
        identity = step.get("structure_input") or {}
        if not identity.get("sha256"):
            raise ValueError("收敛扫描缺少已确认结构的文件哈希")
        identities.append((Path(structure), identity["sha256"]))

    def check():
        for path, digest in identities:
            before = path.stat()
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            after = path.stat()
            if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns) or actual != digest:
                raise ValueError(f"结构或赝势文件已变化（{path.name}），请重新确认方法")

    check()
    return check, natoms, pbc


def _checked_output(output, natoms):
    energy, count = output.get("energy_ry"), output.get("natoms")
    if isinstance(energy, bool) or not isinstance(energy, Real) or not math.isfinite(energy):
        raise ValueError("SCF 点缺少有限总能量")
    if isinstance(count, bool) or not isinstance(count, Integral) or count != natoms:
        raise ValueError("SCF 点原子数与锁定结构不一致")
    return float(energy), int(count)


class _CheckedTool:
    name = "qe"

    def __init__(self, wrapped, check_sources, natoms):
        self.wrapped, self.check_sources, self.natoms = wrapped, check_sources, natoms
        self.executor = getattr(wrapped, "executor", None)

    def execute(self, node, context):
        try:
            self.check_sources()
            output = self.wrapped.execute(node, context)
            self.check_sources()
            _checked_output(output, self.natoms)
            return output
        except (ValueError, OSError) as exc:
            raise ToolError(str(exc), category="convergence_evidence", repairable=False) from exc


def _select_plateau(points, cutoffs, grids, threshold):
    reference = points[-1]
    ref_energy = reference["energy_ry"] / reference["natoms"]
    for point in points:
        point["delta_mev_per_atom"] = abs(point["energy_ry"] / point["natoms"] - ref_energy) * RY_TO_EV * 1000
        point["cost_proxy"] = point["ecutwfc"] * math.prod(point["kpoints"][:3])
    eligible = []
    # The final row/column cannot certify itself. Require another cutoff
    # and, for periodic structures, another k grid above each candidate.
    for ci in range(len(cutoffs) - 1):
        for ki in range(len(grids) - 1 if len(grids) > 1 else 1):
            tail = [point for point in points if point["cutoff_index"] >= ci and point["grid_index"] >= ki]
            if len(tail) > 1 and all(point["delta_mev_per_atom"] <= threshold for point in tail):
                eligible.append(points[ci * len(grids) + ki])
    if not eligible:
        raise ValueError("已确认扫描范围内未形成内部总能平台；请扩大扫描范围后重新确认，下游尚未运行")
    selected = min(eligible, key=lambda row: (row["cost_proxy"], row["ecutwfc"], row["grid_index"]))
    return copy.deepcopy(selected), copy.deepcopy(reference)


def run_parameter_sweep(inputs, method_contract, workdir, cancel_event=None, on_event=None):
    """Return target inputs and method evidence after a successful bounded scan.

    Any failed/cancelled point, changed source, invalid energy or absence of
    an interior plateau raises ``ValueError``; callers must not run the target
    graph in that case. The original recommendation and policy stay intact.
    """
    from dft_forge.tools.graph_tools import DEFAULT_WORKSPACE, _get_engine

    new_inputs, new_method = copy.deepcopy(inputs), copy.deepcopy(method_contract)
    step = (new_method or {}).get("step") or {}
    plan = step.get("convergence_plan")
    if not plan:
        return new_inputs, new_method
    if cancel_event is not None and cancel_event.is_set():
        raise ValueError("收敛扫描已取消，下游尚未运行")
    target_workspace = Path(workdir) if workdir is not None else DEFAULT_WORKSPACE
    workspace = target_workspace.parent / "workflows" / f"parameter_sweep_{uuid.uuid4().hex[:12]}" / "graphs"
    engine = _get_engine(str(workspace))
    original_tool = engine.tools.get("qe")
    if original_tool is None:
        raise ValueError("收敛扫描没有可用的 QE 执行工具")
    execution = engine.execution_profile or {}
    simulated = execution.get("simulated") is True or execution.get("backend") == "fake"
    if new_method.get("mode") == "research" and simulated:
        raise ValueError("科研模式不能使用模拟收敛结果")
    pseudo_dir = getattr(getattr(original_tool, "compiler", None), "pseudo_dir", DEFAULT_PSEUDO_DIR)
    try:
        check_sources, natoms, pbc = _source_guard(new_inputs, step, pseudo_dir)
    except OSError as exc:
        raise ValueError("收敛扫描无法读取已确认的结构或赝势文件") from exc
    cutoffs, grids, rho, threshold, smearing, degauss = _validated_grid(plan, pbc)
    recommendation = step["recommendation"]
    expected_nbnd = step.get("effective", {}).get("nbnd", recommendation.get("parameters", {}).get("nbnd"))
    nbnd = new_inputs.get("nbnd") if new_inputs.get("nbnd") is not None else expected_nbnd
    if isinstance(nbnd, bool) or not isinstance(nbnd, Integral) or nbnd <= 0:
        raise ValueError("收敛扫描必须固定已确认的能带总数")
    if nbnd != expected_nbnd:
        raise ValueError("能带总数与已确认方法不一致，请重新确认")
    locked = {key: copy.deepcopy(new_inputs[key]) for key in (
        "material", "structure", "conv_thr", "spin_polarized", "magnetic_order",
        "magnetic_element", "initial_moment_muB", "tot_charge",
    ) if new_inputs.get(key) is not None}
    locked.update(calc="scf", nbnd=int(nbnd), ecutrho=rho,
                  occupations=plan["occupations"], smearing=smearing, degauss=degauss)
    nodes, points = [], []
    for ci, cutoff in enumerate(cutoffs):
        for ki, grid in enumerate(grids):
            node_id = f"scan_c{ci}_k{ki}"
            params = {**copy.deepcopy(locked), "prefix": node_id, "ecutwfc": cutoff, "kpoints": grid}
            nodes.append(NodeSpec(id=node_id, tool="qe", params=params, max_attempts=1,
                                  max_repair_attempts=0, repair=""))
            points.append({"node_id": node_id, "cutoff_index": ci, "grid_index": ki,
                           "ecutwfc": cutoff, "kpoints": grid, "ecutrho": rho,
                           "occupations": plan["occupations"], "smearing": smearing,
                           "degauss": degauss, "nbnd": int(nbnd)})
    template = GraphTemplate(template_id="parameter_sweep", description="Confirmed bounded SCF parameter sweep", nodes=nodes)
    engine.register_template(template)
    run = engine.create(template.template_id, run_metadata={
        "method_contract": copy.deepcopy(new_method), "workflow_type": "parameter_sweep",
    })

    def emit(event):
        if on_event:
            copied = {**event, "phase": "convergence", "run_id": run.run_id}
            if isinstance(copied.get("node"), str) and not copied["node"].startswith("scan_"):
                copied["node"] = "scan_" + copied["node"]
            on_event(copied)

    engine.tools["qe"] = _CheckedTool(original_tool, check_sources, natoms)
    try:
        run = engine.start(run.run_id, on_event=emit, cancel_event=cancel_event)
    except Exception as exc:
        raise ValueError("收敛扫描执行未完成，下游尚未运行") from exc
    finally:
        engine.tools["qe"] = original_tool
    evidence = {
        "run_id": run.run_id, "state": run.state.value, "selected": None, "reference": None,
        "threshold_mev_per_atom": threshold, "selection_policy": plan["selection_policy"],
        "validation": "not_validated", "simulated": simulated,
        "scope": "bounded_total_energy_plateau_not_band_gap_validation",
        "points": points,
    }
    try:
        for point in points:
            node = run.nodes[point["node_id"]]
            point.update(state=node.state.value, error=node.error)
            if node.state.value == "succeeded":
                point["energy_ry"], point["natoms"] = _checked_output(node.outputs, natoms)
        if cancel_event is not None and cancel_event.is_set():
            raise ValueError("收敛扫描已取消，下游尚未运行")
        if run.state.value != "succeeded" or any(point["state"] != "succeeded" for point in points):
            raise ValueError("收敛扫描存在失败或未完成的 SCF 点，下游尚未运行")
        check_sources()
        evidence["reference"] = copy.deepcopy(points[-1])
        selected, reference = _select_plateau(points, cutoffs, grids, threshold)
        evidence.update(selected=selected, reference=reference)
        evidence["validation"] = ("simulated_bounded_energy_plateau" if simulated else
                                  "bounded_energy_plateau" if execution.get("simulated") is False and execution.get("backend") in {"local", "ssh"}
                                  else "unknown_source_bounded_energy_plateau")
        changes = {"ecutwfc": selected["ecutwfc"], "kpoints": selected["kpoints"], "ecutrho": rho}
        new_inputs.update(copy.deepcopy(changes))
        step.setdefault("effective", {}).update(copy.deepcopy(changes))
        step.setdefault("applied", {}).update(copy.deepcopy(changes))
        step["convergence_evidence"] = evidence
    except (ValueError, OSError) as exc:
        evidence["error"] = str(exc)
        run.metadata["convergence_evidence"] = evidence
        engine.store.save_run(run)
        raise ValueError(str(exc)) from exc
    run.metadata["convergence_evidence"] = copy.deepcopy(evidence)
    run.metadata["method_contract"] = copy.deepcopy(new_method)
    engine.store.save_run(run)
    return new_inputs, new_method
