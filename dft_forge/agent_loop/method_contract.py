"""Product method choices, resolved against the *planned* QE capabilities.

This module is deliberately outside the generic graph runtime. Confirmation
describes requested inputs, not convergence, scientific validity, or values
which the input compiler will choose automatically from a structure.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

from .execution_policy import ExecutionPolicyError

METHOD_KEYS = ("ecutwfc", "ecutrho", "kpoints", "nbnd", "nkpoints_bands")
_MACROS = {"workflow.neutral_vacancy", "workflow.magnetic_compare", "workflow.phonon_dfpt", "workflow.convergence"}
_PRESETS = {
    "quick": {"nkpoints_bands": 60},
    "precise": {"kpoints": [8, 8, 8, 1, 1, 1], "nkpoints_bands": 160},
}
_MACRO_DEFAULTS = {
    "workflow.neutral_vacancy": {"supercell": "2x2x2", "charge": 0, "relax": True, "conv_thr": 1e-8},
    "workflow.magnetic_compare": {"material": "Fe", "orders": ["NM", "FM", "AFM"], "supercell": "1x1x1", "initial_moment_muB": 2.2},
    "workflow.phonon_dfpt": {"material": "MgO", "q_grid": [2, 2, 2], "relax": True, "tr2_ph": 1e-14, "imaginary_threshold_thz": 0.1},
    "workflow.convergence": {"ecutwfc_values": [30, 40, 50, 60], "kpoints_values": [4, 6, 8], "smearing_values": ["mv"], "degauss_values": [0.01], "threshold_mev_per_atom": 1.0, "max_jobs": 64},
}
_DETAIL_KEYS = {
    "material", "structure", "source", "element", "supercell", "index", "charge", "relax",
    "chemical_potential_ry", "chemical_potential_ev", "orders", "magnetic_element", "initial_moment_muB",
    "q_grid", "tr2_ph", "imaginary_threshold_thz", "ecutrho", "conv_thr", "type",
    "ecutwfc_values", "kpoints_values", "smearing_values", "degauss_values", "threshold_mev_per_atom", "max_jobs",
}
_STRUCTURE_BUILDERS = {"structure.generate", "structure.build2d", "structure.dope", "structure.supercell", "structure.vacancy", "structure.molecule", "structure.reference"}


class MethodContractError(ExecutionPolicyError):
    def __init__(self, message, code, contract=None):
        super().__init__(message, code=code)
        self.method_contract = contract
        self.method_token = contract.get("digest") if contract and code == "method_confirmation_required" else None


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _public_args(args):
    return {key: value for key, value in args.items() if not str(key).startswith("_")}


def _file_identity(source, session_dir):
    if not isinstance(source, str) or not source:
        return None
    path = Path(source)
    if not path.is_file():
        return {"reference": source}
    if path.suffix.lower() not in {".cif", ".xyz", ".json", ".pdb", ".vasp", ".poscar", ".in"} and path.name not in {"POSCAR", "CONTCAR"}:
        raise MethodContractError("方法确认不能读取非结构/输入文件，请上传受支持的结构文件", "method_invalid")
    resolved = path.resolve()
    if session_dir is not None and not resolved.is_relative_to(Path(session_dir).resolve()):
        raise MethodContractError("请先将方法所引用的结构/输入文件上传到当前会话", "method_invalid")
    before = path.stat()
    if before.st_size > 16 * 1024 * 1024:
        raise MethodContractError("方法确认的输入文件超过 16 MiB，请使用较小的结构文件", "method_invalid")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(64 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise MethodContractError("结构文件在确认期间发生变化，请重新确认", "method_invalid")
    relative = str(resolved.relative_to(Path(session_dir).resolve())) if session_dir else path.name
    return {"path": relative, "sha256": digest.hexdigest()}


def _execution_selection():
    # Only hashes of addresses/paths leave this function. Do not instantiate
    # an executor (or engine) merely to preview a method.
    keys = ("DFT_FORGE_EXECUTOR", "DFT_FORGE_WALLTIME", "DFT_FORGE_QE_BIN", "PATH",
            "DFT_FORGE_SSH_HOST", "DFT_FORGE_SSH_USER", "DFT_FORGE_SSH_KEY", "DFT_FORGE_SSH_PORT",
            "DFT_FORGE_REMOTE_WORKDIR", "DFT_FORGE_REMOTE_QE_BIN", "DFT_FORGE_SCHEDULER")
    selected = {key: os.environ.get(key, "") for key in keys}
    return {"backend": os.environ.get("DFT_FORGE_EXECUTOR", "local").strip().lower() or "local",
            "walltime_limit_sec": os.environ.get("DFT_FORGE_WALLTIME", "1800"),
            "configuration_digest": _digest(selected)}


def build_method_contract(steps, config, ctx=None, session_dir=None):
    """Pure preflight: no engine, workspace, structure generation or dispatch."""
    if not config or not config.get("mode"):
        return None
    if not any(step.get("tool") in _MACROS | {"graph.run", "input.build", "job.submit"} for step in steps):
        return None
    if config.get("auto_parameters") and config.get("converge_first"):
        calculations = [step for step in steps if step.get("tool") in _MACROS | {"graph.run", "input.build", "job.submit"}]
        if len(calculations) != 1 or calculations[0].get("tool") != "graph.run":
            raise MethodContractError("先检查收敛每次仅支持一个 SCF、能带或 DOS 图任务（最多 9 点）；请将多个计算拆分后分别确认", "method_unsupported")
    requested = {key: copy.deepcopy(config[key]) for key in METHOD_KEYS if config.get(key) is not None}
    auto = _PRESETS.get(config.get("accuracy"), {})
    rows, rejected, templates = [], [], {}
    needs_confirmation = False
    has_input_build = False
    previous_source = (ctx or {}).get("last_structure_source")
    current_structure = _file_identity(previous_source, session_dir) if previous_source and Path(str(previous_source)).is_file() else None
    built_structure = None
    file_inputs = []
    for index, step in enumerate(steps):
        tool = step.get("tool")
        args = _public_args(step.get("args") or {})
        for key in ("source", "structure"):
            if args.get(key):
                file_inputs.append({"index": index, "field": key, "identity": _file_identity(args[key], session_dir)})
        if tool in _STRUCTURE_BUILDERS:
            current_structure = {"from_step": index, "output": "structure"}
            built_structure = current_structure
        elif tool == "structure.import":
            current_structure = _file_identity(args.get("source"), session_dir)
        if tool not in _MACROS | {"graph.run", "input.build", "job.submit"}:
            continue
        # A legacy job has no editable input method of its own. Its preceding
        # input.build, when present, is the place where bindings are applied.
        if tool == "job.submit" and has_input_build:
            continue
        if tool == "input.build":
            has_input_build = True
        params = dict(args.get("inputs") or {}) if tool == "graph.run" else args
        template_id = args.get("template_id") if tool == "graph.run" else None
        defaults = {}
        has_qe_nodes = False
        if tool == "graph.run":
            from dft_forge.tools.graph_tools import DEFAULT_TEMPLATES
            from dft_forge.runtime.graph import load_named_template, _iter_param_strings

            try:
                loaded = load_named_template(DEFAULT_TEMPLATES, template_id)
                errors = loaded.validate()
                if errors:
                    raise ValueError("; ".join(errors))
            except (OSError, TypeError, ValueError, KeyError) as exc:
                raise MethodContractError("无法确认该图模板，请选择已登记且可读取的模板", "method_invalid") from exc
            template = loaded.to_dict()
            defaults = template.get("inputs", {})
            referenced = {ref for node in loaded.nodes for ref in _iter_param_strings(node.params)}
            supported = {key for key in METHOD_KEYS if key in defaults and "${inputs." + key + "}" in referenced}
            templates[str(index)] = template
            has_qe_nodes = any(node.tool == "qe" for node in loaded.nodes)
            if config.get("auto_parameters") and has_qe_nodes:
                if any(node.tool == "qe" and (node.params.get("_parameters_locked") != "${inputs._parameters_locked}"
                         or node.params.get("_source_lock") != "${inputs._source_lock}") for node in loaded.nodes):
                    raise MethodContractError("此模板尚未接入自动方法的参数与来源锁定，请补充模板绑定或使用材料默认接口", "method_unsupported")
        elif tool in _MACROS - {"workflow.convergence"} or tool == "input.build":
            supported = {"ecutwfc", "ecutrho", "kpoints"}
            defaults = {"ecutwfc": None, "kpoints": None}
        else:
            supported = set()

        effective = {key: copy.deepcopy(value) for key, value in _MACRO_DEFAULTS.get(tool, {}).items()}
        effective.update({key: copy.deepcopy(value) for key, value in defaults.items() if key in METHOD_KEYS})
        effective.update({key: copy.deepcopy(value) for key, value in params.items() if key in _DETAIL_KEYS or key in supported})
        applied, ignored, conflicts = {}, {}, {}
        for key, value in requested.items():
            if key in supported:
                applied[key] = copy.deepcopy(value)
                effective[key] = copy.deepcopy(value)
            elif tool == "workflow.convergence" and key in {"ecutwfc", "kpoints"}:
                conflicts[key] = {"requested": value, "reason": "单值不能覆盖收敛扫描；确认后使用下方完整扫描范围"}
            elif config.get("accuracy") != "custom" and key in auto and value == auto[key]:
                ignored[key] = {"requested": value, "reason": "此预置参数不适用于该步骤"}
            else:
                rejected.append({"index": index, "tool": tool, "parameter": key, "requested": value})

        # Planner-supplied fields must not quietly disappear either.
        for key in METHOD_KEYS:
            if params.get(key) is None or key in supported:
                continue
            if tool == "workflow.convergence" and key in {"ecutwfc", "kpoints"}:
                conflicts[key] = {"requested": params[key], "reason": "规划的单值会被扫描网格覆盖；确认完整扫描范围"}
            else:
                rejected.append({"index": index, "tool": tool, "parameter": key, "requested": params[key]})

        notes = []
        if config.get("accuracy") == "quick" and not applied and not config.get("auto_parameters"):
            notes.append("快速预置没有适用于此步骤的覆盖，仍使用规划/工作流默认方法，不承诺加速")
        if tool == "workflow.convergence":
            from dft_forge.workflows import convergence_grid

            try:
                effective["scan_points"] = convergence_grid(**{key: effective[key] for key in (
                    "ecutwfc_values", "kpoints_values", "smearing_values", "degauss_values", "max_jobs",
                )})
            except (TypeError, ValueError, OverflowError) as exc:
                raise MethodContractError(f"收敛扫描参数无效：{exc}", "method_invalid") from exc
            effective["job_count"] = len(effective["scan_points"])
        automatic = sorted(key for key in supported if effective.get(key) is None)
        if automatic:
            notes.append("自动字段将在输入编译时按结构/材料选择；此处不是已解析的数值，也不代表已收敛")
        if tool == "graph.run" and template_id == "t2_bands" and "nkpoints_bands" not in params and "nkpoints_bands" not in applied:
            # Existing preparation may adapt density to a generated supercell.
            effective["nkpoints_bands"] = None
            automatic.append("nkpoints_bands")
            notes.append("能带路径默认 100 点，生成/上传的大结构可自动降至 60 或 40 点；最终值见任务 inputs")
        needs_confirmation |= bool(tool in _MACROS or ignored or conflicts or notes and config.get("accuracy") == "quick")
        needs_confirmation |= any(params.get(key) is not None and params[key] != requested.get(key) for key in METHOD_KEYS)
        row = {
            "index": index, "tool": tool, "template_id": template_id,
            "effective": effective, "applied": applied, "not_applicable": ignored,
            "conflicts": conflicts, "automatic_defaults": automatic, "notes": notes,
        }
        if tool == "graph.run":
            row["structure_input"] = _file_identity(params.get("structure"), session_dir) or current_structure or {"material": params.get("material", defaults.get("material"))}
            if not params.get("structure") and current_structure:
                row["notes"].append("使用前置生成或当前会话结构，覆盖规划中的材料名称")
                needs_confirmation = True
        elif tool == "input.build":
            row["structure_input"] = _file_identity(params.get("structure"), session_dir) or built_structure or {"material": params.get("material")}
            if not params.get("structure") and built_structure:
                needs_confirmation = True
        elif tool == "job.submit":
            row["input_file"] = _file_identity(str(Path(session_dir) / "input.in") if session_dir else args.get("input"), session_dir)
            row["notes"].append("方法来自已有输入文件；方法卡不会改写此文件")
            needs_confirmation = True
        if str(index) in templates:
            row["template_digest"] = _digest(templates[str(index)])
        if config.get("auto_parameters") and (tool != "graph.run" or has_qe_nodes):
            from dft_forge.compiler.qe_parameters import resolve_qe_parameters, convergence_proposal

            needs_confirmation = True
            identity = row.get("structure_input") or {}
            deferred = identity.get("from_step") is not None
            # Macros that build/relax several distinct structures have their
            # own scientific policies. Do not resolve them as the default Si.
            if tool in _MACROS or tool == "job.submit":
                if tool == "workflow.convergence":
                    raise MethodContractError("自动定参模式请使用单个 SCF、能带或 DOS 任务并勾选「先检查收敛」；旧独立扫描不提供本模式的固定参数与内部平台保证", "method_unsupported")
                row["recommendation"] = {
                    "status": "deferred", "validation": "not_tested",
                    "parameters": {}, "reasons": {}, "pseudopotentials": [],
                    "notes": ["此复合工作流按各子步骤结构解析参数；上方列出已确认的工作流策略"],
                }
                if tool in _MACROS:
                    planner_fields = [key for key in METHOD_KEYS if params.get(key) is not None and key not in requested]
                    row["strip_planner_parameters"] = planner_fields
                    for key in planner_fields:
                        row["effective"][key] = defaults.get(key)
                        row["conflicts"][key] = {
                            "requested": copy.deepcopy(params[key]),
                            "source": "planner",
                            "reason": "规划候选值未获用户明确覆盖授权；执行前移除并由复合工作流按子步骤结构解析",
                        }
                    if planner_fields:
                        row["automatic_defaults"] = sorted(set(row["automatic_defaults"]) | set(planner_fields))
                        row["notes"].append("未明确自定义的规划数值不会直接执行；这些字段由工作流按子步骤结构解析")
                if config.get("converge_first"):
                    raise MethodContractError("先检查收敛目前适用于单个 SCF、能带或 DOS 图任务；复合任务请单独规划收敛扫描", "method_unsupported")
            elif deferred:
                row["recommendation"] = {
                    "status": "deferred", "validation": "not_tested",
                    "parameters": {}, "reasons": {}, "pseudopotentials": [],
                    "notes": [f"结构将由第 {identity['from_step'] + 1} 步生成；确认自动解析策略，生成后按实际结构定参并留存证据"],
                }
                row["auto_parameter_policy"] = {"accuracy": config.get("accuracy", "balanced"), "overrides": requested, "supported": sorted(supported), "task": template_id or params.get("type", "scf")}
                if config.get("converge_first"):
                    raise MethodContractError("请先生成结构后再选择先检查收敛，以便预览准确的扫描范围", "method_unsupported")
            else:
                source = params.get("structure") or previous_source
                # structure.import changes current_structure to a file; use
                # that exact file even if the planner also supplied material.
                if identity.get("path") and session_dir:
                    source = str(Path(session_dir) / identity["path"])
                chosen_material = params.get("material") or defaults.get("material")
                if tool == "input.build":
                    chosen_material = params.get("material")
                try:
                    recommendation = resolve_qe_parameters(
                        material=chosen_material, structure=source,
                        accuracy=config.get("accuracy", "balanced"),
                        overrides={k: v for k, v in requested.items() if k in supported},
                        task=template_id or params.get("type", "scf"),
                    )
                except (ValueError, OSError, KeyError, TypeError) as exc:
                    raise MethodContractError(f"无法生成计算方法：{exc}", "method_invalid") from exc
                resolved = {k: v for k, v in recommendation["parameters"].items() if k in supported}
                row["effective"].update(resolved)
                row["applied"].update(resolved)
                row["automatic_defaults"] = []
                row["notes"] = [n for n in row["notes"] if "编译时" not in n and "能带路径默认" not in n]
                for key in METHOD_KEYS:
                    if params.get(key) is not None and key not in requested and key in resolved and params[key] != resolved[key]:
                        recommendation["notes"].append(f"规划中的 {key} 候选值由结构/赝势规则重新解析；可在自定义方法中明确覆盖")
                row["recommendation"] = recommendation
                if config.get("converge_first"):
                    if tool != "graph.run" or template_id not in {"t0_scf", "t2_bands", "t2_dos"}:
                        raise MethodContractError("先检查收敛目前适用于 SCF、能带和 DOS；结构优化请先单独规划收敛扫描", "method_unsupported")
                    scan = convergence_proposal(recommendation)
                    required_rho = max(recommendation["parameters"]["ecutrho"], max(scan["ecutwfc_values"]) * scan["ecutrho_ratio"])
                    if "ecutrho" in requested and requested["ecutrho"] < required_rho:
                        raise MethodContractError("自定义 ecutrho 小于扫描上界所需值；请清空该覆盖或提高后重新预览", "method_invalid")
                    scan["fixed_ecutrho"] = max(required_rho, requested.get("ecutrho", 0))
                    scan["selection_policy"] = "interior_plateau_v1"
                    row["convergence_plan"] = scan
                    recommendation["convergence"] = scan
                    row["notes"].append("先扫描总能内部稳定性；下游使用选中点及固定密度截断能。失败或未形成平台时不运行下游，不证明带隙收敛")
                    row["effective"]["ecutrho"] = scan["fixed_ecutrho"]
                    row["applied"]["ecutrho"] = scan["fixed_ecutrho"]
                row["parameter_repair"] = "require_new_confirmation"
        elif config.get("auto_parameters") and config.get("converge_first"):
            raise MethodContractError("先检查收敛目前只支持 QE 的 SCF、能带或 DOS 图任务", "method_unsupported")
        rows.append(row)
    if not rows:
        return None
    document = {
        "schema": "dft-forge/method-contract-v1", "mode": config["mode"],
        "accuracy": config.get("accuracy", "balanced"), "requested": requested,
        "steps": rows, "requires_confirmation": bool(needs_confirmation), "rejected": rejected,
        "scope": "requested_inputs_not_scientific_validation",
        "automatic_policy": {"resolver": "qe-input-compiler-v1", "band_path": "structure-size-adaptive-v1"},
        "execution_selection": _execution_selection(),
        "file_inputs": file_inputs,
    }
    if config.get("auto_parameters"):
        document["auto_parameters"] = True
        document["converge_first"] = bool(config.get("converge_first"))
        document["planner"] = (ctx or {}).get("last_plan", {}).get("source", "rules")
        document["automatic_policy"]["resolver"] = "qe-parameters-v1"
    # Descriptions can change between LLM plans without changing semantics;
    # tools, arguments and dependencies cannot. Raw plans are hashed, not
    # copied into the portable method record.
    plan = [{"tool": s.get("tool"), "args": _public_args(s.get("args") or {}), "depends_on": s.get("depends_on", [])} for s in steps]
    document["digest"] = _digest({"contract": document, "plan": plan, "templates": templates,
                                  "session_id": (ctx or {}).get("session_id"),
                                  "structure": (ctx or {}).get("last_structure_source")})
    if rejected:
        names = "、".join(f"{item['tool']}.{item['parameter']}" for item in rejected)
        raise MethodContractError(f"所选工作流暂不支持这些显式方法参数：{names}；请清空对应字段或选择支持的任务", "method_unsupported", document)
    return document


def check_method_contract(steps, ctx, session_dir=None):
    # Never accept a stale or planner-originated contract as execution input.
    ctx.pop("effective_method_contract", None)
    try:
        contract = build_method_contract(steps, ctx.get("run_config"), ctx, session_dir)
    except MethodContractError:
        raise
    except (TypeError, ValueError, OSError) as exc:
        raise MethodContractError("无法确认计划中的方法或输入文件，请检查参数并重新规划", "method_invalid") from exc
    if contract is None:
        return None
    supplied = ctx.get("requested_method_token")
    if (contract["requires_confirmation"] or supplied) and supplied != contract["digest"]:
        raise MethodContractError("规划已生成实际方法，请核对逐步骤参数后确认执行；当前尚未运行任何工具", "method_confirmation_required", contract)
    contract["confirmation"] = "effective_contract" if supplied else "general_method_card"
    return contract


def method_for_step(contract, index):
    if not isinstance(contract, dict):
        return None
    row = next((item for item in contract.get("steps", []) if item.get("index") == index), None)
    if row is None:
        return None
    return {key: copy.deepcopy(contract[key]) for key in ("schema", "digest", "mode", "accuracy", "requested", "confirmation", "scope", "automatic_policy", "execution_selection")} | {"step": copy.deepcopy(row)}
