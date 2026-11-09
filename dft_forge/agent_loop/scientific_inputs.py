"""Bind thermodynamic observations from explicit user data, never LLM guesses.

QE setup parameters can have recommended defaults. Reference energies cannot:
they are observations with a particular reference state. The chat boundary
accepts them through structured context or an explicit thermodynamic_inputs
JSON object in the user message. Direct CLI tools keep their numeric API.
"""

from __future__ import annotations

import copy
import json
import math

from dft_forge.agent_loop.planning import mark_blocked


_FIELDS = {
    "workflow.neutral_vacancy": ("chemical_potential_ry", "chemical_potential_ev"),
    "workflow.charged_vacancy": (
        "chemical_potential_ry", "chemical_potential_ev", "dielectric_constant",
        "potential_alignment_ev_by_charge", "electrostatic_correction_ev_by_charge",
        "band_gap_ev",
    ),
    "thermo.chemical_potential_scan": ("formation_energy_ev_per_formula", "competitors"),
}


def _finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _charge_observation(mapping, charge):
    if not isinstance(mapping, dict):
        return None
    for key in (str(charge), f"{charge:+d}", charge):
        if key in mapping:
            return mapping[key]
    return None


def trusted_thermodynamic_inputs(message, ctx):
    """Only parse explicitly named JSON data; prose/model claims are not evidence."""
    trusted = {}
    for value in (ctx.get("thermodynamic_inputs"),):
        if isinstance(value, dict):
            trusted.update(copy.deepcopy(value))
    decoder = json.JSONDecoder()
    for pos, char in enumerate(message or ""):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(message[pos:])
        except ValueError:
            continue
        if isinstance(value, dict) and isinstance(value.get("thermodynamic_inputs"), dict):
            trusted.update(value["thermodynamic_inputs"])
    return trusted


def guard_scientific_inputs(plan, steps, message, ctx):
    """Remove ungrounded actions and bind verified structured input in place.

    This runs again at execution/confirmation time. Private evidence offered
    by the model is discarded; it never authorizes a value by itself.
    """
    trusted = trusted_thermodynamic_inputs(message, ctx)
    kept, blocked_tools = [], set()
    for step in steps:
        tool_id = step.get("tool")
        args = step.get("args") or {}
        args.pop("_input_evidence", None)
        if tool_id not in _FIELDS:
            kept.append(step)
            continue
        supplied = trusted.get(tool_id)
        supplied = supplied if isinstance(supplied, dict) else {}
        if tool_id == "workflow.neutral_vacancy":
            if args.get("reference_mode") == "same_method_bulk_per_atom":
                if not any(args.get(key) is not None for key in _FIELDS[tool_id]):
                    # This is a computed reference strategy, not a guessed
                    # observed number. Runtime validates the actual atoms.
                    kept.append(step)
                    continue
                mark_blocked(plan, "same_method_bulk_per_atom 不能混用显式化学势；请只选择一种参考方式")
                blocked_tools.add(tool_id)
                continue
            present = [key for key in _FIELDS[tool_id] if supplied.get(key) is not None]
            valid = (len(present) == 1 and _finite_number(supplied[present[0]])
                     and supplied.get("reference") == "absolute_dft"
                     and supplied.get("source") == args.get("source")
                     and supplied.get("element") == args.get("element"))
            reason = (
                "中性空位形成能需要与总能量同一参考的绝对 DFT 化学势（每个移除原子）；"
                "O-rich 的 Δμ=0 不能作为绝对 μ。请用 thermodynamic_inputs JSON 提供"
                " chemical_potential_ry 或 chemical_potential_ev，并标明 reference=absolute_dft、source 和 element"
            )
        elif tool_id == "workflow.charged_vacancy":
            charges = args.get("charges") or []
            nonzero = [int(value) for value in charges if isinstance(value, int) and not isinstance(value, bool) and value != 0]
            identity_ok = supplied.get("source") == args.get("source") and supplied.get("element") == args.get("element")
            alignments = supplied.get("potential_alignment_ev_by_charge")
            alignment_ok = bool(nonzero) and all(
                _finite_number(_charge_observation(alignments, charge)) for charge in nonzero
            )
            reference_mode = args.get("reference_mode", "explicit")
            mu_present = [key for key in ("chemical_potential_ry", "chemical_potential_ev") if supplied.get(key) is not None]
            if reference_mode == "same_method_bulk_per_atom":
                mu_ok = not mu_present
            else:
                mu_ok = (len(mu_present) == 1 and _finite_number(supplied[mu_present[0]])
                         and supplied.get("reference") == "absolute_dft")
            correction_mode = args.get("correction_mode", "makov_payne_cubic")
            if correction_mode == "makov_payne_cubic":
                correction_ok = (
                    _finite_number(supplied.get("dielectric_constant"))
                    and supplied["dielectric_constant"] > 0
                )
            else:
                corrections = supplied.get("electrostatic_correction_ev_by_charge")
                correction_ok = bool(nonzero) and all(
                    _finite_number(_charge_observation(corrections, charge)) for charge in nonzero
                )
            gap = supplied.get("band_gap_ev")
            gap_ok = gap is None or (_finite_number(gap) and gap > 0)
            valid = identity_ok and alignment_ok and mu_ok and correction_ok and gap_ok
            reason = (
                "带电空位形成能需要用户结构化提供与 source/element 匹配的绝对化学势"
                "（纯元素 same_method_bulk_per_atom 可省略）、每个非零电荷态的 potential_alignment_ev_by_charge，"
                "以及 Makov-Payne 所需的正 dielectric_constant 或 explicit 模式下的逐电荷静电修正；"
                "这些材料观测量不会由模型猜测。"
            )
        else:
            competitors = supplied.get("competitors", [])
            valid = _finite_number(supplied.get("formation_energy_ev_per_formula")) and isinstance(competitors, list)
            valid = valid and supplied.get("formula") == args.get("formula")
            valid = valid and all(
                isinstance(item, dict) and isinstance(item.get("formula"), str)
                and _finite_number(item.get("formation_energy_ev_per_formula")) for item in competitors
            )
            reason = (
                "化学势窗口需要有来源的父相形成能和所采用的竞争相数据；"
                "请用 thermodynamic_inputs JSON 提供 formation_energy_ev_per_formula"
                "（eV/化学式）和 formula，不能用模型猜测的实验值或占位数补齐"
            )
        for key in _FIELDS[tool_id]:
            args.pop(key, None)
        if valid:
            args.update({key: copy.deepcopy(supplied[key]) for key in _FIELDS[tool_id] if key in supplied})
            args["_input_evidence"] = {"source": "user_structured", "reference": supplied.get("reference")}
            kept.append(step)
        else:
            blocked_tools.add(tool_id)
            mark_blocked(plan, reason)
            if not any(item.get("capability") == tool_id and item.get("description") == reason for item in plan.advisory_steps):
                plan.advisory_steps.append({"capability": tool_id, "description": reason})
    if blocked_tools:
        # Do not continue displaying the rejected guessed numbers as advice.
        plan.reply = "缺少可核对的热力学参考数据；相关计算已转为待补充输入的建议步骤。"
        plan.assumptions = []
        plan.advisory_steps = [
            item for item in plan.advisory_steps
            if item.get("capability") not in blocked_tools
            or item.get("description") in plan.blocking_reasons
        ]
    by_id = {step.get("step_id"): step for step in kept if step.get("step_id")}
    updated = []
    for action in plan.actions:
        tool_id = action.get("tool") or action.get("capability")
        if tool_id in blocked_tools:
            continue
        step = by_id.get(action.get("step_id"))
        if step is not None:
            action["args"] = copy.deepcopy(step["args"])
        updated.append(action)
    plan.actions = updated
    steps[:] = kept
