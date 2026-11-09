"""Validate LLM-planned steps against each tool's registered input schema.

Schemas come from the tool registry itself (single source of truth) — the
catalog the LLM planner sees is generated from the same objects, so validation
can never drift from what the prompt advertises.
"""

from __future__ import annotations

import jsonschema


def validate_steps(steps: list, registry) -> list[str]:
    """Return human-readable arg-schema violations (empty list = valid).

    Unknown tool ids are ignored here — the planner already rejects them
    before validation runs. Missing schema likewise skips the step.
    """
    errors: list[str] = []
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            errors.append(f"step {i}: not an object")
            continue
        tool_id = step.get("tool")
        tool = registry.get(tool_id) if tool_id else None
        if tool is None:
            continue
        schema = tool.input_schema if isinstance(tool.input_schema, dict) else None
        if not schema:
            continue
        args = step.get("args")
        if args is None:
            args = {}
        if not isinstance(args, dict):
            errors.append(f"step {i} ({tool_id}): args must be an object, got {type(args).__name__}")
            continue
        validator = jsonschema.Draft7Validator(schema)
        for err in sorted(validator.iter_errors(args), key=lambda e: list(e.path)):
            loc = ".".join(str(p) for p in err.path) or "args"
            # Composite analysis tools receive numeric values from named
            # upstream outputs during step preparation.  Their runtime
            # schema remains strict; the planner schema is allowed to omit
            # only the explicitly declared deferred inputs.
            if (
                err.validator == "required"
                and tool.deferred_inputs
                and set(err.validator_value) - set(args)
                and set(err.validator_value) - set(args) <= set(tool.deferred_inputs)
            ):
                continue
            errors.append(f"step {i} ({tool_id}): {loc} {err.message}")
    return errors


def semantic_plan_errors(message: str, steps: list) -> list[str]:
    """Catch category mistakes that JSON schema cannot express.

    This is intentionally small and deterministic.  The LLM may describe a
    scientific operation, but only a tool whose declared semantics match that
    operation may be compiled for execution.
    """
    import re
    from dft_forge.agent_loop.helpers import extract_formula

    text = str(message or "").lower()
    errors: list[str] = []
    step_ids = [str(step.get("step_id") or f"step_{i + 1}") if isinstance(step, dict) else None
                for i, step in enumerate(steps)]
    for step_index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        tool = str(step.get("tool") or "")
        args = step.get("args") or {}
        desc = str(step.get("description") or "").lower()
        if tool == "thermo.eads" and args.get("energy_steps") is not None:
            refs = args["energy_steps"]
            if (not isinstance(refs, dict) or set(refs) != {"adsorbed", "surface", "molecule"}
                    or not all(isinstance(value, str) for value in refs.values())
                    or len(set(refs.values())) != 3):
                errors.append("thermo.eads.energy_steps 必须绑定 adsorbed/surface/molecule 三个不同步骤")
            else:
                for ref in refs.values():
                    if (step_ids.count(ref) != 1 or step_ids.index(ref) >= step_index
                            or steps[step_ids.index(ref)].get("tool") != "graph.run"):
                        errors.append(f"thermo.eads 引用 {ref} 必须是唯一且在之前的 graph.run 步骤")
        if tool == "structure.dope" and (
            any(k in text for k in ("空位", "vacancy", "缺陷"))
            or any(k in desc for k in ("空位", "vacancy", "remove", "删除"))
        ):
            errors.append("structure.dope 只做元素替换，不能移除原子生成空位")
        if tool == "structure.dope" and any(
            k in desc for k in ("纯对照", "pristine", "undoped", "无实际掺杂")
        ):
            errors.append("纯相对照必须使用原结构/超胞复制，不能用同元素替换伪造")
        if tool == "structure.dope":
            source = str(args.get("source") or "")
            dopant = str(args.get("element") or "").strip().capitalize()
            source_formula = extract_formula(source)
            # Also recognize a planner's not-yet-created name such as
            # ``si_2x2x2.cif``.  A one-element source plus the same dopant is
            # never a real transformation, regardless of the filename.
            if source_formula and source_formula.rstrip("1") == dopant and any(
                k in desc for k in ("纯", "pristine", "对照", "reference")
            ):
                errors.append("纯相对照不能通过同元素 structure.dope 构建，应使用 structure.supercell")
        if tool == "thermo.formation" and any(k in text for k in ("空位", "vacancy")):
            # A defect study may legitimately compute the parent compound's
            # formation energy first to bound relative chemical potentials.
            parent_reference = any(k in desc for k in ("母相", "母体", "parent", "化学势", "δμ", "delta_mu"))
            defect_reduction = any(k in desc for k in ("空位形成", "缺陷形成", "vacancy formation", "defect formation"))
            if defect_reduction or not parent_reference:
                errors.append("thermo.formation 只适用于化合物相对元素参考态，不是缺陷形成能工具")
        if tool == "structure.build2d" and isinstance(args.get("adsorb"), str):
            errors.append("structure.build2d.adsorb 必须是含 element/site 的对象")
        if tool == "workflow.neutral_vacancy" and re.search(
            r"(?:空位|vacancy|缺陷)[^\n]{0,24}(?:q\s*=\s*)?[+-]\s*[1-9]|"
            r"[+-]\s*[1-9][^\n]{0,24}(?:空位|vacancy|缺陷)",
            text,
        ):
            errors.append("显式非零电荷空位必须使用 workflow.charged_vacancy，不能降级为中性空位")

    # Multi-site adsorption is a composite contract.  A plan that does not
    # contain the final energy reduction is incomplete even if all structures
    # were built successfully.
    requested_sites = {
        site for site in ("top", "bridge", "hollow") if site in text
    }
    if any(k in text for k in ("桥", "洞", "三个位点")):
        if "桥" in text:
            requested_sites.add("bridge")
        if "洞" in text:
            requested_sites.add("hollow")
        if "三个位点" in text:
            requested_sites.update(("top", "bridge", "hollow"))
    if any(k in text for k in ("吸附能", "adsorption energy", "e_ads", "eads")) and len(requested_sites) >= 2:
        macros = [s for s in steps if isinstance(s, dict) and s.get("tool") == "workflow.adsorption_scan"]
        if macros:
            planned = macros[0].get("args", {}).get("sites") or []
            planned_sites = {
                str(item.get("site")) if isinstance(item, dict) else str(item)
                for item in planned
            }
            missing = requested_sites - planned_sites
            if missing:
                errors.append("workflow.adsorption_scan 缺少请求位点: " + ", ".join(sorted(missing)))
        else:
            n_eads = sum(str(s.get("tool")) == "thermo.eads" for s in steps if isinstance(s, dict))
            if n_eads < len(requested_sites):
                errors.append(
                    f"多位点吸附 workflow 需要一个覆盖全部位点的 workflow.adsorption_scan；"
                    f"旧展开计划只有 {n_eads} 个 thermo.eads 汇总"
                )
    return list(dict.fromkeys(errors))
