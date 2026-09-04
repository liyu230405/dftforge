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
            errors.append(f"step {i} ({tool_id}): {loc} {err.message}")
    return errors
