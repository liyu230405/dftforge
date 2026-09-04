"""Tool registry for DFT-Forge."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

import jsonschema

from .models import ToolEntry, ToolResult, validate_tool_id

logger = logging.getLogger(__name__)


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, ToolEntry] = {}

    def register(self, tool: ToolEntry) -> None:
        if not validate_tool_id(tool.id):
            raise ValueError(f"Invalid tool id: {tool.id!r}")
        self._tools[tool.id] = tool
        logger.info("Registered tool: %s (%s)", tool.id, tool.category)

    def unregister(self, tool_id: str) -> None:
        self._tools.pop(tool_id, None)

    def get(self, tool_id: str) -> Optional[ToolEntry]:
        return self._tools.get(tool_id)

    def list_all(self) -> list[ToolEntry]:
        return list(self._tools.values())

    def list_by_category(self, category: str) -> list[ToolEntry]:
        return [t for t in self._tools.values() if t.category == category]

    def enable(self, tool_id: str) -> None:
        tool = self._tools.get(tool_id)
        if tool:
            tool.enabled = True

    def disable(self, tool_id: str) -> None:
        tool = self._tools.get(tool_id)
        if tool:
            tool.enabled = False

    async def call(self, tool_id: str, arguments: dict, **kwargs) -> ToolResult:
        tool = self.get(tool_id)
        if not tool:
            return ToolResult(data={}, output_type="text", tool_id=tool_id, error=f"Tool not found: {tool_id}")
        if not tool.enabled:
            return ToolResult(data={}, output_type=tool.output_type, tool_id=tool_id, error=f"Tool disabled: {tool_id}")
        if not tool.execute_fn:
            return ToolResult(data={}, output_type=tool.output_type, tool_id=tool_id, error=f"Tool missing executor: {tool_id}")

        violation = self._schema_violation(tool, arguments)
        if violation:
            # Execution boundary: the planner's pre-check is advisory (it may
            # pass through after a failed retry), so this is where invalid
            # args are actually refused.
            logger.warning("Tool %s rejected invalid arguments: %s", tool_id, violation)
            return ToolResult(
                data={}, output_type=tool.output_type, tool_id=tool_id,
                error=f"Invalid arguments for {tool_id}: {violation}",
            )

        try:
            # Run in a worker thread: tools may block for minutes (QE runs),
            # and the web SSE stream needs the event loop alive meanwhile.
            result_data = await asyncio.to_thread(tool.execute_fn, arguments)
            if not isinstance(result_data, dict):
                result_data = {"content": str(result_data)}
            return ToolResult(data=result_data, output_type=tool.output_type, tool_id=tool_id)
        except Exception as exc:
            import traceback
            return ToolResult(data={}, output_type=tool.output_type, tool_id=tool_id, error=str(exc), traceback=traceback.format_exc())

    @staticmethod
    def _schema_violation(tool: ToolEntry, arguments: Any) -> Optional[str]:
        """First schema violation in ``arguments``, or None if valid.

        None-valued keys are treated as absent before validating: templates
        and LLM plans legitimately emit explicit nulls for optional params,
        and the tool boundary already coerces them the same way.
        """
        schema = tool.input_schema if isinstance(tool.input_schema, dict) else None
        if not schema:
            return None
        if not isinstance(arguments, dict):
            return f"args must be an object, got {type(arguments).__name__}"
        effective = {k: v for k, v in arguments.items() if v is not None}
        validator = jsonschema.Draft7Validator(schema)
        err = next(iter(sorted(validator.iter_errors(effective), key=lambda e: list(e.path))), None)
        if err is None:
            return None
        loc = ".".join(str(p) for p in err.path) or "args"
        return f"{loc} {err.message}"


registry = ToolRegistry()
