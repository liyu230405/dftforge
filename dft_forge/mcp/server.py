"""MCP server exposing DFT-Forge tools."""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from dft_forge.tools.definitions import register_default_tools
from dft_forge.tools.registry import registry

logger = logging.getLogger(__name__)

register_default_tools()

server = Server("dft-forge")


@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    return [
        Tool(
            name=t.id,
            description=t.description,
            inputSchema=t.input_schema,
        )
        for t in registry.list_all()
        if t.enabled
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    tool = registry.get(name)
    if not tool:
        return [TextContent(type="text", text=json.dumps({"error": f"Tool not found: {name}"}))]
    result = await registry.call(name, arguments)
    payload = {
        "tool_id": result.tool_id,
        "output_type": result.output_type,
        "data": result.data,
        "error": result.error,
        "traceback": result.traceback,
    }
    return [TextContent(type="text", text=json.dumps(payload, default=str))]


async def run_stdio() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main() -> None:
    import asyncio
    asyncio.run(run_stdio())


if __name__ == "__main__":
    main()
