"""Regression tests for agent-loop cancellation and command logging."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from threading import Event

from dft_forge.agent_loop.executor import run_session
from dft_forge.tools.models import ToolResult


class _Planner:
    last_plan_ir = None

    def __init__(self, steps, provider=None):
        self.steps = steps
        self.provider = provider

    def plan(self, message, ctx, history):
        return self.steps, None

    def _get_provider(self):
        return self.provider


class _Rules:
    def __init__(self, steps):
        self.steps = steps

    def plan(self, message, ctx):
        return self.steps


class _CancellingRegistry:
    def __init__(self, cancel_event):
        self.cancel_event = cancel_event
        self.calls = []

    async def call(self, tool_id, arguments, **kwargs):
        self.calls.append((tool_id, arguments, kwargs))
        self.cancel_event.set()
        return ToolResult(
            data={"run_id": "run_test", "state": "paused", "nodes": {}, "outputs": {}},
            output_type="json",
            tool_id=tool_id,
        )


def _steps():
    return [
        {
            "tool": "graph.run",
            "step_id": "first",
            "description": "首个计算",
            "args": {"template_id": "t1_vc_relax", "inputs": {"material": "Si"}},
        },
        {
            "tool": "graph.run",
            "step_id": "second",
            "description": "后续计算",
            "args": {"template_id": "t1_vc_relax", "inputs": {"material": "Al"}},
        },
    ]


def test_cancel_stops_remaining_steps_and_skips_narration(tmp_path, monkeypatch):
    cancel_event = Event()
    registry = _CancellingRegistry(cancel_event)
    planner = _Planner(_steps(), provider=object())
    rules = _Rules(_steps())
    narrated = []
    monkeypatch.setattr(
        "dft_forge.agent_loop.executor.narrate",
        lambda *args, **kwargs: narrated.append(args) or "不应调用",
    )

    result = asyncio.run(
        run_session(
            "运行两个计算",
            tmp_path,
            ctx={"cancel_event": cancel_event},
            registry=registry,
            llm_planner=planner,
            rule_planner=rules,
        )
    )

    assert [tool_id for tool_id, _, _ in registry.calls] == ["graph.run"]
    assert [item["step_id"] for item in result["chain"]] == ["first"]
    assert result["chain"][0]["status"] == "cancelled"
    assert result["reply"] == "计算已停止（进行中的步骤已终止；已完成的结果保留在会话中）。"
    assert narrated == []


def test_command_log_excludes_runtime_handles_and_is_json_serializable(tmp_path):
    cancel_event = Event()
    registry = _CancellingRegistry(cancel_event)
    planner = _Planner(_steps())

    result = asyncio.run(
        run_session(
            "运行两个计算",
            tmp_path,
            ctx={"cancel_event": cancel_event},
            registry=registry,
            llm_planner=planner,
            rule_planner=_Rules(_steps()),
        )
    )

    command_args = result["commands"][0]["args"]
    assert "_cancel_event" not in command_args
    assert "_on_event" not in command_args
    json.dumps(result["commands"])
    assert "_cancel_event" in registry.calls[0][1]
    assert "_on_event" in registry.calls[0][1]
