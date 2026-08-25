"""Self-check loop for the agent planner and web frontend behavior."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from dft_forge.agent_loop import AgentLoop


CASES = [
    "计算nacl的键长",
    "帮我看看Si的结构",
    "NaCl bond length",
    "构建Si的vc-relax输入",
    "build a relax input for Si",
    "提交计算",
    "查询账本",
    "环境诊断",
    "随便说点什么",
    "你好",
]


async def main() -> None:
    loop = AgentLoop()
    session = Path("/tmp/dft-forge-self-check")
    session.mkdir(parents=True, exist_ok=True)

    report = []
    for msg in CASES:
        result = await loop.run(msg, session)
        report.append({
            "message": msg,
            "reply": result.get("reply"),
            "commands": [c.get("command") for c in result.get("commands", [])],
            "errors": [r.get("error") for r in result.get("results", [])],
        })

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
