"""Tests for the minimal web chat backend."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from dft_forge.agent_loop import AgentLoop
from dft_forge.web import app


client = TestClient(app)


class TestWebBackend:
    def test_health(self):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_doctor(self):
        r = client.get("/doctor")
        assert r.status_code == 200
        data = r.json()
        assert "qe_binary" in data

    def test_tools(self):
        r = client.get("/tools")
        assert r.status_code == 200
        data = r.json()
        assert "tools" in data
        ids = [t["id"] for t in data["tools"]]
        assert "structure.import" in ids
        assert "structure.generate" in ids
        assert "structure.analyze" in ids
        assert "input.build" in ids
        assert "doctor" in ids

    def test_chat_structure_import(self):
        r = client.post("/chat", json={"message": "导入结构"})
        assert r.status_code == 200
        data = r.json()
        assert "reply" in data
        assert "commands" in data
        assert "results" in data

    def test_chat_structure_analysis(self):
        r = client.post("/chat", json={"message": "计算nacl的键长"})
        assert r.status_code == 200
        data = r.json()
        assert data["commands"][0]["command"] == "structure.generate"
        assert data["commands"][1]["command"] == "structure.analyze"
        assert data["commands"][0]["args"]["source"] == "NaCl"
        assert data["commands"][1]["args"]["source"] == "__generated__"
        assert data["commands"][1]["args"]["pair_types"] == "Na-Cl"
        assert len(data["results"]) == 2
        assert data["results"][0]["tool_id"] == "structure.generate"
        assert data["results"][1]["tool_id"] == "structure.analyze"

    def test_chat_input_build(self):
        r = client.post("/chat", json={"message": "build a relax input"})
        assert r.status_code == 200
        data = r.json()
        assert data["commands"][0]["command"] == "input.build"
        assert data["commands"][0]["args"]["type"] == "vc-relax"

    def test_chat_help(self):
        r = client.post("/chat", json={"message": "help"})
        assert r.status_code == 200
        data = r.json()
        assert "计算键长/分析结构" in data["reply"]
        assert "生成标准结构" in data["reply"]
        assert data["commands"][0]["command"] == "help"

    def test_chat_multi_step_structure_and_input(self):
        r = client.post("/chat", json={"message": "导入结构然后构建vc-relax输入"})
        assert r.status_code == 200
        data = r.json()
        assert len(data["commands"]) >= 1
        assert data["commands"][0]["command"] == "structure.analyze"
        assert len(data["results"]) == len(data["commands"])
        assert all("returncode" in res for res in data["results"])
        assert all("json" in res for res in data["results"])

    def test_chat_doctor_returns_commands_results(self):
        r = client.post("/chat", json={"message": "环境诊断"})
        assert r.status_code == 200
        data = r.json()
        assert data["commands"][0]["command"] == "doctor"
        assert len(data["results"]) == 1
        assert data["results"][0]["tool_id"] == "doctor"
        assert data["results"][0]["returncode"] == 0
        assert "json" in data["results"][0]
        assert data["results"][0]["json"]["qe_binary"] is not None


class TestAgentLoop:
    def test_list_tools(self):
        loop = AgentLoop()
        tools = loop.list_tools()
        assert any(t["id"] == "structure.import" for t in tools)
        assert any(t["id"] == "structure.generate" for t in tools)
        assert any(t["id"] == "structure.analyze" for t in tools)
        assert any(t["id"] == "input.build" for t in tools)
        assert any(t["id"] == "doctor" for t in tools)

    def test_plan_structure_analysis(self):
        loop = AgentLoop()
        steps = loop.plan("计算nacl的键长")
        assert len(steps) == 2
        assert steps[0]["tool"] == "structure.generate"
        assert steps[0]["args"]["source"] == "NaCl"
        assert steps[1]["tool"] == "structure.analyze"
        assert steps[1]["args"]["source"] == "__generated__"
        assert steps[1]["args"]["pair_types"] == "Na-Cl"

    def test_plan_structure_generate(self):
        loop = AgentLoop()
        steps = loop.plan("帮我看看Si的结构")
        assert len(steps) == 2
        assert steps[0]["tool"] == "structure.generate"
        assert steps[1]["tool"] == "structure.analyze"

    def test_plan_multiple_steps(self):
        loop = AgentLoop()
        steps = loop.plan("导入结构然后构建vc-relax输入")
        assert len(steps) == 1
        assert steps[0]["tool"] == "structure.analyze"

    def test_run_returns_multi_step_results(self):
        import asyncio
        loop = AgentLoop()
        result = asyncio.run(loop.run("导入结构然后构建vc-relax输入", Path("/tmp/dft-forge-web/test-multi")))
        assert "reply" in result
        assert "commands" in result
        assert "results" in result
        assert len(result["commands"]) >= 1
        assert len(result["results"]) == len(result["commands"])
        assert all("returncode" in res for res in result["results"])
        assert all("json" in res for res in result["results"])
        assert all("error" in res for res in result["results"])
