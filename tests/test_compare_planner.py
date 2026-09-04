"""Comparison intent, session ledger, and self-doping guard tests."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List

import pytest

from dft_forge.agent_loop import AgentLoop
from dft_forge.tools.definitions import _analysis_compare


LEDGER_TWO_BANDS: List[Dict[str, Any]] = [
    {"label": "N掺杂石墨烯能带", "template": "t2_bands", "band_gap_ev": 0.6,
     "is_metal": False, "fermi_ev": -1.2, "natoms": 18},
    {"label": "纯石墨烯能带", "template": "t2_bands", "band_gap_ev": 0.0,
     "is_metal": True, "fermi_ev": -0.5, "natoms": 18},
]


class TestAnalysisCompareTool:
    def test_band_gap_math(self):
        r = _analysis_compare({"entries": LEDGER_TWO_BANDS, "metric": "band_gap"})
        assert r["ok"] is True
        assert r["metric"] == "band_gap"
        assert r["items"][0]["band_gap_ev"] == 0.6
        assert r["items"][1]["type"] == "金属 (零带隙)"
        d = r["differences"][0]
        assert d["between"] == ["N掺杂石墨烯能带", "纯石墨烯能带"]
        assert d["delta_band_gap_ev"] == pytest.approx(-0.6, abs=0.001)
        assert d["wider"] == "N掺杂石墨烯能带"

    def test_energy_math_per_atom(self):
        entries = [
            {"label": "体系A", "energy_ry": -100.0, "natoms": 18},
            {"label": "体系B", "energy_ry": -110.0, "natoms": 20},
        ]
        r = _analysis_compare({"entries": entries, "metric": "energy"})
        assert r["ok"] is True
        # per-atom: A = -75.59 eV, B = -74.83 eV → B higher per atom
        a_pa = r["items"][0]["energy_ev_per_atom"]
        b_pa = r["items"][1]["energy_ev_per_atom"]
        assert a_pa == pytest.approx(-100.0 * 13.605693 / 18, abs=0.01)
        assert b_pa == pytest.approx(-110.0 * 13.605693 / 20, abs=0.01)
        d = r["differences"][0]
        assert d["lower"] == "体系B"  # total energy lower
        assert d["delta_energy_ev"] == pytest.approx(-10.0 * 13.605693, abs=0.01)

    def test_auto_metric_picks_band_gap(self):
        r = _analysis_compare({"entries": LEDGER_TWO_BANDS, "metric": "auto"})
        assert r["metric"] == "band_gap"

    def test_subsmearing_gap_labeled_near_metal(self):
        # graphene-family systems: gap below 2×degauss is smearing noise, not physics
        entries = [
            {"label": "纯石墨烯", "band_gap_ev": 0.1782, "is_metal": True, "fermi_ev": -1.0},
            {"label": "N掺杂", "band_gap_ev": 0.0928, "is_metal": True, "fermi_ev": -1.1},
        ]
        r = _analysis_compare({"entries": entries, "metric": "band_gap"})
        assert r["ok"] is True
        assert "近零带隙/金属" in r["items"][0]["type"]
        assert "不可分辨" in r["items"][0]["type"]

    def test_missing_gap_errors(self):
        entries = [
            {"label": "有带隙", "band_gap_ev": 1.0},
            {"label": "只有能量", "energy_ry": -10.0},
        ]
        r = _analysis_compare({"entries": entries, "metric": "band_gap"})
        assert "error" in r and "只有能量" in r["error"]

    def test_single_entry_rejected(self):
        r = _analysis_compare({"entries": [{"label": "x", "band_gap_ev": 1.0}]})
        assert "error" in r


class TestRulePlannerCompare:
    def test_compare_with_ledger_routes_to_compare(self):
        loop = AgentLoop()
        steps = loop.plan("现在开始对比", ctx={"ledger": LEDGER_TWO_BANDS})
        assert len(steps) == 1
        assert steps[0]["tool"] == "analysis.compare"
        assert steps[0]["args"]["use_last"] == 2

    def test_compare_energy_metric(self):
        loop = AgentLoop()
        steps = loop.plan("对比一下能量", ctx={"ledger": LEDGER_TWO_BANDS})
        assert steps[0]["args"]["metric"] == "energy"

    def test_compare_gap_metric(self):
        loop = AgentLoop()
        steps = loop.plan("比较能带", ctx={"ledger": LEDGER_TWO_BANDS})
        assert steps[0]["args"]["metric"] == "band_gap"

    def test_compare_without_ledger_gives_guidance_not_dead_end(self):
        loop = AgentLoop()
        steps = loop.plan("现在开始对比", ctx={})
        assert steps[0]["tool"] == "help"
        assert "先有两个体系" in steps[0]["description"]

    def test_compare_with_one_entry_mentions_it(self):
        loop = AgentLoop()
        steps = loop.plan("对比", ctx={"ledger": LEDGER_TWO_BANDS[:1]})
        assert steps[0]["tool"] == "help"
        assert "N掺杂石墨烯能带" in steps[0]["description"]

    def test_compare_intent_never_falls_into_job_status(self):
        # "那你直接开始这个任务，然后再进行比较" contains 任务/开始
        # which previously routed to a job-status dead end
        loop = AgentLoop()
        steps = loop.plan("那你直接开始这个任务，然后再进行比较任务很难吗", ctx={})
        tools = [s["tool"] for s in steps]
        assert "job.status" not in tools
        assert tools[0] in ("help", "analysis.compare")


class _RecordingRegistry:
    """Stand-in registry: records calls, returns canned payloads."""

    def __init__(self, payloads: Dict[str, Any]):
        self.payloads = payloads
        self.calls: List[Dict[str, Any]] = []

    def list_all(self):
        from dft_forge.tools.definitions import register_default_tools
        from dft_forge.tools.registry import registry as real_registry
        if not real_registry._tools:
            register_default_tools()
        return real_registry.list_all()

    async def call(self, tool_id: str, arguments: dict, **kwargs):
        self.calls.append({"tool": tool_id, "args": dict(arguments)})
        payload = self.payloads.get(tool_id, {})
        if isinstance(payload, Exception):
            raise payload

        class _R:
            def __init__(self, data, tid):
                self.data = data
                self.tool_id = tid
                self.error = None
                self.traceback = None
        return _R(payload, tool_id)


class TestExecutorCompareInjection:
    def test_compare_step_gets_ledger_injected(self, tmp_path: Path):
        reg = _RecordingRegistry({"analysis.compare": {
            "ok": True, "metric": "band_gap",
            "items": [{"label": "A", "band_gap_ev": 1.0}],
            "differences": [], "note": "",
        }})
        loop = AgentLoop(registry_obj=reg)
        result = asyncio.run(loop.run(
            "现在开始对比", tmp_path, ctx={"ledger": LEDGER_TWO_BANDS},
        ))
        compare_calls = [c for c in reg.calls if c["tool"] == "analysis.compare"]
        assert len(compare_calls) == 1
        injected = compare_calls[0]["args"]["entries"]
        assert [e["label"] for e in injected] == ["N掺杂石墨烯能带", "纯石墨烯能带"]
        assert "对比结果" in result["reply"]

    def test_compare_rendering_includes_delta(self, tmp_path: Path):
        reg = _RecordingRegistry({"analysis.compare": {
            "ok": True, "metric": "band_gap",
            "items": [
                {"label": "N掺杂", "band_gap_ev": 0.6, "type": "semiconductor"},
                {"label": "纯", "band_gap_ev": 0.0, "type": "metal (0 gap)"},
            ],
            "differences": [{
                "between": ["N掺杂", "纯"], "delta_band_gap_ev": -0.6, "wider": "N掺杂",
            }],
            "note": "带隙差 = 两者带隙之差",
        }})
        loop = AgentLoop(registry_obj=reg)
        result = asyncio.run(loop.run(
            "现在开始对比", tmp_path, ctx={"ledger": LEDGER_TWO_BANDS},
        ))
        assert "0.6" in result["reply"]
        assert "-0.6000" in result["reply"] or "-0.6" in result["reply"]


class _VagueNarrator:
    """LLM stand-in that paraphrases results into numberless promises."""

    def chat(self, system: str, user: str) -> str:
        return "对比任务已提交，后续将输出两者的带隙差值及相关结果。"


class TestNarrationGuarantees:
    def test_compare_numbers_survive_vague_narration(self, tmp_path: Path):
        reg = _RecordingRegistry({"analysis.compare": {
            "ok": True, "metric": "band_gap",
            "items": [
                {"label": "N掺杂", "band_gap_ev": 0.0928, "type": "semiconductor/insulator"},
                {"label": "纯", "band_gap_ev": 0.1782, "type": "semiconductor/insulator"},
            ],
            "differences": [{
                "between": ["N掺杂", "纯"], "delta_band_gap_ev": 0.0854, "wider": "纯",
            }],
            "note": "带隙差 = 两者带隙之差",
        }})
        loop = AgentLoop(registry_obj=reg)
        loop.llm_planner._provider = _VagueNarrator()
        loop.llm_planner._checked = True
        result = asyncio.run(loop.run(
            "现在开始对比", tmp_path, ctx={"ledger": LEDGER_TWO_BANDS},
        ))
        # the vague narration is kept, but the deterministic numbers follow it
        assert "对比任务已提交" in result["reply"]
        assert "对比结果" in result["reply"]
        assert "0.0928" in result["reply"] and "0.1782" in result["reply"]
        assert "+0.0854" in result["reply"]

    def test_graph_run_workdir_always_forced(self, tmp_path: Path):
        """An LLM-hallucinated workdir must never scatter runs outside graphs/."""
        reg = _RecordingRegistry({"graph.run": {
            "run_id": "run_test", "state": "succeeded", "nodes": {},
        }})
        loop = AgentLoop(registry_obj=reg)
        loop.plan = lambda msg, ctx, history=None: [
            {"tool": "graph.run",
             "args": {"template_id": "t0_scf", "inputs": {"material": "Si"},
                      "workdir": "/hallucinated/path"},
             "description": "测试计算"},
        ]
        asyncio.run(loop.run("测试", tmp_path, ctx={}))
        call = next(c for c in reg.calls if c["tool"] == "graph.run")
        assert call["args"]["workdir"] == str(tmp_path / "graphs")


class TestSelfDopingGuard:
    def test_c_doped_graphene_rejected(self, tmp_path: Path):
        reg = _RecordingRegistry({})
        loop = AgentLoop(registry_obj=reg)
        result = asyncio.run(loop.run("构建C掺杂石墨烯", tmp_path, ctx={}))
        build_calls = [c for c in reg.calls if c["tool"] == "structure.build2d"]
        assert not build_calls, "C-doped graphene build must never execute"
        assert "石墨烯" in result["reply"]
        assert "纯碳" in result["reply"] or "本身就是" in result["reply"]

    def test_si_doped_si_rejected(self, tmp_path: Path):
        reg = _RecordingRegistry({})
        loop = AgentLoop(registry_obj=reg)
        result = asyncio.run(loop.run("构建Si掺杂Si的超胞", tmp_path, ctx={}))
        dope_calls = [c for c in reg.calls if c["tool"] == "structure.dope"]
        assert not dope_calls, "Si-doped Si build must never execute"
        assert "掺杂原子与本体元素相同" in result["reply"]

    def test_real_dopant_still_allowed(self, tmp_path: Path):
        # N-doped graphene must NOT be blocked (regression guard)
        reg = _RecordingRegistry({"structure.build2d": {
            "ok": True, "output": str(tmp_path / "g.cif"), "formula": "C17N",
            "natoms": 18, "kind": "graphene",
        }})
        loop = AgentLoop(registry_obj=reg)
        asyncio.run(loop.run("构建N掺杂石墨烯", tmp_path, ctx={}))
        build_calls = [c for c in reg.calls if c["tool"] == "structure.build2d"]
        assert build_calls, "N-doped graphene build must execute"


class TestLedgerExtraction:
    def test_graph_run_success_appends_ledger_entry(self, tmp_path: Path):
        payload = {
            "run_id": "run_test1",
            "state": "succeeded",
            "nodes": {
                "scf": {"state": "succeeded", "outputs": {"energy_ry": -100.5, "natoms": 18}},
                "bands": {"state": "succeeded", "outputs": {"band_gap_ev": 1.5, "is_metal": False, "fermi_ev": -2.0}},
            },
        }
        reg = _RecordingRegistry({"graph.run": payload})
        loop = AgentLoop(registry_obj=reg)
        result = asyncio.run(loop.run("算NaCl的能带", tmp_path, ctx={}))
        # run() shallow-copies ctx; the ledger lives in the RETURNED ctx,
        # which the web layer persists across turns
        ledger = result["ctx"].get("ledger")
        assert ledger, "ledger must be populated after a successful graph run"
        e = ledger[-1]
        assert e["run_id"] == "run_test1"
        assert e["band_gap_ev"] == 1.5
        assert e["energy_ry"] == -100.5
        assert e["natoms"] == 18

    def test_failed_run_not_in_ledger(self, tmp_path: Path):
        payload = {
            "run_id": "run_fail",
            "state": "failed",
            "nodes": {"scf": {"state": "failed", "outputs": {}}},
        }
        reg = _RecordingRegistry({"graph.run": payload})
        loop = AgentLoop(registry_obj=reg)
        result = asyncio.run(loop.run("算NaCl的能带", tmp_path, ctx={}))
        assert not result["ctx"].get("ledger")
