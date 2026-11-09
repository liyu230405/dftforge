"""Regression tests for plan safety, capability status, and structure semantics."""

from __future__ import annotations

import asyncio
from pathlib import Path

from dft_forge.agent_loop import AgentLoop
from dft_forge.agent_loop.planning import (
    PlanIR,
    build_capability_graph,
    compile_plan,
    render_plan_reply,
    requested_plan_only,
)
from dft_forge.agent_loop.step_validator import semantic_plan_errors, validate_steps
from dft_forge.tools.definitions import register_default_tools
from dft_forge.tools.registry import registry


def _ensure_registered():
    if not registry.get("graph.run"):
        register_default_tools()


def test_plan_only_language_is_a_safety_downgrade():
    assert requested_plan_only("请先规划，不要实际运行")
    assert requested_plan_only("just plan, without running")


def test_plan_only_never_calls_graph(monkeypatch, tmp_path):
    loop = AgentLoop()
    called = []

    def fake_plan(message, ctx, history=None):
        return ([{
            "tool": "graph.run",
            "args": {"template_id": "t1_vc_relax", "inputs": {"material": "Si"}},
            "description": "规划 Si 结构优化",
        }], None)

    original_call = registry.call

    async def spy_call(tool_id, arguments, **kwargs):
        called.append(tool_id)
        return await original_call(tool_id, arguments, **kwargs)

    monkeypatch.setattr(loop.llm_planner, "plan", fake_plan)
    monkeypatch.setattr(registry, "call", spy_call)
    result = asyncio.run(loop.run("请先规划 Si 的 vc-relax，不要实际运行", tmp_path))
    assert result["commands"] == []
    assert result["results"] == []
    assert called == []
    assert "未执行" in result["reply"]


def test_registry_plan_mode_blocks_external_execution():
    _ensure_registered()
    result = asyncio.run(registry.call(
        "graph.run",
        {"template_id": "t0_scf", "inputs": {"material": "Si"}},
        execution_mode="plan",
    ))
    assert result.error and "plan-only" in result.error


def test_unknown_actions_are_reported_without_silent_truncation():
    _ensure_registered()
    plan = PlanIR(
        goal="long workflow",
        actions=[
            {"capability": "graph.templates", "args": {}, "description": str(i)}
            for i in range(9)
        ],
    )
    steps = compile_plan(plan, registry)
    assert len(steps) == 9
    assert plan.over_budget is True
    assert plan.support_status == "partial"
    assert plan.blocking_reasons


def test_framework_external_capability_is_advisory_only():
    _ensure_registered()
    plan = PlanIR(
        goal="phonons",
        actions=[{"capability": "phonon.dispersion", "args": {}, "description": "DFPT 声子"}],
    )
    steps = compile_plan(plan, registry)
    assert steps == []
    assert plan.support_status == "unsupported"
    assert "phonon.dispersion" in plan.missing_capabilities


def test_deferred_thermo_inputs_are_valid_only_at_plan_time():
    _ensure_registered()
    assert validate_steps([
        {"tool": "thermo.formation", "args": {"formula": "NaCl"}},
        {"tool": "thermo.eads", "args": {}},
    ], registry) == []
    result = asyncio.run(registry.call("thermo.formation", {"formula": "NaCl"}))
    assert result.error and "compound_energy" in result.error


def test_semantic_guards_reject_vacancy_and_self_doping_mappings():
    vacancy_errors = semantic_plan_errors("计算 MgO 的 O 空位形成能", [{
        "tool": "structure.dope",
        "args": {"source": "MgO", "element": "O"},
        "description": "移除 O 原子生成空位",
    }])
    assert any("不能移除原子" in e for e in vacancy_errors)

    self_errors = semantic_plan_errors("比较 P 掺杂 Si 与纯 Si", [{
        "tool": "structure.dope",
        "args": {"source": "si_2x2x2.cif", "element": "Si"},
        "description": "构建纯 Si 对照超胞（无实际掺杂）",
    }])
    assert any("structure.supercell" in e for e in self_errors)


def test_explicit_bulk_supercell_and_vacancy_builders():
    from dft_forge.structure_builder import supercell_3d, vacancy_3d

    pristine = supercell_3d("Si", supercell="2x2x2")
    defect = vacancy_3d("MgO", "O", supercell="2x2x2")
    assert pristine.atoms.get_chemical_formula() == "Si16"
    assert defect.atoms.get_chemical_formula() == "Mg8O7"


def test_dos_capability_advertises_projected_dos_engine():
    _ensure_registered()
    caps = build_capability_graph(
        registry, Path(__file__).resolve().parents[1] / "templates"
    )
    dos = next(c for c in caps if c.id == "run.t2_dos")
    assert "projwfc.x" in dos.description
    assert "pdos_elements" in dos.outputs


def test_plan_reply_does_not_duplicate_steps():
    plan = PlanIR.from_steps("test", [
        {"tool": "graph.templates", "description": "one"},
        {"tool": "graph.status", "description": "two"},
    ])
    reply = render_plan_reply(plan, plan.actions)
    assert reply.count("one") == 1
    assert reply.count("two") == 1
