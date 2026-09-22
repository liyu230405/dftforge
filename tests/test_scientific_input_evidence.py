"""Reference-state inputs cannot be invented by a chat planner."""
import asyncio
import copy
import json

import pytest

from dft_forge.agent_loop import AgentLoop
from dft_forge.agent_loop.planning import PlanIR, compile_plan
from dft_forge.agent_loop.scientific_inputs import guard_scientific_inputs
from dft_forge.tools.registry import registry


def vacancy():
    return {"tool": "workflow.neutral_vacancy", "args": {
        "source": "MgO", "element": "O", "chemical_potential_ev": 0,
        "_input_evidence": {"source": "user_structured"},
    }}


def apply(steps, values=None, message="运行空位形成能"):
    plan = PlanIR.from_steps(message, copy.deepcopy(steps), source="llm")
    compiled = compile_plan(plan, registry)
    ctx = {"thermodynamic_inputs": values} if values is not None else {}
    guard_scientific_inputs(plan, compiled, message, ctx)
    return plan, compiled, ctx


def explicit_mu(**overrides):
    return {"source": "MgO", "element": "O", "reference": "absolute_dft", "chemical_potential_ev": -20.0, **overrides}


def test_llm_numeric_placeholders_and_private_evidence_do_not_authorize_inputs():
    scan = {"tool": "thermo.chemical_potential_scan", "args": {
        "formula": "MgO", "formation_energy_ev_per_formula": -6.24,
        "competitors": [{"formula": "MgO2", "formation_energy_ev_per_formula": -9.99}],
    }}
    plan, steps, _ = apply([vacancy(), scan])
    assert steps == [] and plan.actions == []
    assert plan.support_status == "partial"
    assert len(plan.advisory_steps) == 2
    assert "-6.24" not in json.dumps(plan.to_dict())
    assert "-9.99" not in json.dumps(plan.to_dict())


@pytest.mark.parametrize("entry", [
    explicit_mu(reference="delta_mu"), explicit_mu(reference=None),
    explicit_mu(chemical_potential_ry=-1), explicit_mu(chemical_potential_ev=float("nan")),
    explicit_mu(chemical_potential_ev=True), explicit_mu(source="Si"), explicit_mu(element="Mg"),
])
def test_ambiguous_nonfinite_or_mismatched_references_are_blocked(entry):
    plan, steps, _ = apply([vacancy()], {"workflow.neutral_vacancy": entry})
    assert steps == [] and plan.support_status == "partial"


def test_explicit_absolute_mu_overrides_model_and_survives_confirmation():
    plan, steps, ctx = apply([vacancy()], {"workflow.neutral_vacancy": explicit_mu()})
    assert steps[0]["args"]["chemical_potential_ev"] == -20
    steps[0]["args"]["chemical_potential_ev"] = 0
    guard_scientific_inputs(plan, steps, "确认执行", ctx)
    assert steps[0]["args"]["chemical_potential_ev"] == -20
    assert plan.support_status == "complete"


def test_only_explicit_structured_user_data_can_supply_zero_absolute_mu():
    values = {"workflow.neutral_vacancy": explicit_mu(chemical_potential_ev=0)}
    plan, steps, ctx = apply([vacancy()], message=json.dumps({"thermodynamic_inputs": values}))
    assert steps[0]["args"]["chemical_potential_ev"] == 0
    assert ctx["thermodynamic_inputs"] == values
    assert plan.support_status == "complete"
    plan, steps, _ = apply([vacancy()], message="O-rich Δμ=0，实验形成焓 -6.24 eV。")
    assert not steps and plan.support_status == "partial"


def test_model_competitors_are_discarded_when_only_parent_is_supplied():
    entry = {"formula": "MgO", "formation_energy_ev_per_formula": -5.0}
    plan, steps, _ = apply([{"tool": "thermo.chemical_potential_scan", "args": {
        **entry, "competitors": [{"formula": "MgO2", "formation_energy_ev_per_formula": -9.99}],
    }}], {"thermo.chemical_potential_scan": entry})
    assert "competitors" not in steps[0]["args"]
    assert "-9.99" not in json.dumps(plan.to_dict())


def test_executor_blocks_unguarded_planner_before_any_tool(monkeypatch, tmp_path):
    loop = AgentLoop()
    monkeypatch.setattr(loop.llm_planner, "plan", lambda *a, **kw: ([vacancy()], None))
    calls = []
    async def forbidden_call(*args, **kwargs):
        calls.append(args)
        raise AssertionError("must not execute")
    monkeypatch.setattr(registry, "call", forbidden_call)
    result = asyncio.run(loop.run("执行 MgO 中性空位形成能", tmp_path))
    assert calls == [] and result["commands"] == []
    assert result["ctx"]["last_plan"]["support_status"] == "partial"


def test_direct_numeric_thermodynamic_api_remains_available():
    result = asyncio.run(registry.call("thermo.chemical_potential_scan", {
        "formula": "MgO", "formation_energy_ev_per_formula": -6.0,
    }))
    assert result.error is None
