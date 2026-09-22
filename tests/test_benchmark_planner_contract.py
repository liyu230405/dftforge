"""Regressions from the live DeepSeek planning benchmark; no network/QE."""

import json
import asyncio

from dft_forge.agent_loop import AgentLoop
from dft_forge.agent_loop.llm_planner import LLMPlanner
from dft_forge.agent_loop.planning import capability_catalog_text
from dft_forge.agent_loop.step_validator import validate_steps
from dft_forge.tools.definitions import register_default_tools
from dft_forge.tools.registry import registry


def setup_module():
    if registry.get("workflow.phonon_dfpt") is None:
        register_default_tools()


def test_live_catalog_exposes_runtime_schema_contracts():
    catalog = capability_catalog_text(registry)
    for tool_id in ("structure.import", "structure.build2d", "workflow.phonon_dfpt", "thermo.chemical_potential_scan"):
        line = next(line for line in catalog.splitlines() if line.startswith(f"- {tool_id} "))
        schema = json.loads(line.split(" | args_schema: ", 1)[1])
        assert schema == registry.get(tool_id).input_schema


def test_atomic_adsorption_rejects_molecular_or_missing_species():
    for adsorb in ({"element": "CO2", "site": "top"}, {"molecule": "CO2"}, {"site": "top"}):
        assert validate_steps([{"tool": "structure.build2d", "args": {"kind": "graphene", "adsorb": adsorb}}], registry)
    assert not validate_steps([{"tool": "structure.build2d", "args": {
        "kind": "mos2", "adsorb": {"element": "O", "site": "top"},
    }}], registry)


def test_phonon_grid_contract_rejects_non_grid_values():
    for grid in ("4x4x4", [4, 4], [4, 4, 0], [4, 4, 4, 4]):
        assert validate_steps([{"tool": "workflow.phonon_dfpt", "args": {"q_grid": grid}}], registry)
    assert not validate_steps([{"tool": "workflow.phonon_dfpt", "args": {"q_grid": [4, 4, 4]}}], registry)


def test_schema_retry_keeps_the_original_structure_context():
    class Provider:
        def __init__(self):
            self.prompts = []

        def chat(self, system, user):
            self.prompts.append(user)
            grid = "4x4x4" if len(self.prompts) == 1 else [4, 4, 4]
            return json.dumps({"steps": [{"tool": "workflow.phonon_dfpt", "args": {"q_grid": grid}}]})

    provider = Provider()
    planner = LLMPlanner(registry)
    planner._provider, planner._checked = provider, True
    steps, _ = planner.plan("先规划这个结构的声子，不要运行", {"last_structure_source": "/test/uploaded.cif"})
    assert steps[0]["args"]["q_grid"] == [4, 4, 4]
    assert len(provider.prompts) == 2
    assert all("/test/uploaded.cif" in user for user in provider.prompts)


def test_advisory_only_plan_retains_metadata_and_emits_plan_event(tmp_path):
    class Provider:
        def chat(self, system, user):
            return json.dumps({
                "execution_mode": "plan", "support_status": "partial",
                "reply": "缺少化学势，先补充参考态。", "actions": [],
                "blocking_reasons": ["缺少绝对化学势"],
                "advisory_steps": [{"capability": "workflow.neutral_vacancy", "description": "待补充参考态"}],
            })

    loop = AgentLoop()
    loop.llm_planner._provider, loop.llm_planner._checked = Provider(), True
    events = []
    result = asyncio.run(loop.run("只规划氧空位形成能", tmp_path, on_event=events.append))
    assert result["ctx"]["last_plan"]["support_status"] == "partial"
    assert result["ctx"]["last_plan"]["execution_mode"] == "plan"
    assert result["commands"] == []
    assert "未执行任何工具" in result["reply"]
    assert any(event["type"] == "plan" for event in events)
