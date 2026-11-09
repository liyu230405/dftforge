"""Tool side-effect classification, mirrored from dft-forge.

Single source of truth in the product is
``dft_forge/tools/definitions.py`` (the ``metadata`` table) plus
``graph_tools.register_graph_tools``.  The harness must stay usable with only a
run record on disk, so we keep a copy here; when dft_forge is importable the
live registry is consulted and any drift is reported (never silently scored
against a stale table).
"""
from __future__ import annotations

from typing import Dict, Optional

READ_ONLY = "read_only"
WORKSPACE_WRITE = "workspace_write"
EXTERNAL_EXECUTION = "external_execution"

# Mirrored from dft_forge/tools/definitions.py::register_default_tools.
SIDE_EFFECTS: Dict[str, str] = {
    # structure builders
    "structure.generate": WORKSPACE_WRITE,
    "structure.build2d": WORKSPACE_WRITE,
    "structure.dope": WORKSPACE_WRITE,
    "structure.supercell": WORKSPACE_WRITE,
    "structure.vacancy": WORKSPACE_WRITE,
    "structure.molecule": WORKSPACE_WRITE,
    "structure.reference": WORKSPACE_WRITE,
    "structure.import": WORKSPACE_WRITE,
    "structure.analyze": WORKSPACE_WRITE,
    # input / execution
    "input.build": WORKSPACE_WRITE,
    "job.submit": EXTERNAL_EXECUTION,
    "job.status": READ_ONLY,
    "result.parse": WORKSPACE_WRITE,
    "result.verify": WORKSPACE_WRITE,
    # inspection / reduction
    "ledger.query": READ_ONLY,
    "doctor": READ_ONLY,
    "analysis.compare": READ_ONLY,
    "thermo.eads": READ_ONLY,
    "thermo.formation": READ_ONLY,
    "thermo.chemical_potential_scan": READ_ONLY,
    # macro workflows (dispatch real QE jobs)
    "workflow.convergence": EXTERNAL_EXECUTION,
    "workflow.neutral_vacancy": EXTERNAL_EXECUTION,
    "workflow.charged_vacancy": EXTERNAL_EXECUTION,
    "workflow.adsorption_scan": EXTERNAL_EXECUTION,
    "workflow.magnetic_compare": EXTERNAL_EXECUTION,
    "workflow.phonon_dfpt": EXTERNAL_EXECUTION,
    # graph tools
    "graph.templates": READ_ONLY,
    "graph.run": EXTERNAL_EXECUTION,
    "graph.status": READ_ONLY,
    "help": READ_ONLY,
}

# Effects that count as "the agent actually did something" in plan-only
# questions. workspace_write is included on purpose: even generating files
# violates an explicit "只规划，不要实际运行" contract; the product's own
# registry blocks every non-read_only tool in plan mode.
_NON_READ_ONLY = {WORKSPACE_WRITE, EXTERNAL_EXECUTION, "unknown"}


def side_effect(tool_id: Optional[str]) -> str:
    if not tool_id:
        return "unknown"
    return SIDE_EFFECTS.get(tool_id, "unknown")


def is_read_only(tool_id: Optional[str]) -> bool:
    return side_effect(tool_id) == READ_ONLY


def violates_plan_only(tool_id: Optional[str]) -> bool:
    return side_effect(tool_id) in _NON_READ_ONLY


def verify_against_registry() -> list[str]:
    """Return drift descriptions vs the live dft_forge registry ([] = in sync)."""
    try:
        from dft_forge.tools.definitions import register_default_tools
        from dft_forge.tools.registry import registry as global_registry

        register_default_tools()
        tools = {t.id: t.side_effect for t in global_registry.list_all()}
    except Exception as exc:
        return [f"dft_forge not importable ({exc}); using mirrored side-effect table"]
    drift = []
    for tool_id, effect in SIDE_EFFECTS.items():
        if tool_id in tools and tools[tool_id] != effect:
            drift.append(f"{tool_id}: mirror={effect} live={tools[tool_id]}")
    live_ids = set(tools) - set(SIDE_EFFECTS)
    # Framework-internal tools that never appear in plans may exist; only flag
    # externally registered compute tools as actionable drift.
    for tool_id in sorted(live_ids):
        if tools[tool_id] != READ_ONLY:
            drift.append(f"{tool_id}: missing from mirror (live={tools[tool_id]})")
    return drift
