"""MCP tool bindings for the graph runtime (CatGo-style graph tools).

graph.templates : list available graph templates
graph.run       : create + run a template to completion (local executor)
graph.status    : inspect a run's node states and outputs
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dft_forge.tools.models import ToolEntry

DEFAULT_WORKSPACE = Path(os.environ.get("DFT_FORGE_WORKSPACE", "./work/graphs"))
DEFAULT_TEMPLATES = Path(__file__).resolve().parents[2] / "templates"

_ENGINE_CACHE: dict = {}


def _get_engine(workdir: Optional[str] = None):
    """Lazily construct a GraphEngine with the QE tool (FakeExecutor offline)."""
    key = str(workdir or DEFAULT_WORKSPACE)
    if key not in _ENGINE_CACHE:
        from dft_forge.engines.qe import QECalcTool, QEParamRepairer
        from dft_forge.executor import FakeExecutor, LocalExecutor
        from dft_forge.runtime.engine import GraphEngine

        executor = _pick_executor()
        engine = GraphEngine(
            base_dir=Path(key),
            tools={"qe": QECalcTool(executor=executor)},
            repairers={"qe_param_bump": QEParamRepairer()},
            templates_dir=DEFAULT_TEMPLATES,
        )
        _ENGINE_CACHE[key] = engine
    return _ENGINE_CACHE[key]


def _pick_executor():
    from dft_forge.executor import FakeExecutor, LocalExecutor
    import shutil

    qe_bin = os.environ.get("DFT_FORGE_QE_BIN")
    if qe_bin and Path(qe_bin).is_dir():
        return LocalExecutor(qe_bin_dir=Path(qe_bin))
    pw = shutil.which("pw.x")
    if pw:
        return LocalExecutor(qe_bin_dir=Path(pw).parent)
    return FakeExecutor()


def _graph_templates(args: dict) -> dict:
    engine = _get_engine(args.get("workdir"))
    return {"templates": engine.list_templates()}


def _graph_run(args: dict) -> dict:
    template_id = args.get("template_id", "")
    if not template_id:
        return {"error": "template_id is required"}
    engine = _get_engine(args.get("workdir"))
    inputs = args.get("inputs") or {}
    try:
        run = engine.run_template(template_id, inputs)
    except ValueError as exc:
        return {"error": str(exc)}
    nodes = {
        nid: {
            "state": n.state.value,
            "error": n.error,
            "outputs": (
                {k: (v.item() if hasattr(v, "item") else v) for k, v in n.outputs.items()}
                if n.state.value == "succeeded"
                else None
            ),
        }
        for nid, n in run.nodes.items()
    }
    return {"run_id": run.run_id, "state": run.state.value, "nodes": nodes}


def _graph_status(args: dict) -> dict:
    run_id = args.get("run_id", "")
    if not run_id:
        runs = _get_engine(args.get("workdir")).list_runs()
        return {"runs": runs}
    try:
        return _get_engine(args.get("workdir")).status(run_id)
    except ValueError as exc:
        return {"error": str(exc)}


def register_graph_tools(registry) -> None:
    registry.register(
        ToolEntry(
            id="graph.templates",
            name="List Graph Templates",
            description="List available workflow graph templates (t1_vc_relax, t2_bands, t2_dos, ...).",
            category="graph",
            input_schema={"type": "object", "properties": {"workdir": {"type": "string"}}},
            execute_fn=_graph_templates,
        )
    )
    registry.register(
        ToolEntry(
            id="graph.run",
            name="Run Graph Template",
            description=(
                "Create and run a workflow graph to completion. "
                "Inputs: template_id + template inputs (e.g. {material: 'NaCl'}). "
                "Set DFT_FORGE_QE_BIN for real pw.x; otherwise a fake executor runs offline."
            ),
            category="graph",
            input_schema={
                "type": "object",
                "properties": {
                    "template_id": {"type": "string"},
                    "inputs": {"type": "object"},
                    "workdir": {"type": "string"},
                },
                "required": ["template_id"],
            },
            execute_fn=_graph_run,
        )
    )
    registry.register(
        ToolEntry(
            id="graph.status",
            name="Graph Run Status",
            description="Inspect graph runs: node states, attempts, errors, and outputs. Omit run_id to list runs.",
            category="graph",
            input_schema={
                "type": "object",
                "properties": {"run_id": {"type": "string"}, "workdir": {"type": "string"}}},
            execute_fn=_graph_status,
        )
    )
