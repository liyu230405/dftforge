"""MCP tool bindings for the graph runtime.

graph.templates : list available graph templates
graph.run       : create + run a template to completion (local executor)
graph.status    : inspect a run's node states and outputs
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dft_forge.runtime.graph import OUTPUT_REF_RE
from dft_forge.tools.models import ToolEntry

DEFAULT_WORKSPACE = Path(os.environ.get("DFT_FORGE_WORKSPACE", "./work/graphs"))
DEFAULT_TEMPLATES = Path(__file__).resolve().parents[2] / "templates"

_ENGINE_CACHE: dict = {}


def reset_engine_cache() -> None:
    """Drop cached engines after compute-backend settings change."""
    cancel_running_graphs()
    _ENGINE_CACHE.clear()


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
    from dft_forge.executor import FakeExecutor, LocalExecutor, SSHExecutor
    import shutil

    executor_kind = os.environ.get("DFT_FORGE_EXECUTOR", "local").strip().lower()
    if executor_kind == "ssh":
        host_spec = os.environ.get("DFT_FORGE_SSH_HOST", "").strip()
        if not host_spec:
            raise ValueError("DFT_FORGE_SSH_HOST is required when DFT_FORGE_EXECUTOR=ssh")
        username = os.environ.get("DFT_FORGE_SSH_USER") or None
        host = host_spec
        if "@" in host_spec and username is None:
            username, host = host_spec.rsplit("@", 1)
        key = os.environ.get("DFT_FORGE_SSH_KEY", "").strip()
        return SSHExecutor(
            host=host,
            username=username,
            ssh_key=Path(key).expanduser() if key else None,
            port=int(os.environ.get("DFT_FORGE_SSH_PORT", "22")),
            remote_workdir=os.environ.get("DFT_FORGE_REMOTE_WORKDIR", "/root/workspace"),
            qe_bin_dir=os.environ.get("DFT_FORGE_REMOTE_QE_BIN", "/opt/qe/bin"),
            scheduler=os.environ.get("DFT_FORGE_SCHEDULER", "none"),
            max_walltime_sec=int(os.environ.get("DFT_FORGE_WALLTIME", "1800")),
        )
    if executor_kind not in {"", "local", "fake"}:
        raise ValueError(f"unsupported DFT_FORGE_EXECUTOR: {executor_kind!r}")
    if executor_kind == "fake":
        return FakeExecutor()

    qe_bin = os.environ.get("DFT_FORGE_QE_BIN")
    if qe_bin and Path(qe_bin).is_dir():
        return LocalExecutor(qe_bin_dir=Path(qe_bin), max_walltime_sec=1800)
    pw = shutil.which("pw.x")
    if pw:
        return LocalExecutor(qe_bin_dir=Path(pw).parent, max_walltime_sec=1800)
    return FakeExecutor()


def _graph_templates(args: dict) -> dict:
    engine = _get_engine(args.get("workdir"))
    return {"templates": engine.list_templates()}


def _resolve_template_outputs(template, run) -> dict:
    """Resolve the template's top-level output contract against a finished run.

    Each declared output ("band_gap_ev": "${nodes.bands.outputs.band_gap_ev}")
    becomes a concrete value; a ref whose node did not succeed yields None —
    callers can trust the key set even for failed runs.
    """
    outs: dict = {}
    for name, ref in (template.outputs or {}).items():
        m = OUTPUT_REF_RE.match(str(ref))
        if not m:
            outs[name] = ref
            continue
        node = run.nodes.get(m.group(1))
        if node is None or node.state.value != "succeeded":
            outs[name] = None
            continue
        v = node.outputs.get(m.group(2))
        outs[name] = v.item() if hasattr(v, "item") else v
    return outs


def _graph_run(args: dict) -> dict:
    template_id = args.get("template_id", "")
    if not template_id:
        return {"error": "template_id is required"}
    engine = _get_engine(args.get("workdir"))
    inputs = args.get("inputs") or {}
    on_event = args.pop("_on_event", None)
    cancel_event = args.pop("_cancel_event", None)
    try:
        run = engine.run_template(template_id, inputs, on_event=on_event, cancel_event=cancel_event)
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
    return {
        "run_id": run.run_id,
        "state": run.state.value,
        "nodes": nodes,
        "outputs": _resolve_template_outputs(engine.get_template(template_id), run),
    }


def _graph_status(args: dict) -> dict:
    run_id = args.get("run_id", "")
    if not run_id:
        runs = _get_engine(args.get("workdir")).list_runs()
        return {"runs": runs}
    try:
        return _get_engine(args.get("workdir")).status(run_id)
    except ValueError as exc:
        return {"error": str(exc)}


def cancel_graphs_under(base_dir) -> int:
    """Kill in-flight subprocesses ONLY for engines whose workdir is under
    ``base_dir``.

    Sessions run their graphs under ``<session_dir>/graphs``, so a stop click
    passes the session dir here; other sessions' engines are untouched.
    """
    killed = 0
    base = Path(base_dir).resolve()
    for key, engine in _ENGINE_CACHE.items():
        try:
            workdir = Path(key).resolve()
        except OSError:
            continue
        if workdir == base or base in workdir.parents:
            killed += engine._cancel_executors()
    return killed


def cancel_running_graphs() -> int:
    """Kill in-flight subprocesses across ALL cached engines.

    Shutdown-level sweep only — per-session cancellation must use
    cancel_graphs_under() so one stop click doesn't kill other sessions.
    """
    killed = 0
    for engine in _ENGINE_CACHE.values():
        killed += engine._cancel_executors()
    return killed


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
