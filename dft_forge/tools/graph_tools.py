"""MCP tool bindings for the graph runtime.

graph.templates : list available graph templates
graph.run       : create + run a template to completion (local executor)
graph.status    : inspect a run's node states and outputs
"""

from __future__ import annotations

import os
import hashlib
import json
import shutil
import stat
from pathlib import Path
from typing import Optional

from dft_forge.runtime.graph import OUTPUT_REF_RE
from dft_forge.tools.models import ToolEntry

DEFAULT_WORKSPACE = Path(os.environ.get("DFT_FORGE_WORKSPACE", "./work/graphs"))
DEFAULT_TEMPLATES = Path(__file__).resolve().parents[2] / "templates"

_ENGINE_CACHE: dict = {}
_DEFAULT_WALLTIME = 1800
_MIN_WALLTIME = 30
_MAX_WALLTIME = 604800
_QE_EXECUTABLES = (
    "pw.x",
    "bands.x",
    "dos.x",
    "projwfc.x",
    "ph.x",
    "q2r.x",
    "matdyn.x",
)


class _ReadOnlyTool:
    """Tool-name placeholder used to validate QE templates without running them."""

    name = "qe"


class _ReadOnlyVASPTool:
    """Placeholder so VASP templates validate in read-only engines."""

    name = "vasp"


class _ReadOnlyABINITTool:
    """Placeholder so ABINIT templates validate in read-only engines."""

    name = "abinit"


def reset_engine_cache() -> None:
    """Drop cached engines after compute-backend settings change."""
    cancel_running_graphs()
    _ENGINE_CACHE.clear()


def _get_engine(workdir: Optional[str] = None, *, read_only: bool = False):
    """Lazily construct an execution or read-only graph engine."""
    key = str(workdir or DEFAULT_WORKSPACE)
    if read_only:
        if key in _ENGINE_CACHE:
            return _ENGINE_CACHE[key]
        cache_key = (key, True)
        if cache_key not in _ENGINE_CACHE:
            from dft_forge.runtime.engine import GraphEngine

            engine = GraphEngine(
                base_dir=Path(key),
                tools={"qe": _ReadOnlyTool(), "vasp": _ReadOnlyVASPTool(),
                       "abinit": _ReadOnlyABINITTool()},
                templates_dir=DEFAULT_TEMPLATES,
            )
            _ENGINE_CACHE[cache_key] = engine
        return _ENGINE_CACHE[cache_key]

    # Keep the test/integration injection hook used by callers that provide a
    # fully configured GraphEngine directly under its workdir string.
    if key in _ENGINE_CACHE:
        return _ENGINE_CACHE[key]

    executor, profile, identity = _build_execution()
    # Walltime is deliberately excluded from the execution identity, but it
    # must still select a fresh engine so new attempts carry the new profile.
    profile_key = json.dumps(profile, sort_keys=True, separators=(",", ":"))
    cache_key = (key, False, identity, profile_key)
    if cache_key not in _ENGINE_CACHE:
        from dft_forge.engines.abinit import ABINITCalcTool
        from dft_forge.engines.abinit.tool import ABINITParamRepairer
        from dft_forge.engines.qe import QECalcTool, QEParamRepairer
        from dft_forge.engines.vasp import VASPCalcTool
        from dft_forge.engines.vasp.tool import VASPParamRepairer
        from dft_forge.runtime.engine import GraphEngine

        engine = GraphEngine(
            base_dir=Path(key),
            tools={
                "qe": QECalcTool(executor=executor),
                "vasp": VASPCalcTool(executor=executor),
                "abinit": ABINITCalcTool(executor=executor),
            },
            repairers={
                "qe_param_bump": QEParamRepairer(),
                "vasp_param_bump": VASPParamRepairer(),
                "abinit_param_bump": ABINITParamRepairer(),
            },
            templates_dir=DEFAULT_TEMPLATES,
            execution_identity=identity,
            execution_profile=profile,
        )
        _ENGINE_CACHE[cache_key] = engine
    return _ENGINE_CACHE[cache_key]


def _pick_executor():
    """Return the configured executor, failing closed for invalid local QE."""
    return _build_execution()[0]


def _parse_walltime() -> int:
    raw = os.environ.get("DFT_FORGE_WALLTIME", str(_DEFAULT_WALLTIME)).strip()
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"DFT_FORGE_WALLTIME must be an integer between {_MIN_WALLTIME} and {_MAX_WALLTIME} seconds"
        ) from exc
    if not _MIN_WALLTIME <= value <= _MAX_WALLTIME:
        raise ValueError(
            f"DFT_FORGE_WALLTIME must be between {_MIN_WALLTIME} and {_MAX_WALLTIME} seconds"
        )
    return value


def _executable_file(path: Path) -> bool:
    try:
        mode = path.stat().st_mode
    except OSError:
        return False
    return path.is_file() and bool(mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)) and os.access(path, os.X_OK)


def _local_qe_bin_dir() -> Path:
    configured = os.environ.get("DFT_FORGE_QE_BIN", "").strip()
    if configured:
        directory = Path(configured).expanduser()
        candidate = directory / "pw.x"
    else:
        found = shutil.which("pw.x")
        candidate = Path(found) if found else Path()
        directory = candidate.parent if found else Path()
    if not directory.is_dir() or not _executable_file(candidate):
        raise ValueError(
            "本地 QE 不可用：请将 DFT_FORGE_QE_BIN 设置为包含可执行 pw.x 的目录，"
            "或将 pw.x 加入 PATH；也可显式设置 DFT_FORGE_EXECUTOR=fake 用于离线测试。"
        )
    return directory.resolve()


def _stable_identity(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return "exec_" + hashlib.sha256(encoded).hexdigest()


def _config_digest(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    """Hash executable contents so provenance identifies the program, not its path."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _local_qe_executable_hashes(qe_bin_dir: Path) -> dict[str, str]:
    """Return content hashes for every supported QE executable that is present."""
    return {
        name: _file_sha256(path)
        for name in _QE_EXECUTABLES
        if _executable_file(path := qe_bin_dir / name)
    }


def _build_execution():
    from dft_forge.executor import FakeExecutor, LocalExecutor, SSHExecutor

    executor_kind = os.environ.get("DFT_FORGE_EXECUTOR", "local").strip().lower()
    walltime = _parse_walltime()
    if executor_kind == "ssh":
        host_spec = os.environ.get("DFT_FORGE_SSH_HOST", "").strip()
        if not host_spec:
            raise ValueError("DFT_FORGE_SSH_HOST is required when DFT_FORGE_EXECUTOR=ssh")
        username = os.environ.get("DFT_FORGE_SSH_USER") or None
        host = host_spec
        if "@" in host_spec and username is None:
            username, host = host_spec.rsplit("@", 1)
        key = os.environ.get("DFT_FORGE_SSH_KEY", "").strip()
        port = int(os.environ.get("DFT_FORGE_SSH_PORT", "22"))
        remote_workdir = os.environ.get("DFT_FORGE_REMOTE_WORKDIR", "/root/workspace")
        remote_qe_bin = os.environ.get("DFT_FORGE_REMOTE_QE_BIN", "/opt/qe/bin")
        scheduler = os.environ.get("DFT_FORGE_SCHEDULER", "none")
        ssh_key_sha256 = hashlib.sha256(key.encode()).hexdigest() if key else None
        identity_payload = {
            "backend": "ssh",
            "host": host,
            "username": username,
            "port": port,
            "remote_workdir": remote_workdir,
            "qe_bin_dir": remote_qe_bin,
            "scheduler": scheduler,
            "ssh_key_sha256": ssh_key_sha256,
        }
        profile = {
            "backend": "ssh",
            "simulated": False,
            "walltime_limit_sec": walltime,
            "scheduler": scheduler,
            "remote_config_sha256": _config_digest(identity_payload),
        }
        identity = _stable_identity(identity_payload)
        executor = SSHExecutor(
            host=host,
            username=username,
            ssh_key=Path(key).expanduser() if key else None,
            port=port,
            remote_workdir=remote_workdir,
            qe_bin_dir=remote_qe_bin,
            scheduler=scheduler,
            max_walltime_sec=walltime,
        )
        return executor, profile, identity
    if executor_kind not in {"", "local", "fake"}:
        raise ValueError(f"unsupported DFT_FORGE_EXECUTOR: {executor_kind!r}")
    if executor_kind == "fake":
        profile = {"backend": "fake", "simulated": True, "walltime_limit_sec": walltime}
        identity = _stable_identity({"backend": "fake"})
        return FakeExecutor(), profile, identity

    vasp_bin = os.environ.get("DFT_FORGE_VASP_BIN", "").strip() or None
    abinit_bin = os.environ.get("DFT_FORGE_ABINIT_BIN", "").strip() or None
    if abinit_bin:
        abinit_bin = str(Path(shutil.which(abinit_bin) or abinit_bin).expanduser().resolve())
    try:
        qe_bin_dir = _local_qe_bin_dir()
    except ValueError:
        if not (abinit_bin and _executable_file(Path(abinit_bin))):
            raise
        qe_bin_dir = None
    qe_executables_sha256 = _local_qe_executable_hashes(qe_bin_dir) if qe_bin_dir else {}
    abinit_executables_sha256 = {
        name: _file_sha256(path)
        for name in ("abinit", "anaddb", "mrgddb")
        if abinit_bin and _executable_file(path := (Path(abinit_bin) if name == "abinit" else Path(abinit_bin).parent / name))
    }
    abinit_pp_dir = os.environ.get("DFT_FORGE_ABINIT_PP_DIR", "").strip()
    qe_bin_config_sha256 = _config_digest({"qe_bin_dir": str(qe_bin_dir)})
    profile = {
        "backend": "local",
        "simulated": False,
        "walltime_limit_sec": walltime,
        "qe_bin_config_sha256": qe_bin_config_sha256,
        "qe_executables_sha256": qe_executables_sha256,
        "vasp_enabled": bool(vasp_bin),
        "abinit_enabled": bool(abinit_bin),
        "abinit_executables_sha256": abinit_executables_sha256,
    }
    identity = _stable_identity({
        "backend": "local",
        "qe_bin_dir": str(qe_bin_dir),
        "qe_executables_sha256": qe_executables_sha256,
        "vasp_bin": vasp_bin,
        "abinit_bin": abinit_bin,
        "abinit_executables_sha256": abinit_executables_sha256,
        "abinit_pp_dir": str(Path(abinit_pp_dir).expanduser().resolve()) if abinit_pp_dir else None,
    })
    return LocalExecutor(
        qe_bin_dir=qe_bin_dir, max_walltime_sec=walltime,
        vasp_bin=vasp_bin, abinit_bin=abinit_bin,
    ), profile, identity


def _graph_templates(args: dict) -> dict:
    engine = _get_engine(args.get("workdir"), read_only=True)
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
    method_contract = args.pop("_method_contract", None)
    try:
        metadata = {}
        if method_contract:
            from dft_forge.agent_loop.method_contract import _digest

            expected = method_contract.get("step", {}).get("template_digest")
            if expected and expected != _digest(engine.get_template(template_id).to_dict()):
                return {"error": "图模板已变化，请重新确认方法后执行"}
            row = method_contract.get("step", {})
            recommendation = row.get("recommendation") or {}
            if recommendation.get("status") == "recommended":
                from dft_forge.compiler.qe_parameters import validate_source_lock
                from dft_forge.compiler import DEFAULT_PSEUDO_DIR

                source_identity = row.get("resolved_structure_input") or row.get("structure_input") or {}
                source_lock = {"pseudopotentials": recommendation["pseudopotentials"],
                               "structure_sha256": source_identity.get("sha256")}
                validate_source_lock(source_lock, inputs.get("structure"), DEFAULT_PSEUDO_DIR)
                inputs = {**inputs, "_source_lock": source_lock}
            if row.get("convergence_plan"):
                from dft_forge.agent_loop.parameter_sweep import run_parameter_sweep

                inputs, method_contract = run_parameter_sweep(
                    inputs, method_contract, args.get("workdir"),
                    cancel_event=cancel_event, on_event=on_event,
                )
            if row.get("parameter_repair") == "require_new_confirmation":
                inputs = {**inputs, "_parameters_locked": True}
            metadata["run_metadata"] = {"method_contract": method_contract}
        run = engine.run_template(template_id, inputs, on_event=on_event, cancel_event=cancel_event, **metadata)
    except (ValueError, OSError) as exc:
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
        "metadata": run.metadata,
        "nodes": nodes,
        "outputs": _resolve_template_outputs(engine.get_template(template_id), run),
    }


def _graph_status(args: dict) -> dict:
    run_id = args.get("run_id", "")
    if not run_id:
        runs = _get_engine(args.get("workdir"), read_only=True).list_runs()
        return {"runs": runs}
    try:
        return _get_engine(args.get("workdir"), read_only=True).status(run_id)
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
    for cache_key, engine in list(_ENGINE_CACHE.items()):
        try:
            configured_workdir = getattr(engine, "base_dir", None)
            if configured_workdir is None:
                configured_workdir = cache_key[0] if isinstance(cache_key, tuple) else cache_key
            workdir = Path(configured_workdir).resolve()
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
    for engine in list(_ENGINE_CACHE.values()):
        killed += engine._cancel_executors()
    return killed


def register_graph_tools(registry) -> None:
    registry.register(
        ToolEntry(
            id="graph.templates",
            name="List Graph Templates",
            description="List available workflow graph templates (t1_vc_relax, t2_bands, t2_dos, ...).",
            category="graph",
            side_effect="read_only",
            semantic_contract="inspect available deterministic workflow templates",
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
                "Local execution requires an executable pw.x; set DFT_FORGE_EXECUTOR=fake for offline tests."
            ),
            category="graph",
            side_effect="external_execution",
            semantic_contract="execute a registered deterministic graph with a selected compute backend",
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
            side_effect="read_only",
            semantic_contract="inspect graph state and verified outputs without running nodes",
            input_schema={
                "type": "object",
                "properties": {"run_id": {"type": "string"}, "workdir": {"type": "string"}}},
            execute_fn=_graph_status,
        )
    )
