"""Render SLURM (sbatch) and PBS (qsub) job scripts.

Parameter resolution order:
    node params > job_defaults > built-in fallbacks.
Optional directives with empty values are stripped.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

ENGINE_COMMANDS = {
    "qe": "srun pw.x -in {input_file} > {output_file}",
    "pw": "srun pw.x -in {input_file} > {output_file}",
}

DEFAULTS: Dict[str, Any] = {
    "nodes": 1,
    "ntasks": 8,
    "cpus_per_task": 1,
    "walltime": "24:00:00",
    "partition": "",
    "account": "",
    "memory": "",
    "module_loads": "",
    "env_setup": "",
}

SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={{job_name}}
#SBATCH --nodes={{nodes}}
#SBATCH --ntasks={{ntasks}}
#SBATCH --cpus-per-task={{cpus_per_task}}
#SBATCH --time={{walltime}}
#SBATCH --partition={{partition}}
#SBATCH --account={{account}}
#SBATCH --mem={{memory}}
#SBATCH --output={{job_name}}-%j.out
#SBATCH --error={{job_name}}-%j.err

{{module_loads}}
{{env_setup}}

cd {{work_dir}}
{{run_command}}
"""

QSUB_TEMPLATE = """#!/bin/bash
#PBS -N {{job_name}}
#PBS -l nodes={{nodes}}:ppn={{cpus_per_task}}
#PBS -l walltime={{walltime}}
#PBS -q {{partition}}
#PBS -o {{job_name}}-$PBS_JOBID.out
#PBS -e {{job_name}}-$PBS_JOBID.err

{{module_loads}}
{{env_setup}}

cd {{work_dir}}
{{run_command}}
"""

_EMPTY_DIRECTIVE = re.compile(r"^#(SBATCH|PBS)\s+\S+\s*=\s*$")


def _resolve(params: Dict[str, Any], job_defaults: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = dict(DEFAULTS)
    if job_defaults:
        merged.update({k: v for k, v in job_defaults.items() if v not in (None, "")})
    int_alias = {"ppn": "cpus_per_task"}
    for key in ("nodes", "ntasks", "cpus_per_task", "ppn"):
        if key in params and params[key] not in (None, ""):
            merged[int_alias.get(key, key)] = int(params[key])
    for key in ("walltime", "time_limit", "partition", "queue", "account", "memory",
                "module_loads", "env_setup"):
        if key in params and params[key] not in (None, ""):
            merged[{"time_limit": "walltime", "queue": "partition"}.get(key, key)] = params[key]
    if job_defaults:
        for key in ("walltime", "time_limit", "partition", "queue", "account", "memory",
                    "module_loads", "env_setup"):
            if key in job_defaults and job_defaults[key] not in (None, "") and key not in params:
                merged[{"time_limit": "walltime", "queue": "partition"}.get(key, key)] = job_defaults[key]
    return merged


def _run_command(params: Dict[str, Any], engine: str) -> str:
    if params.get("run_command"):
        return str(params["run_command"])
    template = ENGINE_COMMANDS.get(engine, "srun {engine_placeholder}")
    if engine in ENGINE_COMMANDS:
        return template.format(
            input_file=params.get("input_file", "pw.in"),
            output_file=params.get("output_file", "pw.out"),
        )
    return f"srun {engine}"


def _render(template: str, values: Dict[str, Any]) -> str:
    text = template
    for key, val in values.items():
        text = text.replace("{{" + key + "}}", str(val))
    lines = [ln for ln in text.splitlines() if not _EMPTY_DIRECTIVE.match(ln.strip())]
    return "\n".join(lines).strip() + "\n"


def render_sbatch(
    params: Dict[str, Any],
    job_defaults: Optional[Dict[str, Any]] = None,
    *,
    engine: str = "qe",
    job_name: str = "dftforge",
    work_dir: str = ".",
) -> str:
    resolved = _resolve(params, job_defaults)
    resolved["run_command"] = _run_command(params, engine)
    resolved["job_name"] = job_name
    resolved["work_dir"] = work_dir
    return _render(SBATCH_TEMPLATE, resolved)


def render_qsub(
    params: Dict[str, Any],
    job_defaults: Optional[Dict[str, Any]] = None,
    *,
    engine: str = "qe",
    job_name: str = "dftforge",
    work_dir: str = ".",
) -> str:
    resolved = _resolve(params, job_defaults)
    resolved["run_command"] = _run_command(params, engine)
    resolved["job_name"] = job_name
    resolved["work_dir"] = work_dir
    return _render(QSUB_TEMPLATE, resolved)
