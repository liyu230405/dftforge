"""DFT-Forge tool definitions."""

from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout, redirect_stderr
from typing import Any

from dft_forge.tools.models import ToolEntry
from dft_forge.tools.registry import registry


def _call_cmd(cmd_fn, ns) -> dict:
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
        rc = cmd_fn(ns)
    payload: dict = {"returncode": rc, "stdout": stdout_buf.getvalue(), "stderr": stderr_buf.getvalue()}
    try:
        text = stdout_buf.getvalue().strip()
        if text:
            payload["json"] = json.loads(text)
    except json.JSONDecodeError:
        pass
    return payload


def _structure_import(args: dict) -> dict:
    from dft_forge.cli import cmd_structure_import
    import argparse
    ns = argparse.Namespace(
        source=args.get("source", ""),
        format=args.get("format"),
        fractional=args.get("fractional", False),
        min_nn=args.get("min_nn", 0.8),
        write_cif=args.get("write_cif"),
        output=args.get("output"),
    )
    return _call_cmd(cmd_structure_import, ns)


def _input_build(args: dict) -> dict:
    from dft_forge.cli import cmd_input_build
    import argparse
    ns = argparse.Namespace(
        structure=args.get("structure"),
        structure_format=args.get("structure_format"),
        fractional=args.get("fractional", False),
        material=args.get("material"),
        type=args.get("type", "scf"),
        prefix=args.get("prefix"),
        ecutwfc=args.get("ecutwfc"),
        ecutrho=args.get("ecutrho"),
        kpoints=args.get("kpoints"),
        conv_thr=args.get("conv_thr", 1.0e-8),
        nstep=args.get("nstep", 200),
        pseudo_dir=args.get("pseudo_dir"),
        output=args.get("output"),
    )
    return _call_cmd(cmd_input_build, ns)


def _job_submit(args: dict) -> dict:
    from dft_forge.cli import cmd_job_submit
    import argparse
    ns = argparse.Namespace(
        input=args.get("input", ""),
        backend=args.get("backend", "local"),
        backend_config=args.get("backend_config"),
        output=args.get("output"),
    )
    return _call_cmd(cmd_job_submit, ns)


def _job_status(args: dict) -> dict:
    from dft_forge.cli import cmd_job_status
    import argparse
    ns = argparse.Namespace(
        job_id=args.get("job_id", ""),
        backend=args.get("backend", "local"),
        backend_config=args.get("backend_config"),
        output=args.get("output"),
    )
    return _call_cmd(cmd_job_status, ns)


def _result_parse(args: dict) -> dict:
    from dft_forge.cli import cmd_result_parse
    import argparse
    ns = argparse.Namespace(
        input=args.get("input", ""),
        type=args.get("type", "vc-relax"),
        xml_path=args.get("xml_path"),
        output=args.get("output"),
    )
    return _call_cmd(cmd_result_parse, ns)


def _result_verify(args: dict) -> dict:
    from dft_forge.cli import cmd_result_verify
    import argparse
    ns = argparse.Namespace(
        input=args.get("input", ""),
        task_type=args.get("task_type"),
        output=args.get("output"),
    )
    return _call_cmd(cmd_result_verify, ns)


def _ledger_query(args: dict) -> dict:
    from dft_forge.cli import cmd_ledger_query
    import argparse
    ns = argparse.Namespace(
        database=args.get("database", "ledger.db"),
        task_id=args.get("task_id"),
        limit=args.get("limit", 20),
        output=args.get("output"),
    )
    return _call_cmd(cmd_ledger_query, ns)


def _doctor(args: dict) -> dict:
    from dft_forge.cli import cmd_doctor
    import argparse
    ns = argparse.Namespace(output=args.get("output"))
    return _call_cmd(cmd_doctor, ns)


def _structure_generate(args: dict) -> dict:
    from dft_forge.cli import cmd_structure_generate
    import argparse
    ns = argparse.Namespace(
        source=args.get("source", ""),
        output=args.get("output"),
    )
    return _call_cmd(cmd_structure_generate, ns)


def _structure_build2d(args: dict) -> dict:
    from dft_forge.cli import cmd_structure_build2d
    import argparse
    dopants = args.get("dopants") or []
    ns = argparse.Namespace(
        kind=args.get("kind", "graphene"),
        supercell=args.get("supercell", "1x1"),
        vacancy=args.get("vacancy"),
        dopant=[f"{d['element']}@{d['index']}" for d in dopants] or None,
        adsorb=(
            f"{args['adsorb']['element']}@{args['adsorb'].get('site', 'top')}"
            if args.get("adsorb")
            else None
        ),
        height=args.get("height", 1.5),
        vacuum=args.get("vacuum", 15.0),
        output=args.get("output"),
    )
    return _call_cmd(cmd_structure_build2d, ns)


def _structure_analyze(args: dict) -> dict:
    from dft_forge.cli import cmd_structure_analyze
    import argparse
    # clamp: LLMs invent destructive values (e.g. min_nn=4 rejects every
    # real bond — C-C is 1.42 Å); this is a validity floor, not a physics
    # threshold, so cap it below the shortest real bonds (H2 0.74 Å)
    min_nn = min(max(float(args.get("min_nn", 0.8) or 0.8), 0.3), 1.2)
    fractional = str(args.get("fractional", False)).lower() in ("true", "1", "yes")
    ns = argparse.Namespace(
        source=args.get("source", ""),
        format=args.get("format"),
        fractional=fractional,
        min_nn=min_nn,
        pair_types=args.get("pair_types"),
        output=args.get("output"),
    )
    return _call_cmd(cmd_structure_analyze, ns)


def register_default_tools() -> None:
    tools = [
        ToolEntry(
            id="structure.generate",
            name="Generate Structure",
            description="Generate a standard crystal structure for known materials like NaCl, MgO, Si, Al.",
            category="structure",
            input_schema={
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "output": {"type": "string"},
                },
                "required": ["source"],
            },
            execute_fn=_structure_generate,
        ),
        ToolEntry(
            id="structure.build2d",
            name="Build 2D Material",
            description=(
                "Build a 2D material (graphene or h-BN monolayer) with optional supercell, "
                "doping, vacancy, and adsorbate on top/bridge/hollow site. "
                "Example: {kind: 'graphene', supercell: '4x4', dopants: [{index: 0, element: 'N'}], "
                "adsorb: {element: 'O', site: 'hollow'}}"
            ),
            category="structure",
            input_schema={
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["graphene", "bn"]},
                    "supercell": {"type": "string", "description": "e.g. '3x3' or '4x4'"},
                    "vacancy": {"type": "integer", "description": "atom index to remove"},
                    "dopants": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"index": {"type": "integer"}, "element": {"type": "string"}},
                        },
                    },
                    "adsorb": {
                        "type": "object",
                        "properties": {
                            "element": {"type": "string"},
                            "site": {"type": "string", "enum": ["top", "bridge", "hollow"]},
                            "height": {"type": "number"},
                        },
                    },
                    "height": {"type": "number"},
                    "vacuum": {"type": "number"},
                    "output": {"type": "string"},
                },
                "required": ["kind"],
            },
            execute_fn=_structure_build2d,
        ),
        ToolEntry(
            id="structure.import",
            name="Import Structure",
            description="Import and validate a structure from CIF, POSCAR, XYZ, QE input, or explicit dict.",
            category="structure",
            input_schema={
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "format": {"type": "string", "enum": ["cif", "poscar", "xyz", "qe_input", "explicit"]},
                    "fractional": {"type": "boolean"},
                    "min_nn": {"type": "number"},
                    "write_cif": {"type": "string"},
                    "output": {"type": "string"},
                },
                "required": ["source"],
            },
            execute_fn=_structure_import,
        ),
        ToolEntry(
            id="structure.analyze",
            name="Analyze Structure",
            description="Analyze a structure for formula, cell info, bond lengths, and pair distances.",
            category="structure",
            input_schema={
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "format": {"type": "string", "enum": ["cif", "poscar", "xyz", "qe_input", "explicit"]},
                    "fractional": {"type": "boolean"},
                    "min_nn": {"type": "number"},
                    "pair_types": {"type": "string"},
                    "output": {"type": "string"},
                },
                "required": ["source"],
            },
            execute_fn=_structure_analyze,
        ),
        ToolEntry(
            id="input.build",
            name="Build QE Input",
            description="Build a Quantum ESPRESSO input file for a known material or user structure.",
            category="input",
            input_schema={
                "type": "object",
                "properties": {
                    "structure": {"type": "string"},
                    "structure_format": {"type": "string"},
                    "fractional": {"type": "boolean"},
                    "material": {"type": "string"},
                    "type": {"type": "string", "enum": ["scf", "vc-relax", "relax", "bands", "dos"]},
                    "prefix": {"type": "string"},
                    "ecutwfc": {"type": "number"},
                    "ecutrho": {"type": "number"},
                    "kpoints": {"type": "array", "items": {"type": "integer"}},
                    "conv_thr": {"type": "number"},
                    "nstep": {"type": "integer"},
                    "pseudo_dir": {"type": "string"},
                    "output": {"type": "string"},
                },
                "required": ["type"],
            },
            execute_fn=_input_build,
        ),
        ToolEntry(
            id="job.submit",
            name="Submit Job",
            description="Submit a QE job on the selected backend.",
            category="job",
            input_schema={
                "type": "object",
                "properties": {
                    "input": {"type": "string"},
                    "backend": {"type": "string", "enum": ["local", "ssh"]},
                    "backend_config": {"type": "object"},
                    "output": {"type": "string"},
                },
                "required": ["input"],
            },
            execute_fn=_job_submit,
        ),
        ToolEntry(
            id="job.status",
            name="Job Status",
            description="Check job status by job id.",
            category="job",
            input_schema={
                "type": "object",
                "properties": {
                    "job_id": {"type": "string"},
                    "backend": {"type": "string", "enum": ["local", "ssh"]},
                    "backend_config": {"type": "object"},
                    "output": {"type": "string"},
                },
                "required": ["job_id"],
            },
            execute_fn=_job_status,
        ),
        ToolEntry(
            id="result.parse",
            name="Parse Result",
            description="Parse QE output into structured JSON.",
            category="result",
            input_schema={
                "type": "object",
                "properties": {
                    "input": {"type": "string"},
                    "type": {"type": "string", "enum": ["vc-relax", "scf", "nscf", "bands", "dos"]},
                    "xml_path": {"type": "string"},
                    "output": {"type": "string"},
                },
                "required": ["input", "type"],
            },
            execute_fn=_result_parse,
        ),
        ToolEntry(
            id="result.verify",
            name="Verify Result",
            description="Run scientific verification on parsed result JSON.",
            category="result",
            input_schema={
                "type": "object",
                "properties": {
                    "input": {"type": "string"},
                    "task_type": {"type": "string", "enum": ["T1", "T2"]},
                    "output": {"type": "string"},
                },
                "required": ["input"],
            },
            execute_fn=_result_verify,
        ),
        ToolEntry(
            id="ledger.query",
            name="Query Ledger",
            description="Query evidence ledger runs.",
            category="ledger",
            input_schema={
                "type": "object",
                "properties": {
                    "database": {"type": "string"},
                    "task_id": {"type": "string"},
                    "limit": {"type": "integer"},
                    "output": {"type": "string"},
                },
            },
            execute_fn=_ledger_query,
        ),
        ToolEntry(
            id="doctor",
            name="Environment Doctor",
            description="Probe QE, pseudopotentials, packages, and GPU environment.",
            category="general",
            input_schema={
                "type": "object",
                "properties": {
                    "output": {"type": "string"},
                },
            },
            execute_fn=_doctor,
        ),
    ]

    for tool in tools:
        registry.register(tool)

    from dft_forge.tools.graph_tools import register_graph_tools

    register_graph_tools(registry)
