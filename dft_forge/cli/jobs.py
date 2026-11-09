from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any, Dict

from ._backend import _resolve_executor
from ._helpers import _error_response, _success_response


def _local_job_id(workdir: Path) -> str:
    return f"local_{workdir.name}_{int(time.time())}"



def cmd_job_submit(args: argparse.Namespace) -> int:
    input_file = Path(args.input)
    if not input_file.exists():
        return _error_response(f"Input file not found: {input_file}")

    executor, backend_name, backend_config = _resolve_executor(args)
    workdir = input_file.parent.resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    data: Dict[str, Any] = {
        "command": "job.submit",
        "backend": backend_name,
        "input_file": str(input_file),
        "workdir": str(workdir),
    }

    if backend_name == "local":
        result = executor.run_pw(input_file, workdir)
        job_id = getattr(result, "remote_path", None) or _local_job_id(workdir)
        data.update({
            "job_id": job_id,
            "success": result.success,
            "exit_code": result.exit_code,
            "walltime_sec": result.walltime_sec,
            "job_done": result.job_done,
            "output_files": result.output_files,
        })
        if not result.success:
            data["verifier_summary"] = "Local job reported failure"
            return _error_response("Local job failed", code=3, details=data)
        return _success_response(data, args.output)

    # SSH backend: one atomic command = upload + run + download.
    # SSHExecutor's stage/submit/status/fetch lifecycle needs in-process
    # state; CLI invocations are separate processes, so the synchronous
    # run_pw round trip is the only contract that can actually work here.
    try:
        result = executor.run_pw(input_file, workdir)
    except Exception as exc:
        return _error_response("Submit failed", details={"exception": str(exc), "backend": backend_name})

    data.update({
        "job_id": result.scheduler_job_id or result.remote_path or str(uuid.uuid4()),
        "remote_path": result.remote_path,
        "scheduler_job_id": result.scheduler_job_id,
        "success": result.success,
        "exit_code": result.exit_code,
        "walltime_sec": result.walltime_sec,
        "job_done": result.job_done,
        "output_files": result.output_files,
    })
    if not result.success:
        return _error_response("SSH job failed", code=3, details=data)
    return _success_response(data, args.output)



def cmd_job_status(args: argparse.Namespace) -> int:
    executor, backend_name, backend_config = _resolve_executor(args)
    job_id = args.job_id

    data: Dict[str, Any] = {
        "command": "job.status",
        "backend": backend_name,
        "job_id": job_id,
    }

    try:
        if backend_name == "local":
            # Local jobs are synchronous; infer from workdir if possible.
            workdir = Path(job_id) if Path(job_id).exists() else None
            if workdir and (workdir / "result.json").exists():
                data.update({
                    "state": "done",
                    "workdir": str(workdir),
                    "result_path": str(workdir / "result.json"),
                })
            else:
                data.update({"state": "unknown"})
            return _success_response(data, args.output)

        # SSH: job_id is the scheduler job id printed by submit (stateless
        # query works across processes); direct-mode jobs run synchronously
        # and leave nothing to query
        adapter = getattr(executor, "scheduler_adapter", None)
        if adapter is not None:
            status = adapter.get_job_status(job_id)
            data["state"] = status.value if status is not None else "unknown"
        else:
            data["state"] = "unknown"
            data["note"] = "direct-mode SSH jobs run synchronously; there is no persistent job state to query"
    except NotImplementedError:
        data["state"] = "unknown"
        return _success_response(data, args.output)
    except Exception as exc:
        return _error_response("Status check failed", details={"exception": str(exc), "backend": backend_name})

    return _success_response(data, args.output)



def cmd_job_cancel(args: argparse.Namespace) -> int:
    executor, backend_name, backend_config = _resolve_executor(args)
    job_id = args.job_id
    data: Dict[str, Any] = {
        "command": "job.cancel",
        "backend": backend_name,
        "job_id": job_id,
    }
    try:
        if backend_name == "local":
            data["cancelled"] = False
            data["note"] = "local backend jobs are synchronous; there is nothing to cancel"
        elif getattr(executor, "scheduler_adapter", None) is not None:
            # stateless: job_id is the scheduler job id printed by submit
            ok, msg = executor.scheduler_adapter.cancel_job(job_id)
            data["cancelled"] = bool(ok)
            if not ok:
                data["note"] = msg
        else:
            data["cancelled"] = False
            data["note"] = "direct-mode SSH jobs are synchronous; there is nothing to cancel"
    except Exception as exc:
        return _error_response("Cancel failed", details={"exception": str(exc), "backend": backend_name})
    return _success_response(data, args.output)



def cmd_job_fetch(args: argparse.Namespace) -> int:
    executor, backend_name, backend_config = _resolve_executor(args)
    job_id = args.job_id
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data: Dict[str, Any] = {
        "command": "job.fetch",
        "backend": backend_name,
        "job_id": job_id,
        "output_dir": str(output_dir),
    }

    try:
        if backend_name == "local":
            src = Path(job_id)
            if src.exists() and src.is_dir():
                data["fetched"] = str(src)
                return _success_response(data, args.output)
            data["fetched"] = str(output_dir)
            return _success_response(data, args.output)

        # SSH: stateless download — job_id is the remote job directory that
        # submit printed as remote_path
        from dft_forge.executor import JobHandle

        result = executor.fetch(JobHandle(job_id=job_id, remote_path=job_id), output_dir)
        data["fetched"] = str(result.destination)
        data["output_files"] = result.output_files
        if not result.success:
            return _error_response("Fetch failed", details={**data, "error": result.error_message})
    except NotImplementedError:
        data["fetched"] = str(output_dir)
    except Exception as exc:
        return _error_response("Fetch failed", details={"exception": str(exc), "backend": backend_name})

    return _success_response(data, args.output)
