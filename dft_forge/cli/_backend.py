from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def _resolve_executor(args: argparse.Namespace, default_bin_dir: Path = Path("/opt/homebrew/bin")):
    """Return (executor, backend_name, backend_config)."""
    backend = getattr(args, "backend", "local") or "local"
    config_path = getattr(args, "backend_config", None)
    config: Dict[str, Any] = {}
    if config_path:
        config_path = Path(config_path)
        if config_path.exists():
            config = json.loads(config_path.read_text())

    if backend == "local":
        from dft_forge.executor import LocalExecutor
        executor = LocalExecutor(
            qe_bin_dir=Path(config.get("qe_bin_dir", str(default_bin_dir))),
            max_walltime_sec=int(config.get("max_walltime_sec", 600)),
            max_retries=int(config.get("max_retries", 1)),
        )
        return executor, "local", config

    if backend == "ssh":
        from dft_forge.executor import SSHExecutor
        required = ["host", "username", "remote_workdir"]
        missing = [k for k in required if k not in config]
        if missing:
            raise ValueError(f"SSH backend config missing keys: {missing}")
        executor = SSHExecutor(
            host=config["host"],
            username=config["username"],
            remote_workdir=Path(config["remote_workdir"]),
            ssh_key=config.get("ssh_key"),
            qe_bin_dir=Path(config.get("qe_bin_dir", "/opt/software/q-e/bin")),
            scheduler=config.get("scheduler", "slurm"),
            max_walltime_sec=int(config.get("max_walltime_sec", 3600)),
            max_retries=int(config.get("max_retries", 1)),
            port=int(config.get("port", 22)),
        )
        return executor, "ssh", config

    raise ValueError(f"Unsupported backend: {backend}")
