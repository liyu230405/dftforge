from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from ._helpers import _success_response


def cmd_doctor(args: argparse.Namespace) -> int:
    report: Dict[str, Any] = {
        "command": "doctor",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "architecture": platform.machine(),
        "python_version": sys.version,
        "cpu_count": os.cpu_count(),
    }

    qe_bin = shutil.which("pw.x")
    report["qe_binary"] = qe_bin
    if qe_bin:
        try:
            proc = subprocess.run(["pw.x", "--version"], capture_output=True, text=True, timeout=10)
            report["qe_version"] = proc.stdout.strip()[:200]
        except Exception:
            report["qe_version"] = "unknown"
    else:
        report["qe_version"] = None

    report["bands_x_binary"] = shutil.which("bands.x")
    report["dos_x_binary"] = shutil.which("dos.x")

    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        report["gpu"] = gpu.stdout.strip() if gpu.returncode == 0 else "none"
    except Exception:
        report["gpu"] = "none"

    report["cuda_available"] = False
    try:
        import torch
        report["cuda_available"] = torch.cuda.is_available()
        report["cuda_device_count"] = torch.cuda.device_count()
    except ImportError:
        pass

    packages = {}
    for pkg in ["numpy", "scipy", "ase", "pymatgen", "spglib", "seekpath"]:
        try:
            mod = __import__(pkg)
            packages[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            packages[pkg] = "not installed"
    report["python_packages"] = packages

    try:
        import psutil
        report["ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 1)
    except ImportError:
        report["ram_gb"] = "unknown (install psutil)"

    from dft_forge.compiler import DEFAULT_PSEUDO_DIR

    pseudo_dir = DEFAULT_PSEUDO_DIR
    report["pseudo_dir"] = str(pseudo_dir)
    report["available_pseudos"] = [
        f.name for f in pseudo_dir.iterdir() if f.suffix.lower() in {".upf", ".gth"}
    ] if pseudo_dir.exists() else []

    return _success_response(report, args.output)
