"""Command runners: execute a shell command locally or over SSH.

Commands are wrapped in ``bash -l -c`` (login shell) so module-managed
binaries (sbatch, qsub, pw.x) are on PATH — same trick CatGo uses.
The runner is the only place remote execution happens, so tests inject a
fake runner and no network is ever touched.
"""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class CommandResult:
    exit_code: int
    stdout: str = ""
    stderr: str = ""

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


class CommandRunner:
    """Runs a command through ``bash -l -c`` on some machine."""

    def run(self, cmd: str, timeout: float = 60.0) -> CommandResult:
        raise NotImplementedError


class LocalRunner(CommandRunner):
    def run(self, cmd: str, timeout: float = 60.0) -> CommandResult:
        proc = subprocess.run(
            ["bash", "-l", "-c", cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return CommandResult(proc.returncode, proc.stdout, proc.stderr)


class SSHRunner(CommandRunner):
    """Runs commands on a remote host via the system ``ssh`` binary.

    Uses ssh config aliases/agent/keys — no passwords are handled here.
    ``jump_host`` chains through a ProxyJump-style hop.
    """

    def __init__(
        self,
        host: str,
        *,
        username: Optional[str] = None,
        port: int = 22,
        key_file: Optional[str] = None,
        jump_host: Optional[str] = None,
        connect_timeout: int = 15,
    ):
        self.host = host
        self.username = username
        self.port = port
        self.key_file = key_file
        self.jump_host = jump_host
        self.connect_timeout = connect_timeout

    def _ssh_argv(self, remote_cmd: str) -> List[str]:
        argv = ["ssh", "-o", f"ConnectTimeout={self.connect_timeout}"]
        if self.port != 22:
            argv += ["-p", str(self.port)]
        if self.key_file:
            argv += ["-i", self.key_file]
        if self.jump_host:
            argv += ["-J", self.jump_host]
        target = f"{self.username}@{self.host}" if self.username else self.host
        argv += [target, "bash -l -c " + shlex.quote(remote_cmd)]
        return argv

    def run(self, cmd: str, timeout: float = 120.0) -> CommandResult:
        try:
            proc = subprocess.run(
                self._ssh_argv(cmd), capture_output=True, text=True, timeout=timeout
            )
        except subprocess.TimeoutExpired:
            return CommandResult(-1, stderr=f"ssh timed out after {timeout}s")
        return CommandResult(proc.returncode, proc.stdout, proc.stderr)
