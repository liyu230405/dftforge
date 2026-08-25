"""Minimal FastAPI backend for DFT-Forge chat UI."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from dft_forge.agent_loop import AgentLoop
from dft_forge.tools.definitions import register_default_tools
from dft_forge.tools.registry import registry

app = FastAPI(title="DFT-Forge Chat")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

register_default_tools()

agent_loop = AgentLoop()
WEB_SESSION_CTX: Dict[str, Dict[str, Any]] = {}

CLI = [sys.executable, "-m", "dft_forge.cli"]
WEB_WORKDIR = Path("/tmp/dft-forge-web")
WEB_WORKDIR.mkdir(parents=True, exist_ok=True)

FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"
app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    backend: Optional[str] = "local"
    backend_config: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    commands: List[Dict[str, Any]]
    results: List[Dict[str, Any]]


def _ensure_pythonpath() -> Path:
    return Path(__file__).resolve().parent.parent


PYTHONPATH_ENV = _ensure_pythonpath()


def _run(args: List[str], cwd: Path) -> Dict[str, Any]:
    env = {**os.environ, "PYTHONPATH": str(PYTHONPATH_ENV)}
    proc = subprocess.run(CLI + args, cwd=cwd, capture_output=True, text=True, env=env)
    payload: Dict[str, Any] = {"args": args, "returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
    try:
        if proc.stdout.strip():
            payload["json"] = json.loads(proc.stdout)
    except json.JSONDecodeError:
        pass
    return payload


def _doctor_check() -> Dict[str, Any]:
    out = _run(["doctor"], WEB_WORKDIR)
    if out["returncode"] == 0 and isinstance(out.get("json"), dict):
        return out["json"]
    return {"qe_binary": shutil.which("pw.x"), "bands_x_binary": shutil.which("bands.x"), "dos_x_binary": shutil.which("dos.x")}


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/doctor")
def doctor() -> Dict[str, Any]:
    return _doctor_check()


@app.get("/tools")
def list_tools() -> Dict[str, Any]:
    return {"tools": agent_loop.list_tools()}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "app.html")


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    session_id = req.session_id or str(uuid.uuid4())
    session_dir = WEB_WORKDIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    ctx = WEB_SESSION_CTX.setdefault(session_id, {})
    result = await agent_loop.run(req.message.strip(), session_dir, ctx=ctx)
    ctx.update(result.get("ctx") or {})
    return ChatResponse(
        session_id=session_id,
        reply=result["reply"],
        commands=result["commands"],
        results=result["results"],
    )
