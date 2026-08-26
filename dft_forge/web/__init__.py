"""Minimal FastAPI backend for DFT-Forge chat UI."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
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
WEB_WORKDIR = Path(os.environ.get("DFT_FORGE_WEB_WORKDIR", "/tmp/dft-forge-web"))
WEB_WORKDIR.mkdir(parents=True, exist_ok=True)

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
_LLM_ENV_KEYS = ("DFT_FORGE_LLM_PROVIDER", "DFT_FORGE_LLM_BASE_URL", "DFT_FORGE_LLM_MODEL", "DFT_FORGE_LLM_API_KEY")


def _mask(secret: str) -> str:
    if len(secret) <= 8:
        return "*" * len(secret)
    return f"{secret[:4]}…{secret[-4:]}"


def _update_env_file(values: Dict[str, str]) -> None:
    """Persist DFT_FORGE_LLM_* settings into .env, preserving unrelated lines."""
    lines: List[str] = []
    if ENV_FILE.exists():
        lines = ENV_FILE.read_text().splitlines()
    seen = set()
    out: List[str] = []
    for line in lines:
        m = re.match(r"(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=", line)
        key = m.group(1) if m else None
        if key in values:
            seen.add(key)
            out.append(f"{key}={values[key]}")
        else:
            out.append(line)
    for key, val in values.items():
        if key not in seen:
            out.append(f"{key}={val}")
    ENV_FILE.write_text("\n".join(out) + "\n")


class ConfigRequest(BaseModel):
    provider: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None


def _config_state() -> Dict[str, Any]:
    provider = os.environ.get("DFT_FORGE_LLM_PROVIDER", "dummy").lower()
    key = os.environ.get("DFT_FORGE_LLM_API_KEY", "")
    ready = provider in ("openai", "openai_compatible", "openai-compatible") and bool(key)
    return {
        "provider": provider,
        "base_url": os.environ.get("DFT_FORGE_LLM_BASE_URL", ""),
        "model": os.environ.get("DFT_FORGE_LLM_MODEL", ""),
        "has_api_key": bool(key),
        "api_key_hint": _mask(key) if key else "",
        "llm_ready": ready,
    }


@app.get("/api/config")
def get_config() -> Dict[str, Any]:
    return _config_state()


@app.post("/api/config")
def save_config(req: ConfigRequest) -> Dict[str, Any]:
    updates: Dict[str, str] = {}
    if req.provider is not None:
        updates["DFT_FORGE_LLM_PROVIDER"] = req.provider
    if req.base_url is not None:
        updates["DFT_FORGE_LLM_BASE_URL"] = req.base_url.strip()
    if req.model is not None:
        updates["DFT_FORGE_LLM_MODEL"] = req.model.strip()
    if req.api_key is not None:
        updates["DFT_FORGE_LLM_API_KEY"] = req.api_key.strip()
    if updates:
        _update_env_file(updates)
        os.environ.update(updates)
        agent_loop.reset_llm()
    return _config_state()


def _session_file(session_id: str) -> Path:
    return WEB_WORKDIR / session_id / "session.json"


def _load_session(session_id: str) -> Dict[str, Any]:
    """Session memory: {ctx, messages}. Reloaded from disk so restarts survive."""
    if session_id in WEB_SESSION_CTX:
        return WEB_SESSION_CTX[session_id]
    sf = _session_file(session_id)
    if sf.exists():
        try:
            data = json.loads(sf.read_text())
            WEB_SESSION_CTX[session_id] = data
            return data
        except (OSError, ValueError):
            pass
    fresh = {"ctx": {}, "messages": [], "created_at": time.time(), "title": ""}
    WEB_SESSION_CTX[session_id] = fresh
    return fresh


def _save_session(session_id: str) -> None:
    session = WEB_SESSION_CTX.get(session_id)
    if session is None:
        return
    sf = _session_file(session_id)
    sf.parent.mkdir(parents=True, exist_ok=True)
    sf.write_text(json.dumps(session, ensure_ascii=False, default=str))

FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"


class NoCacheStaticFiles(StaticFiles):
    """Dev server: always revalidate JS modules so fixes reach the browser."""

    def file_response(self, *args, **kwargs):
        resp = super().file_response(*args, **kwargs)
        resp.headers["Cache-Control"] = "no-cache"
        return resp


app.mount("/frontend", NoCacheStaticFiles(directory=FRONTEND_DIR), name="frontend")


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


@app.get("/api/sessions")
def list_sessions() -> Dict[str, Any]:
    sessions = []
    for sf in sorted(WEB_WORKDIR.glob("*/session.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(sf.read_text())
        except (OSError, ValueError):
            continue
        if not data.get("messages"):
            continue  # never show empty/aborted sessions
        sessions.append({
            "session_id": sf.parent.name,
            "title": data.get("title") or "未命名会话",
            "n_messages": len(data["messages"]),
            "updated_at": sf.stat().st_mtime,
        })
    return {"sessions": sessions}


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str) -> Dict[str, Any]:
    session = _load_session(session_id)
    return {
        "session_id": session_id,
        "title": session.get("title") or "",
        "messages": session.get("messages", []),
        "ctx": session.get("ctx", {}),
    }


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str) -> Dict[str, Any]:
    WEB_SESSION_CTX.pop(session_id, None)
    d = WEB_WORKDIR / session_id
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
    return {"deleted": session_id}


def _history_for(session: Dict[str, Any], limit: int = 8) -> List[Dict[str, Any]]:
    """Recent turns for the planner: pronouns like "它/再算一次" resolve here."""
    return [
        {"role": m.get("role"), "text": str(m.get("text", ""))[:400]}
        for m in (session.get("messages") or [])[-limit:]
        if m.get("text")
    ]


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    session_id = req.session_id or str(uuid.uuid4())
    session_dir = WEB_WORKDIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    session = _load_session(session_id)
    ctx = session.setdefault("ctx", {})
    messages = session.setdefault("messages", [])
    if not session.get("title"):
        session["title"] = req.message.strip()[:40]

    result = await agent_loop.run(req.message.strip(), session_dir, ctx=ctx, history=_history_for(session))
    ctx.update(result.get("ctx") or {})

    messages.append({"role": "user", "text": req.message.strip(), "ts": time.time()})
    messages.append({
        "role": "agent",
        "text": result["reply"],
        "commands": result["commands"],
        "results": result["results"],
        "chain": result.get("chain", []),
        "ts": time.time(),
    })
    session["updated_at"] = time.time()
    _save_session(session_id)

    return ChatResponse(
        session_id=session_id,
        reply=result["reply"],
        commands=result["commands"],
        results=result["results"],
    )


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """SSE stream of the agent's execution chain: plan → steps → nodes → reply."""
    session_id = req.session_id or str(uuid.uuid4())
    session_dir = WEB_WORKDIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    session = _load_session(session_id)
    ctx = session.setdefault("ctx", {})
    messages = session.setdefault("messages", [])
    if not session.get("title"):
        session["title"] = req.message.strip()[:40]
    message = req.message.strip()

    q: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def emit(ev) -> None:
        loop.call_soon_threadsafe(q.put_nowait, ev)

    async def worker() -> None:
        try:
            result = await agent_loop.run(
                message, session_dir, ctx=ctx, history=_history_for(session), on_event=emit
            )
            ctx.update(result.get("ctx") or {})
            messages.append({"role": "user", "text": message, "ts": time.time()})
            messages.append({
                "role": "agent",
                "text": result["reply"],
                "commands": result.get("commands", []),
                "results": result.get("results", []),
                "chain": result.get("chain", []),
                "ts": time.time(),
            })
            session["updated_at"] = time.time()
            _save_session(session_id)
        except Exception as exc:  # noqa: BLE001
            emit({"type": "error", "message": str(exc)})
        finally:
            emit(None)

    task = asyncio.create_task(worker())

    async def gen():
        yield "data: " + json.dumps({"type": "start", "session_id": session_id}, ensure_ascii=False) + "\n\n"
        while True:
            ev = await q.get()
            if ev is None:
                break
            yield "data: " + json.dumps(ev, ensure_ascii=False, default=str) + "\n\n"
        await task
        yield "data: " + json.dumps({"type": "done", "session_id": session_id}, ensure_ascii=False) + "\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
