"""Minimal FastAPI backend for DFT-Forge chat UI."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from dft_forge.agent_loop import AgentLoop
from dft_forge.tools.definitions import register_default_tools
from dft_forge.tools.registry import registry

app = FastAPI(title="DFT-Forge Chat")

_CORS_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "DFT_FORGE_WEB_ORIGINS",
        "http://127.0.0.1:8000,http://localhost:8000",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)

register_default_tools()

agent_loop = AgentLoop()
WEB_SESSION_CTX: Dict[str, Dict[str, Any]] = {}
# session_id -> {"task": asyncio.Task, "cancel": threading.Event, "ts": float}
# tracks in-flight chat runs so POST /api/sessions/{id}/cancel can stop them
WEB_ACTIVE_RUNS: Dict[str, Dict[str, Any]] = {}

CLI = [sys.executable, "-m", "dft_forge.cli"]
# 默认落在项目根目录 sessions/ 下：/tmp 会随系统重启清空，计算结果必须持久化
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEB_WORKDIR = Path(os.environ.get("DFT_FORGE_WEB_WORKDIR", _PROJECT_ROOT / "sessions"))
WEB_WORKDIR.mkdir(parents=True, exist_ok=True)

ENV_FILE = _PROJECT_ROOT / ".env"
_LLM_ENV_KEYS = (
    "DFT_FORGE_LLM_PROVIDER", "DFT_FORGE_LLM_VENDOR", "DFT_FORGE_LLM_BASE_URL",
    "DFT_FORGE_LLM_MODEL", "DFT_FORGE_LLM_API_KEY",
)

LLM_PROVIDERS = [
    {"id": "rules", "name": "内置规则", "base_url": "", "model": "", "needs_key": False},
    {"id": "openai", "name": "OpenAI", "base_url": "https://api.openai.com/v1", "model": "", "needs_key": True},
    {"id": "deepseek", "name": "DeepSeek", "base_url": "https://api.deepseek.com/v1", "model": "deepseek-chat", "needs_key": True},
    {"id": "qwen", "name": "通义千问", "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "model": "qwen-plus", "needs_key": True},
    {"id": "moonshot", "name": "Moonshot", "base_url": "https://api.moonshot.cn/v1", "model": "", "needs_key": True},
    {"id": "ollama", "name": "Ollama", "base_url": "http://127.0.0.1:11434/v1", "model": "", "needs_key": False},
    {"id": "custom", "name": "OpenAI Compatible", "base_url": "", "model": "", "needs_key": False},
]


def _update_env_file(values: Dict[str, str]) -> None:
    """Persist settings into .env, preserving unrelated lines."""
    for key, value in values.items():
        if "\n" in value or "\r" in value:
            raise HTTPException(status_code=400, detail=f"Invalid newline in {key}")
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
    payload = "\n".join(out) + "\n"
    ENV_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=ENV_FILE.parent, prefix=".env-", delete=False
        ) as tmp:
            tmp.write(payload)
            tmp_path = Path(tmp.name)
        os.replace(tmp_path, ENV_FILE)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


class ConfigRequest(BaseModel):
    provider: Optional[str] = None
    vendor: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None


def _config_state() -> Dict[str, Any]:
    provider = os.environ.get("DFT_FORGE_LLM_PROVIDER", "dummy").lower()
    key = os.environ.get("DFT_FORGE_LLM_API_KEY", "")
    vendor = os.environ.get("DFT_FORGE_LLM_VENDOR", "custom").lower()
    ready = provider in ("openai", "openai_compatible", "openai-compatible") and (
        bool(key) or vendor == "ollama"
    )
    return {
        "provider": provider,
        "vendor": vendor,
        "base_url": os.environ.get("DFT_FORGE_LLM_BASE_URL", ""),
        "model": os.environ.get("DFT_FORGE_LLM_MODEL", ""),
        "has_api_key": bool(key),
        # Never send any portion of a credential to the browser.  Keep the
        # legacy field as a null value for clients that still deserialize it.
        "api_key_hint": None,
        "llm_ready": ready,
    }


@app.get("/api/config")
def get_config() -> Dict[str, Any]:
    return {**_config_state(), "compute": _compute_state(), "providers": LLM_PROVIDERS}


@app.post("/api/config")
def save_config(req: ConfigRequest) -> Dict[str, Any]:
    updates: Dict[str, str] = {}
    if req.provider is not None:
        if req.provider not in {"dummy", "openai", "openai_compatible", "openai-compatible"}:
            raise HTTPException(status_code=400, detail="Unsupported LLM provider")
        updates["DFT_FORGE_LLM_PROVIDER"] = req.provider
    if req.vendor is not None:
        if req.vendor.strip().lower() not in {item["id"] for item in LLM_PROVIDERS}:
            raise HTTPException(status_code=400, detail="Unsupported LLM vendor")
        updates["DFT_FORGE_LLM_VENDOR"] = req.vendor.strip().lower()
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


class ConfigTestRequest(ConfigRequest):
    pass


@app.post("/api/config/test")
async def test_config(req: ConfigTestRequest) -> Dict[str, Any]:
    """Make one tiny real request without persisting the submitted settings."""
    from urllib.parse import urlparse

    from dft_forge.llm import OpenAICompatProvider

    current = _config_state()
    base_url = (req.base_url or current.get("base_url") or "").strip().rstrip("/")
    model = (req.model or current.get("model") or "").strip()
    key = req.api_key if req.api_key is not None else os.environ.get("DFT_FORGE_LLM_API_KEY", "")
    vendor = (req.vendor or current.get("vendor") or "custom").lower()
    parsed_url = urlparse(base_url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise HTTPException(status_code=400, detail="请输入有效的 http(s) API 地址")
    if not model:
        raise HTTPException(status_code=400, detail="请填写模型名称")
    if not key and vendor != "ollama":
        raise HTTPException(status_code=400, detail="请填写 API Key")
    provider = OpenAICompatProvider(
        base_url=base_url,
        api_key=key or "ollama",
        model=model,
        timeout=20.0,
        max_attempts=1,
    )
    started = time.perf_counter()
    try:
        reply = await asyncio.to_thread(provider.chat, "Reply with exactly: OK", "connection test")
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)[:500]) from exc
    return {
        "ok": True,
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "model": model,
        "reply": str(reply).strip()[:80],
    }


class ComputeConfigRequest(BaseModel):
    executor: str = "local"
    qe_bin: Optional[str] = None
    ssh_host: Optional[str] = None
    ssh_key: Optional[str] = None
    scheduler: str = "none"
    remote_workdir: Optional[str] = None
    remote_qe_bin: Optional[str] = None
    walltime: int = 1800


def _compute_state() -> Dict[str, Any]:
    key_path = os.environ.get("DFT_FORGE_SSH_KEY", "")
    return {
        "executor": os.environ.get("DFT_FORGE_EXECUTOR", "local"),
        "qe_bin": os.environ.get("DFT_FORGE_QE_BIN", ""),
        "ssh_host": os.environ.get("DFT_FORGE_SSH_HOST", ""),
        "has_ssh_key": bool(key_path),
        "ssh_key_name": Path(key_path).name if key_path else "",
        "scheduler": os.environ.get("DFT_FORGE_SCHEDULER", "none"),
        "remote_workdir": os.environ.get("DFT_FORGE_REMOTE_WORKDIR", "/root/workspace"),
        "remote_qe_bin": os.environ.get("DFT_FORGE_REMOTE_QE_BIN", "/opt/qe/bin"),
        "walltime": int(os.environ.get("DFT_FORGE_WALLTIME", "1800")),
    }


@app.post("/api/config/compute")
def save_compute_config(req: ComputeConfigRequest) -> Dict[str, Any]:
    if WEB_ACTIVE_RUNS:
        raise HTTPException(status_code=409, detail="有计算正在运行，停止后才能切换算力后端")
    executor = req.executor.strip().lower()
    scheduler = req.scheduler.strip().lower()
    if executor not in {"local", "fake", "ssh"}:
        raise HTTPException(status_code=400, detail="executor must be local, fake or ssh")
    if scheduler not in {"none", "slurm", "pbs"}:
        raise HTTPException(status_code=400, detail="scheduler must be none, slurm or pbs")
    if executor == "ssh" and not (req.ssh_host or "").strip():
        raise HTTPException(status_code=400, detail="SSH 模式必须填写主机")
    if not 30 <= req.walltime <= 604800:
        raise HTTPException(status_code=400, detail="walltime must be between 30 and 604800 seconds")
    updates = {
        "DFT_FORGE_EXECUTOR": executor,
        "DFT_FORGE_QE_BIN": (req.qe_bin or "").strip(),
        "DFT_FORGE_SSH_HOST": (req.ssh_host or "").strip(),
        "DFT_FORGE_SCHEDULER": scheduler,
        "DFT_FORGE_REMOTE_WORKDIR": (req.remote_workdir or "/root/workspace").strip(),
        "DFT_FORGE_REMOTE_QE_BIN": (req.remote_qe_bin or "/opt/qe/bin").strip(),
        "DFT_FORGE_WALLTIME": str(req.walltime),
    }
    if req.ssh_key is not None:
        updates["DFT_FORGE_SSH_KEY"] = req.ssh_key.strip()
    _update_env_file(updates)
    os.environ.update(updates)
    from dft_forge.tools.graph_tools import reset_engine_cache

    reset_engine_cache()
    return _compute_state()


_SAFE_SESSION_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _session_dir(session_id: str) -> Path:
    """Validated session directory.

    session_id arrives from the client and lands in filesystem paths, so it
    must be a safe token (no separators, no '..', bounded length) AND the
    resolved directory must stay inside WEB_WORKDIR — a symlink planted at
    sessions/<id> must not turn delete/download into an arbitrary-path tool.
    """
    if not isinstance(session_id, str) or not _SAFE_SESSION_ID.match(session_id):
        raise HTTPException(status_code=400, detail="Invalid session id")
    root = WEB_WORKDIR.resolve()
    d = (WEB_WORKDIR / session_id).resolve()
    if d.parent != root:
        raise HTTPException(status_code=400, detail="Invalid session id")
    return d


def _session_file(session_id: str) -> Path:
    return _session_dir(session_id) / "session.json"


def _inside_session(session_dir: Path, value: str) -> Path:
    """Resolve a client-supplied artifact path without allowing traversal."""
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = session_dir / candidate
    resolved = candidate.resolve()
    root = session_dir.resolve()
    if resolved != root and root not in resolved.parents:
        raise HTTPException(status_code=400, detail="Artifact path escapes the session")
    return resolved


def _load_session(session_id: str) -> Dict[str, Any]:
    """Session memory: {ctx, messages}. Reloaded from disk so restarts survive."""
    if session_id in WEB_SESSION_CTX:
        return WEB_SESSION_CTX[session_id]
    sf = _session_file(session_id)
    if sf.exists():
        try:
            data = json.loads(sf.read_text())
            if isinstance(data.get("ctx"), dict):
                data["ctx"].pop("cancel_event", None)
            WEB_SESSION_CTX[session_id] = data
            return data
        except (OSError, ValueError):
            pass
    fresh = {"ctx": {}, "messages": [], "created_at": time.time(), "title": ""}
    WEB_SESSION_CTX[session_id] = fresh
    return fresh


_BULKY_NODE_OUTPUTS = (
    "eigenvalues_ev", "dos_curve", "pdos_curve", "k_axis", "k_ticks", "k_labels",
)


def _slim_for_disk(session: Dict[str, Any]) -> Dict[str, Any]:
    """Disk copy of a session: keep only the newest chart/viewer and strip
    bulky node-output arrays. The live in-memory session keeps everything;
    frontend restoreFigures only needs the newest of each payload, and the
    chart dict already embeds its own copy of the curves."""
    slim = json.loads(json.dumps(session, ensure_ascii=False, default=str))
    # Runtime-only handles must never become conversational context after a
    # restart (the cancel event is intentionally process-local).
    if isinstance(slim.get("ctx"), dict):
        slim["ctx"].pop("cancel_event", None)
    keep_chart = keep_viewer = False
    for msg in reversed(slim.get("messages", [])):
        for r in msg.get("results") or []:
            if r.get("chart"):
                if keep_chart:
                    r.pop("chart")
                else:
                    keep_chart = True
            if r.get("viewer"):
                if keep_viewer:
                    r.pop("viewer")
                else:
                    keep_viewer = True
            j = r.get("json")
            if isinstance(j, dict):
                for node in (j.get("nodes") or {}).values():
                    outs = node.get("outputs")
                    if isinstance(outs, dict):
                        for k in _BULKY_NODE_OUTPUTS:
                            outs.pop(k, None)
    return slim


def _save_session(session_id: str) -> None:
    session = WEB_SESSION_CTX.get(session_id)
    if session is None:
        return
    sf = _session_file(session_id)
    sf.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_slim_for_disk(session), ensure_ascii=False, default=str)
    # Replace atomically so a crash cannot leave a half-written session.json.
    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=sf.parent, prefix=".session-", delete=False
        ) as tmp:
            tmp.write(payload)
            tmp_path = Path(tmp.name)
        os.replace(tmp_path, sf)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

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
    structure_path: Optional[str] = None
    run_config: Optional[Dict[str, Any]] = None
    approved: bool = False


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    commands: List[Dict[str, Any]]
    results: List[Dict[str, Any]]


def _apply_request_context(req: ChatRequest, session_dir: Path, ctx: Dict[str, Any]) -> None:
    if req.run_config:
        allowed = {"mode", "accuracy", "ecutwfc", "kpoints", "nbnd", "nkpoints_bands"}
        clean = {k: v for k, v in req.run_config.items() if k in allowed}
        clean["approved"] = bool(req.approved)
        if clean.get("mode") == "research" and not req.approved:
            raise HTTPException(status_code=409, detail="科研模式需要先确认计算方法")
        ctx["run_config"] = clean
    if req.structure_path:
        structure = _inside_session(session_dir, req.structure_path)
        if not structure.is_file():
            raise HTTPException(status_code=400, detail="Uploaded structure no longer exists")
        ctx["last_structure_source"] = str(structure)
        ctx["last_structure_file"] = str(structure)


_STRUCTURE_SUFFIXES = {".cif", ".vasp", ".poscar", ".contcar", ".xyz"}


@app.post("/api/sessions/{session_id}/structure")
async def upload_structure(session_id: str, file: UploadFile = File(...)) -> Dict[str, Any]:
    """Upload, validate and normalize a structure for this session."""
    session_dir = _session_dir(session_id)
    upload_dir = session_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    original_name = Path(file.filename or "structure.cif").name
    suffix = Path(original_name).suffix.lower()
    if suffix not in _STRUCTURE_SUFFIXES and original_name.upper() not in {"POSCAR", "CONTCAR"}:
        raise HTTPException(status_code=400, detail="支持 CIF、POSCAR、CONTCAR、VASP 和 XYZ")
    content = await file.read(20 * 1024 * 1024 + 1)
    if not content:
        raise HTTPException(status_code=400, detail="结构文件为空")
    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="结构文件不能超过 20 MB")
    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", original_name)[:100] or "structure.cif"
    original = upload_dir / f"{uuid.uuid4().hex[:8]}_{safe_name}"
    original.write_bytes(content)
    normalized = upload_dir / f"structure_{uuid.uuid4().hex[:8]}.cif"
    try:
        from ase.io import write as ase_write

        from dft_forge.compiler import read_structure

        atoms = read_structure(original)
        if len(atoms) == 0:
            raise ValueError("structure contains no atoms")
        ase_write(str(normalized), atoms, format="cif")
    except Exception as exc:
        original.unlink(missing_ok=True)
        normalized.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"无法解析结构文件：{exc}") from exc
    session = _load_session(session_id)
    ctx = session.setdefault("ctx", {})
    ctx["last_structure_source"] = str(normalized)
    ctx["last_structure_file"] = str(normalized)
    ctx["last_formula"] = atoms.get_chemical_formula()
    session.setdefault("attachments", []).append({
        "name": original_name,
        "path": str(normalized),
        "formula": atoms.get_chemical_formula(),
        "natoms": len(atoms),
        "ts": time.time(),
    })
    _save_session(session_id)
    return {
        "ok": True,
        "name": original_name,
        "path": str(normalized),
        "formula": atoms.get_chemical_formula(),
        "natoms": len(atoms),
        "cif": normalized.read_text(errors="replace"),
    }


def _artifact_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in _STRUCTURE_SUFFIXES or path.name.upper() in {"POSCAR", "CONTCAR"}:
        return "structure"
    if suffix in {".in", ".pwi"}:
        return "input"
    if suffix in {".out", ".err", ".log", ".xml"}:
        return "output"
    if "evidence" in path.name.lower() or "manifest" in path.name.lower():
        return "evidence"
    if suffix in {".db", ".sqlite", ".sqlite3", ".db-wal", ".db-shm"}:
        return "state"
    return "data"


def _session_artifacts(session_id: str) -> List[Dict[str, Any]]:
    root = _session_dir(session_id)
    if not root.exists():
        return []
    artifacts: List[Dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "session.json" or path.is_symlink():
            continue
        try:
            relative = path.relative_to(root).as_posix()
            stat = path.stat()
        except OSError:
            continue
        digest = None
        if stat.st_size <= 32 * 1024 * 1024:
            try:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError:
                pass
        artifacts.append({
            "path": relative,
            "name": path.name,
            "kind": _artifact_kind(path),
            "size": stat.st_size,
            "updated_at": stat.st_mtime,
            "sha256": digest,
            "download_url": f"/api/sessions/{session_id}/artifacts/{relative}",
        })
    return artifacts


@app.get("/api/sessions/{session_id}/artifacts")
def list_session_artifacts(session_id: str) -> Dict[str, Any]:
    return {"artifacts": _session_artifacts(session_id)}


@app.get("/api/sessions/{session_id}/artifacts/{artifact_path:path}")
def download_session_artifact(session_id: str, artifact_path: str) -> FileResponse:
    root = _session_dir(session_id)
    path = _inside_session(root, artifact_path)
    if not path.is_file() or path.is_symlink():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(path, filename=path.name)


def _graph_runs_for_session(session_id: str) -> List[Dict[str, Any]]:
    db = _session_dir(session_id) / "graphs" / "graph_state.db"
    if not db.exists():
        return []
    try:
        with sqlite3.connect(db) as conn:
            rows = conn.execute(
                "SELECT id, template_id, status, updated_at FROM graph_runs "
                "ORDER BY CAST(updated_at AS REAL) DESC"
            ).fetchall()
    except (sqlite3.Error, OSError):
        return []
    return [
        {"run_id": row[0], "template_id": row[1], "status": row[2], "updated_at": row[3]}
        for row in rows
    ]


@app.get("/api/runs")
def list_product_runs() -> Dict[str, Any]:
    runs: List[Dict[str, Any]] = []
    for session_path in WEB_WORKDIR.iterdir() if WEB_WORKDIR.exists() else []:
        if not session_path.is_dir() or not _SAFE_SESSION_ID.match(session_path.name):
            continue
        for item in _graph_runs_for_session(session_path.name):
            item["session_id"] = session_path.name
            item["active"] = session_path.name in WEB_ACTIVE_RUNS
            runs.append(item)
    runs.sort(key=lambda item: float(item.get("updated_at") or 0), reverse=True)
    return {"runs": runs[:100]}


@app.get("/api/sessions/{session_id}/reproducibility")
def reproducibility_manifest(session_id: str) -> Dict[str, Any]:
    session = _load_session(session_id)
    safe_config = _config_state()
    safe_config.pop("api_key_hint", None)
    return {
        "schema": "dft-forge/reproducibility-v1",
        "session_id": session_id,
        "created_at": time.time(),
        "llm": {k: safe_config.get(k) for k in ("vendor", "base_url", "model")},
        "compute": {
            key: _compute_state().get(key)
            for key in ("executor", "scheduler", "remote_qe_bin", "walltime")
        },
        "run_config": session.get("ctx", {}).get("run_config", {}),
        "structure": session.get("ctx", {}).get("last_structure_source"),
        "requests": [m.get("text") for m in session.get("messages", []) if m.get("role") == "user"],
        "tool_calls": [
            command
            for message in session.get("messages", [])
            for command in (message.get("commands") or [])
        ],
        "runs": _graph_runs_for_session(session_id),
        "artifacts": _session_artifacts(session_id),
    }


class RunActionRequest(BaseModel):
    action: str = "retry"
    run_id: Optional[str] = None


@app.post("/api/sessions/{session_id}/runs/action")
async def run_action(session_id: str, req: RunActionRequest) -> Dict[str, Any]:
    if session_id in WEB_ACTIVE_RUNS:
        raise HTTPException(status_code=409, detail="该会话已有任务正在运行")
    runs = _graph_runs_for_session(session_id)
    run_id = req.run_id or (runs[0]["run_id"] if runs else None)
    if not run_id:
        raise HTTPException(status_code=404, detail="该会话没有可恢复的图任务")
    from dft_forge.tools.graph_tools import _get_engine

    engine = _get_engine(str(_session_dir(session_id) / "graphs"))
    try:
        if req.action == "resume":
            run = await asyncio.to_thread(engine.resume, run_id)
        elif req.action == "retry":
            run = await asyncio.to_thread(engine.retry, run_id)
        else:
            raise HTTPException(status_code=400, detail="action must be retry or resume")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return engine.status(run.run_id)


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
        "attachments": session.get("attachments", []),
    }


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str) -> Dict[str, Any]:
    d = _session_dir(session_id)  # 400 on traversal/symlink escape
    if session_id in WEB_ACTIVE_RUNS:
        raise HTTPException(
            status_code=409,
            detail="Session is running; stop the calculation before deleting it",
        )
    WEB_SESSION_CTX.pop(session_id, None)
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
    return {"deleted": session_id}


@app.post("/api/sessions/{session_id}/cancel")
async def cancel_session(session_id: str) -> Dict[str, Any]:
    """Stop an in-flight calculation: signal the scheduler AND kill pw.x.

    Two-pronged because the scheduler only notices its event between ticks —
    the process kill is what actually frees the CPU within milliseconds.
    Only THIS session's engine is swept: sessions get their own engine under
    <session_dir>/graphs, and a global sweep would kill other sessions' runs.
    """
    session_dir = _session_dir(session_id)
    entry = WEB_ACTIVE_RUNS.get(session_id)
    if entry is None:
        return {"cancelled": False, "reason": "no active run for this session"}

    entry["cancel"].set()
    killed = 0
    try:
        from dft_forge.tools.graph_tools import cancel_graphs_under

        killed = cancel_graphs_under(session_dir)
    except Exception:
        pass
    return {"cancelled": True, "processes_killed": killed}


@app.get("/api/sessions/{session_id}/running")
def session_running(session_id: str) -> Dict[str, Any]:
    return {"running": session_id in WEB_ACTIVE_RUNS}


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
    session_dir = _session_dir(session_id)
    session_dir.mkdir(parents=True, exist_ok=True)

    session = _load_session(session_id)
    ctx = session.setdefault("ctx", {})
    _apply_request_context(req, session_dir, ctx)
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
    if session_id in WEB_ACTIVE_RUNS:
        raise HTTPException(status_code=409, detail="该会话已有任务正在运行")
    session_dir = _session_dir(session_id)
    session_dir.mkdir(parents=True, exist_ok=True)

    session = _load_session(session_id)
    ctx = session.setdefault("ctx", {})
    _apply_request_context(req, session_dir, ctx)
    messages = session.setdefault("messages", [])
    if not session.get("title"):
        session["title"] = req.message.strip()[:40]
    message = req.message.strip()

    q: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    cancel_event = threading.Event()

    def emit(ev) -> None:
        loop.call_soon_threadsafe(q.put_nowait, ev)

    async def worker() -> None:
        # the graph scheduler polls this event and kills pw.x when it fires
        ctx["cancel_event"] = cancel_event
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
            if cancel_event.is_set():
                messages.append({
                    "role": "agent",
                    "text": "计算已停止（进行中的计算被终止；已完成的结果保留在会话中）。",
                    "ts": time.time(),
                })
                _save_session(session_id)
        except Exception as exc:  # noqa: BLE001
            emit({"type": "error", "message": str(exc)})
        finally:
            ctx.pop("cancel_event", None)
            WEB_ACTIVE_RUNS.pop(session_id, None)
            emit(None)

    task = asyncio.create_task(worker())
    WEB_ACTIVE_RUNS[session_id] = {"task": task, "cancel": cancel_event, "ts": time.time()}

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
