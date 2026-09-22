"""Portable, per-run evidence. This is not a scientific validity certificate."""

from __future__ import annotations

import math
import re
import time
from pathlib import Path

_PRIVATE_KEY = re.compile(r"(?:api[_-]?key|password|secret|token|credential|ssh|host|username)", re.I)
_LOCAL_PATH = re.compile(r"/(?:Users|home|root|private|tmp|var|opt|etc|mnt|Volumes)/[^\s\"'<>]+")


def build_evidence(snapshot: dict, root: Path, artifacts: list[dict], *, live: bool = False) -> dict:
    notices = ["artifacts_not_atomic", "scientific_validation_not_assessed", "review_before_sharing"]

    def notice(value):
        if value not in notices:
            notices.append(value)

    def clean(value):
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                if _PRIVATE_KEY.search(str(key)):
                    notice("private_fields_omitted")
                else:
                    result[key] = clean(item)
            return result
        if isinstance(value, list):
            return [clean(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value):
            notice("nonfinite_values_omitted")
            return None
        if isinstance(value, str):
            if value == str(root) or str(root) + "/" in value:
                notice("paths_redacted")
                value = "." if value == str(root) else value.replace(str(root) + "/", "./")
            if value.startswith("/") or re.match(r"^[A-Za-z]:[\\/]", value):
                notice("paths_redacted")
                return "[external path omitted]"
            value, count = _LOCAL_PATH.subn("[external path omitted]", value)
            if count:
                notice("paths_redacted")
        return value

    def profile(raw):
        if not isinstance(raw, dict):
            return {}
        return clean({key: raw[key] for key in ("backend", "simulated", "scheduler", "walltime_limit_sec") if key in raw})

    def execution(raw):
        if not isinstance(raw, dict) or not raw:
            return {"origin_unknown": True, "attempts": []}
        origin = raw.get("origin") if isinstance(raw.get("origin"), dict) else {}
        identity = origin.get("identity")
        # Only the actual adapter's opaque digest is portable; arbitrary
        # plugin identities may contain private addresses or account names.
        safe_identity = identity if isinstance(identity, str) and re.fullmatch(r"exec_[a-f0-9]{64}", identity) else None
        identity_status = "recorded" if safe_identity else ("omitted_nonportable" if identity else "unavailable")
        if identity_status == "omitted_nonportable":
            notice("execution_identity_omitted")
        attempts = raw.get("attempts")
        if not isinstance(attempts, list):
            attempts = []
            notice("execution_attempts_unavailable")
        return {
            "origin_unknown": bool(raw.get("origin_unknown", not bool(origin))),
            "origin": {"identity": safe_identity, "identity_status": identity_status, "profile": profile(origin.get("profile"))},
            "attempts": [{
                "operation": item.get("operation"), "started_at": clean(item.get("started_at")),
                "profile": profile(item.get("profile")),
            } for item in attempts if isinstance(item, dict) and item.get("operation") in {"start", "resume", "retry"}],
        }

    raw_run = snapshot["run"]
    run = clean({key: value for key, value in raw_run.items() if key != "execution"})
    run["execution"] = execution(raw_run.get("execution"))
    live = live or run["state"] in {"created", "validated", "running"}
    if live:
        notice("run_still_active")
    if run["workspace"].startswith("workflows/"):
        notice("derived_macro_results_may_be_stale")
    if run["template_snapshot_status"] != "recorded":
        notice("template_snapshot_" + run["template_snapshot_status"])
    nodes = []
    for raw in snapshot["nodes"]:
        outputs = raw.get("outputs") or {}
        if not isinstance(outputs, dict):
            outputs = {}
            notice("node_outputs_unavailable")
        nodes.append(clean({
            "node_id": raw["node_id"], "state": raw["state"],
            "attempt": raw.get("attempt", 0), "repair_attempts": raw.get("repair_attempts", 0),
            "params_resolved": bool(raw.get("params_resolved", False)),
            "resolved_params": raw.get("params", {}) if raw.get("params_resolved") else None,
            "scalar_outputs": {key: value for key, value in outputs.items() if value is None or isinstance(value, (str, int, float, bool))},
            "error": raw.get("error"),
        }))
    return {
        "schema": "dft-forge/run-evidence-v1", "exported_at": time.time(),
        "snapshot": {"live": live, "database_consistent": True, "artifacts_atomic": False},
        "run": run, "nodes": nodes, "artifacts": artifacts,
        "scientific_validation": "not_assessed", "notices": notices,
    }
