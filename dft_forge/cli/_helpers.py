from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _json_dumps(data: Any) -> str:
    return json.dumps(data, indent=2, default=str)



def _write_output(data: Any, output_path: Optional[Path], output_format: str = "json") -> Optional[Path]:
    """Write data to stdout or file depending on flags."""
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_format == "json":
            output_path.write_text(_json_dumps(data))
        else:
            output_path.write_text(str(data))
        return output_path
    else:
        if output_format == "json":
            print(_json_dumps(data))
        else:
            print(data)
        return None



def _success_response(data: Dict[str, Any], output_path: Optional[Path]) -> int:
    data.setdefault("ok", True)
    _write_output(data, output_path)
    return 0



def _error_response(message: str, code: int = 1, details: Optional[Dict[str, Any]] = None) -> int:
    payload: Dict[str, Any] = {
        "ok": False,
        "error": message,
        "code": code,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if details:
        payload["details"] = details
    print(_json_dumps(payload), file=sys.stderr)
    return code
