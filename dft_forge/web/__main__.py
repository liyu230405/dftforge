"""Run DFT-Forge web backend."""

from __future__ import annotations

import os

import uvicorn

from dft_forge.web import app


def main() -> None:
    # The API can launch calculations and persist credentials, so it must not
    # be exposed to the LAN unless the operator opts in explicitly.
    host = os.environ.get("DFT_FORGE_WEB_HOST", "127.0.0.1")
    port = int(os.environ.get("DFT_FORGE_WEB_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
