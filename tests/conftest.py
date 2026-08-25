"""Shared test fixtures.

Tests must stay deterministic and offline: pin the LLM provider to the
rule-based dummy before any AgentLoop import loads .env, and keep web
sessions out of the real user workdir.
"""

import os
import tempfile

os.environ.setdefault("DFT_FORGE_LLM_PROVIDER", "dummy")
os.environ.pop("DFT_FORGE_LLM_API_KEY", None)
os.environ["DFT_FORGE_WEB_WORKDIR"] = os.path.join(tempfile.gettempdir(), "dft-forge-web-test")
