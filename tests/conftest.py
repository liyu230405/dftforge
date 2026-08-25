"""Shared test fixtures.

Tests must stay deterministic and offline: pin the LLM provider to the
rule-based dummy before any AgentLoop import loads .env.
"""

import os

os.environ.setdefault("DFT_FORGE_LLM_PROVIDER", "dummy")
os.environ.pop("DFT_FORGE_LLM_API_KEY", None)
