# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Existing codebase answers it: FastAPI backend (dft_forge/web), vanilla JS + ES modules frontend (dft_forge/web/frontend), 3Dmol.js for structure rendering. No build step; files served statically.

## Users

[Inferred from brief] The owner-developer and lab colleagues who run DFT (Quantum ESPRESSO) calculations: materials-science researchers who want to say "算 GaAs 的结构优化" in Chinese and get verified numbers, then inspect the structure and results visually.

## Product Purpose

DFT-Forge is a computation agent: natural language in, verified QE calculations out. Success = a researcher completes a full calc loop (structure → input → run → parse → verify → visualize) from the chat box without touching a CLI.

## Positioning

LLM planner (OpenAI-compatible API) + deterministic graph runtime (DAG engine) + real QE binaries on local/SSH/HPC — the agent owns the whole pipeline, not just input generation.

## Operating Context

- Chat sessions in Chinese; planner falls back to deterministic rules when LLM unavailable
- Local pw.x first (BYOC philosophy); SSH/SLURM/PBS adapters ready
- 57-material library + GBRV pseudopotentials; 2D builder (graphene/h-BN, doping, adsorption sites)
- Sessions live under /tmp/dft-forge-web/<session_id>/; .env holds LLM + QE paths

## Capabilities and Constraints

- Graph templates: t1_vc_relax, t2_bands, t2_dos
- Structure tools: import/generate/analyze/build2d (top/bridge/hollow sites)
- Confirmed needs (this round): persistent session memory, reloadable chat history, a genuinely useful structure viewer (rotate/zoom, unit-cell frame, periodic replicas, result charts), journal-grade visual redesign
- [Undecided] multi-user auth (currently single-user localhost)

## Product Principles

1. Numbers must be verifiable — every result carries provenance (run dirs, SQLite state)
2. The LLM plans; deterministic code executes and validates — never the reverse
3. Density serves scanning: researchers read tables and numbers, not marketing prose
4. Chinese-first UX; scientific terminology stays in English where the field uses it

## Accessibility & Inclusion

[None established yet]
