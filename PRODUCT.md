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
- Sessions persist under sessions/<session_id>/ by default (override with DFT_FORGE_WEB_WORKDIR); .env holds LLM + QE paths
- One local server process owns session execution, recovery and cancellation

## Capabilities and Constraints

- Graph templates: t1_vc_relax, t2_bands, t2_dos
- Structure tools: import/generate/analyze/build2d (top/bridge/hollow sites)
- Confirmed needs (this round): persistent session memory, reloadable chat history, a genuinely useful structure viewer (rotate/zoom, unit-cell frame, periodic replicas, result charts), journal-grade visual redesign
- Run center: inspect node failures/results, retry a selected failed graph, resume paused graphs, stop background recovery, and retain successful nodes
- Reproducibility: new graph runs retain their original template and repair parameters; old dynamic graphs without snapshots are inspectable but cannot be resumed
- Graph recovery restores graph nodes and artifacts; replaying a composite workflow's reduction or the rest of a chat plan remains a separate follow-up
- Research/demo mode is enforced against planned computation tools; research requires explicit method approval and a real backend, demo requires the explicit fake backend
- Invalid method parameters are rejected before execution; plan-only flags and custom method overrides apply to one request
- Method previews follow the planned capability: per-step overrides, automatic policies, inapplicable preset fields and actual convergence grids are disclosed; changed plans or structure contents invalidate earlier confirmation
- Task-aware parameter previews precede execution: the LLM selects the workflow, deterministic structure/pseudopotential rules resolve concrete parameters, and the user confirms one frozen method card
- Confirmed parameters, pseudo hashes and source reasons are stored with the graph; source changes and parameter-changing repair require a new preview
- Optional bounded cutoff/k-point scans feed a selected internal energy plateau into SCF/bands/DOS, with failed/cancelled or boundary-only scans blocking the target calculation
- Artifact listing and direct downloads share a visibility policy: scientific files only, excluding session history, hidden files, links and live SQLite state
- Per-run evidence exports preserve the recorded template, resolved parameters, execution provenance and scoped artifact hashes; live snapshots and unassessed scientific validity are explicitly labeled
- [Undecided] multi-user auth (currently single-user localhost)

## Product Principles

1. Numbers must be verifiable — every result carries provenance (run dirs, SQLite state)
2. The LLM plans; deterministic code executes and validates — never the reverse
3. Density serves scanning: researchers read tables and numbers, not marketing prose
4. Chinese-first UX; scientific terminology stays in English where the field uses it

## Accessibility & Inclusion

[None established yet]
