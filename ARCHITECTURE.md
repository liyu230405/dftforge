# DFT-Forge Architecture

**A general-purpose scientific computation agent** — built around a CatGo-inspired graph runtime. Quantum ESPRESSO is the first *engine*, not the identity of the project.

> Design principle: LLM plans; deterministic code executes. The runtime, scheduling, HPC submission, parsing, and verification never depend on an LLM.

## Layered Design

```
┌─────────────────────────────────────────────────────────┐
│  LLM Planner (OpenAI-compatible API, optional)          │
│  natural language → WorkflowIR / template_id + inputs   │
├─────────────────────────────────────────────────────────┤
│  Tools Layer        tools/ (registry + MCP bindings)    │
│                     graph.run / graph.status / ...      │
├─────────────────────────────────────────────────────────┤
│  Graph Engine       runtime/ (CatGo-style core)         │
│  ┌───────────────────────────────────────────────────┐  │
│  │ GraphEngine  facade: create/start/pause/resume/   │  │
│  │               retry/status                        │  │
│  │ GraphScheduler  dependency-driven dispatch loop,  │  │
│  │                 concurrency caps, repair hooks    │  │
│  │ GraphTemplate  JSON DAG specs + validation        │  │
│  │ SQLiteStateStore  crash-safe persistence, resume  │  │
│  │ States         node & run state machines          │  │
│  └───────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────┤
│  Engine Tools       engines/ — node implementations    │
│                     qe.py (vc-relax/scf/nscf/bands/dos) │
│                     ...add VASP/ABINIT/LAMMPS here     │
├─────────────────────────────────────────────────────────┤
│  Compute Layer      hpc/ (runner abstraction +         │
│                     SLURM/PBS schedulers + job scripts)│
│                     executor.py (Local/SSH/Fake)       │
├─────────────────────────────────────────────────────────┤
│  Domain Kernels     compiler/ parser.py verifier.py    │
│                     catalog/ (materials + pseudos)     │
└─────────────────────────────────────────────────────────┘
```

## 1. Graph Runtime (`dft_forge/runtime/`)

The heart of the system. Generic over tools — nothing in this layer knows about DFT.

- **`states.py`** — `NodeState` (pending → ready → running → succeeded/failed/repairing/blocked/skipped/cancelled) and `RunState` machines with explicit allowed-transition tables; illegal transitions raise.
- **`graph.py`** — `GraphTemplate` / `NodeSpec`: a JSON-declared DAG (`templates/*.json`). Validation covers unique ids, existing dependencies, tool availability, unknown tools, reference syntax, and cycle detection.
- **`run.py`** — `GraphRun` / `NodeRun` instances plus parameter binding: `"${inputs.material}"` and `"${nodes.scf.outputs.stdout_file}"` are resolved once per dispatch, then kept stable across repairs.
- **`scheduler.py`** — dependency-driven loop: promote pending → ready, dispatch under a global concurrency cap (thread pool), feed failures to registered repairers (`RepairHandler`), let the run state machine aggregate (succeeded / partially_succeeded / failed).
- **`store.py`** — SQLite persistence for runs and nodes (`INSERT OR REPLACE`, idempotent). `resume()` resets in-flight nodes so an interrupted run continues after a crash.
- **`engine.py`** — `GraphEngine` facade: register tools/templates, `create/start/run_template/pause/resume/retry/status/list_runs`. This is the only class callers need.

### Adding a new workload

The runtime is engine-agnostic. A "tool" is any object with `name` and `execute(node, ctx) -> dict`:

```python
class VASPCalcTool:
    name = "vasp"
    def execute(self, node, ctx): ...   # returns outputs dict

engine.register_tool(VASPCalcTool())
engine.templates_dir = Path("templates/")   # JSON templates reference "tool": "vasp"
```

Cross-node data flows through `${nodes.<id>.outputs.<key>}` bindings — e.g. the `bands` node consumes the `scf` node's `stdout_file`. Upstream artifacts are linked via `ctx.upstream` (the runtime guarantees dependencies succeeded first).

## 2. HPC Layer (`dft_forge/hpc/`)

User-supplied compute (BYOC), mirroring CatGo:

- **`runner.py`** — `CommandRunner` abstraction with `LocalRunner` / `SSHRunner` (subprocess-backed SSH, supports jump hosts, keys, timeouts). All remote operations are constructed internally; the LLM never generates shell/SSH strings.
- **`scheduler.py`** — `SchedulerInterface` with `SlurmScheduler` (sbatch/squeue/sacct/scancel, sacct fallback for finished jobs) and `PbsScheduler` (qsub/qstat/qdel). Normalized `JobStatus` for both.
- **`job_script.py`** — deterministic sbatch/qsub script rendering with parameter-priority resolution (params > job defaults > safe defaults) and alias mapping (e.g. `ppn` → `cpus_per_task`).
- **`executor.SSHExecutor`** — grafted onto this layer (no longer a scaffold): every remote operation goes through an injectable `CommandRunner`, so tests script a fake runner. Two modes — direct (`scheduler="none"`, remote-side `timeout` wrapper, exit code via a stdout marker) and scheduler (`slurm`/`pbs`, submit + poll to a terminal state). File transport is chunked tar+base64 over the text channel with `.save` symlink dereference on upload. Not yet validated end-to-end on a live cluster.

## 3. Engine Tools (`dft_forge/engines/`)

Node implementations that combine domain kernels:

- **`qe.py`** — `QECalcTool` supports `vc-relax`, `scf`, `nscf`, `bands`, `dos`. Each node: compile input → execute → parse → verify → emit typed outputs (raising `ToolError(category=..., repairable=...)` on failure). `QEParamRepairer` implements monotone parameter strengthening (cutoff bumps) for convergence-class failures.
- **QE 7.x compatibility notes** (hard-won): `CELL_PARAMETERS` must be omitted when `ibrav ≠ 0`; `K_POINTS crystal` lines require the weight column; the eigenvalue XML (`<prefix>.save/data-file-schema.xml`) is namespace-less at child level and uses **Hartree** units; NSCF output has no SCF-convergence marker (verified via `End of band structure calculation` instead); lattice constants parsed from `ibrav=0` primitive cells are converted to conventional cubic form (fcc: |a₁|·√2).

## 4. Tools & MCP (`dft_forge/tools/`, `dft_forge/mcp/`)

- `tools/registry.py` + `models.py`: `ToolEntry` registry with JSON-schema input contracts.
- `tools/graph_tools.py`: exposes `graph.run`, `graph.list_templates`, `graph.status` as callable tools.
- `mcp/server.py`: MCP server exposing the registry to external agents.

## 5. LLM Integration (`dft_forge/llm.py`)

OpenAI-compatible chat-completion provider configured via `DFT_FORGE_LLM_*` env vars (provider/base_url/api_key/model). The LLM's only job is planning: natural language → template selection + inputs (WorkflowIR). Everything downstream is deterministic.

## 6. Provenance & Trust

- `verifier.py` — physics-aware checks per calculation type (convergence, forces, pressure, NSCF markers, band-gap sanity), not just exit codes.
- `ledger.py` — SQLite evidence trail of attempts, failures, and recovery actions.
- Every run leaves a full artifact tree: `<base_dir>/<run_id>/<node_id>/` with inputs, stdout/stderr, XML, and parsed JSON.

## Verified End-to-End (real pw.x, QE 7.5)

| Template | Material | Result |
|---|---|---|
| `t1_vc_relax` | NaCl | a = 5.701 Å (PBE ≈ 5.70, exp. 5.64), E = −128.90 Ry |
| `t2_bands` | Si | indirect gap 0.575 eV (PBE ≈ 0.6, exp. 1.17), E_F = 6.53 eV, 100 k-points on Γ-X-W-K-Γ-L-U-W-L-K\|U-X |

## Test Suite

```bash
python -m pytest tests/ -q        # 300 tests: runtime, scheduler, hpc, engines, e2e (fake + real QE)
```
