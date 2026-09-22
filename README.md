# DFT-Forge

**General-purpose scientific computation agent** — a graph runtime with Quantum ESPRESSO as its first engine.

> "LLM 只决定做什么；确定性程序决定输入文件如何生成、程序如何运行、结果如何解析和是否可信。"

## What is DFT-Forge?

DFT-Forge is an agent-friendly CLI and web toolchain for scientific computing workflows. It is designed to be used by coding agents, chat UIs, and researchers who want structured, inspectable, and recoverable computation workflows:

- **Graph runtime**: JSON-declared DAG templates, node/run state machines, dependency-driven scheduling, crash-safe SQLite persistence, repair & retry.
- **Engine-agnostic core**: the runtime knows nothing about DFT; QE is one engine (`engines/qe.py`). New engines (VASP/ABINIT/LAMMPS/…) plug in as tools.
- **User-supplied compute (BYOC)**: local, SSH, or HPC schedulers (SLURM/PBS) — the LLM never generates shell/SSH/scheduler strings.
- **Agent-friendly CLI**: every command does one thing and outputs JSON.
- **Deterministic kernels**: compiler (input generation), executor, parser, verifier — no LLM in the execution path.
- **Evidence ledger**: SQLite-backed audit trail of runs, failures, and recovery actions.
- **Tool registry + MCP server**: structured tools that external agents can call.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full layered design and the "adding a new engine" guide.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Check environment
python -m dft_forge.cli doctor

# Import a structure
python -m dft_forge.cli structure import POSCAR_Si --output structure.json

# Build a QE input
python -m dft_forge.cli input build --material Si --type scf --output si.scf.in

# Run locally
python -m dft_forge.cli job submit si.scf.in

# Parse and verify
python -m dft_forge.cli result parse si.scf.out --type scf --output result.json
python -m dft_forge.cli result verify result.json --task-type T1

# Query ledger
python -m dft_forge.cli ledger query --task-id T1_Si_vcrelax --limit 20

# Start web chat backend
python -m dft_forge.web
```

## Project Structure

```
dft-forge/
├── dft_forge/
│   ├── runtime/        # Graph runtime core (engine-agnostic): states, graph
│   │                   # templates, scheduler, SQLite store, GraphEngine facade
│   ├── hpc/            # BYOC compute layer: runner abstraction, SLURM/PBS
│   │                   # schedulers, sbatch/qsub script rendering
│   ├── engines/        # Node tools per engine (qe.py: pw.x + DFPT post-processing)
│   ├── workflows/      # Deterministic convergence/defect/thermo/magnetism/phonon macros
│   ├── protocol/       # TaskSpec, WorkflowIR, PlanPatch, EvidenceBundle (schemas)
│   ├── compiler/       # Deterministic QE input generation
│   ├── executor.py     # Pluggable executor abstraction
│   ├── parser.py       # QE stdout/XML parsing
│   ├── verifier.py     # Physical convergence checks
│   ├── catalog/        # Material profiles and task definitions
│   ├── runner.py       # Deterministic task execution
│   ├── recovery.py     # Failure classification and bounded recovery
│   ├── ledger.py       # SQLite evidence ledger
│   ├── structure.py    # Structure import/validation
│   ├── llm.py          # LLM provider abstraction
│   ├── planner.py      # WorkflowIR generation
│   ├── agent.py        # Agent solve loop
│   ├── tools/          # Tool registry and tool definitions
│   ├── mcp/            # MCP server for external agents
│   ├── web/            # FastAPI chat backend + frontend
│   └── cli.py          # Agent-friendly CLI entry points
├── templates/          # Graph templates (JSON DAGs): t1_vc_relax, t2_bands, t2_dos
├── assets/pseudos/     # Pseudopotential files
├── tests/              # Pytest test suite (579 tests)
├── scripts/            # Utility scripts
├── outputs/            # Experiment outputs
└── work/               # Working files
```

## Supported Ranges

### Materials
- Library: 57 materials (MP structures + hand-tuned Si/Al/NaCl/MgO), GBRV USPP PBE v1.5 pseudopotentials for 65 elements
- Dynamic generation for arbitrary formulas via ASE prototypes
- User-provided structures: CIF, POSCAR/CONTCAR, XYZ, QE input, explicit lattice+coordinates

### Calculations
- T1 vc-relax / relax
- T2 bands and DOS
- Generic SCF / NSCF
- Parameter convergence scans with a bounded ecut/k-point/smearing grid
- Neutral bulk vacancy formation-energy workflow with explicit chemical potential
- Binary/ternary chemical-potential stability-domain scan with supplied competitors
- Collinear NM/FM/AFM comparison with initial moments and realized magnetization checks
- DFPT phonons: relax → SCF → ph.x → q2r.x → matdyn.x, including imaginary-mode detection

The five composite capabilities are registered as `workflow.convergence`,
`workflow.neutral_vacancy`, `thermo.chemical_potential_scan`,
`workflow.magnetic_compare`, and `workflow.phonon_dfpt`. Each macro owns a
bounded DAG and its scientific reduction; the LLM selects the macro but does
not invent the internal graph. Real phonon runs require `ph.x`, `q2r.x`, and
`matdyn.x` alongside `pw.x`.

Current boundaries are deliberate: charged-defect electrostatics and
finite-size corrections, SOC/noncollinear magnetism, automatic exhaustive
competitor retrieval, finite-temperature chemical potentials, and phonon
non-analytic LO/TO corrections are not silently approximated.

### Structure Operations
- Import and validate structures
- Build unchanged bulk supercells for matched pristine controls
- Build explicit neutral bulk vacancies (element removal; no charge correction)
- Normalize to primitive cell
- Nearest-neighbor sanity checks
- Space-group detection via spglib

### Executors
- Local executor (real subprocess pw.x/bands.x/dos.x runs)
- SSH executor (`--backend ssh`): real remote execution grafted onto the
  `hpc/` layer (SSHRunner + SLURM/PBS scheduler adapters)
  - Two modes: `scheduler="none"` runs the QE binary directly under a
    remote-side `timeout`; `scheduler="slurm"|"pbs"` renders an sbatch/qsub
    script and polls squeue/sacct (or qstat) to completion
  - File transport is chunked tar+base64 over the SSH command channel; the
    upload dereferences `.save` symlink chains so scf→nscf→bands carry real
    wavefunctions to the remote side
  - Cancel semantics: scheduler mode issues scancel/qdel; direct mode kills
    the remote run by job directory (`pkill -f <job dir>`, which matches the
    QE process because it is launched with an absolute `-in` path). A cancel
    that never reached the host reports `False` instead of recording a stop
    that did not happen
  - Failure semantics: a dead transport (ssh timeout, dropped connection)
    and a lost download are reported as failed runs — never as a successful
    run with empty output
  - Status: transport and both execution modes are covered by offline tests
    against a scripted fake runner; they have not yet been validated
    end-to-end on a live cluster, and multi-GB wavefunction dirs will be
    slow over the text-channel transport — prefer scheduler mode for
    production runs
- Fake executor for explicit offline demonstrations (`DFT_FORGE_EXECUTOR=fake`).
  A missing local `pw.x` is a configuration error; it must not silently produce
  simulated results. `DFT_FORGE_WALLTIME` controls both local and SSH limits.

### Verification
- SCF convergence
- Ionic convergence
- Maximum force threshold
- Pressure threshold
- Cell sanity
- Parser data-quality checks

### Recovery
- Bounded recovery with max attempts
- Safe parameter adjustments: ecut, nbnd, k-points, smearing/occupations
- Failure classification: scf_not_converged, ionic_not_converged, timeout, bad_occupations, insufficient_nbnd, missing_restart, invalid_structure, inconsistent_prefix, missing_output, unknown

### Ledger
- SQLite-backed run records
- Query by task_id
- Failure pattern summaries

## Agent / Tool Interface

### CLI Design Philosophy
- Each command does one thing.
- Default output is JSON.
- Commands are composable for agents.

### Tool Registry
The built-in tool registry exposes:
- `structure.import`
- `input.build`
- `job.submit`
- `job.status`
- `result.parse`
- `result.verify`
- `ledger.query`
- `doctor`
- `workflow.convergence`
- `workflow.neutral_vacancy`
- `thermo.chemical_potential_scan`
- `workflow.magnetic_compare`
- `workflow.phonon_dfpt`

### MCP Server
```bash
python -m dft_forge.mcp.server
```

This exposes the same tools via MCP for Claude Code, Codex, Gemini, or other compatible agents.

### Planning and execution safety

The chat planner separates scientific reasoning from executable tools. A request
that says `只规划` / `不要运行` (or the equivalent English wording) is forced
into plan-only mode and cannot invoke workspace-writing or compute tools. Plans
marked `partial` or `unsupported`, unknown capabilities, and workflows over the
safe action budget are reported without running their executable prefix.

`structure.dope` is strictly elemental substitution. Use
`structure.supercell` for an unchanged comparison cell and
`structure.vacancy` for removing a bulk atom. Framework-external operations can
still be described as advisory scientific steps, but they cannot be executed
until a deterministic, schema-validated tool is registered.

### Web Chat
```bash
python -m dft_forge.web
```

Opens a FastAPI backend and chat UI for interactive use.

The **证据 / 任务中心** panel lists both template runs and graphs created by
composite workflows. Select a run to inspect its node states, attempts,
failure messages, and scalar results. Retry targets that selected run;
resume continues a paused/interrupted graph while keeping successful nodes.
Stopping a task also stops later steps in the agent's plan. A page reload
does not start another calculation: the running task remains discoverable
and can be stopped from the workspace.

New runs save an immutable per-run graph template and resolved repair
parameters in SQLite. Old dynamic workflows without a saved template remain
inspectable, but cannot be resumed by guessing their original graph. Graph
recovery restores node execution and outputs; it does not replay the original
chat plan or regenerate a composite workflow's scientific summary.

Session operations are coordinated within **one server process**. Start the
web server with a single Uvicorn worker; multi-worker coordination is not
implemented. Session paths are local and persisted under `sessions/` by
default (`DFT_FORGE_WEB_WORKDIR` can override this location).

The method card's **科研** mode requires explicit approval before a plan can
invoke a computation tool, and requires a local or SSH backend. **演示** mode
requires the explicit fake backend. These checks apply to the actual planned
tools, so an unfamiliar phrasing cannot bypass confirmation; help and other
read-only requests remain available without approval. Web requests that omit
`run_config` default to research mode. Direct CLI/AgentLoop integrations retain
their existing interface.

Method parameters are validated before a session is changed: finite positive
`ecutwfc`, positive integer `nbnd`, at least 10 band path points, and six integer
k-point values (three positive mesh counts followed by three 0/1 offsets).
An explicit `execution_mode` and method overrides apply only to the current
request; a previous plan-only request does not silently disable future runs.

After planning, the method is checked against each step's actual capability.
Unsupported custom fields are rejected before any preparation runs. Composite
workflows and mismatches with the generic card return
`method_confirmation_required` with `method_contract` and `method_token` in the
409 response or SSE error. Confirmation retries the same request with that
token; changes to the plan, referenced structure contents, or selected compute
configuration require a new confirmation. The token detects stale choices; it
is not an authentication credential.

The contract shows applied fields, inapplicable preset fields, automatic
resolution policies and convergence sweep points. A single cutoff or k-grid
never silently replaces a convergence scan. With automatic parameters enabled,
known structures receive resolved values before confirmation, which are locked
for calculation. Legacy/deferred choices are resolved before input compilation;
inspect generated input files for the actual numbers. Per-run evidence retains the confirmed policy beside stored
node parameters and input-file hashes. None of these records certifies
scientific convergence.

The Web UI now enables **task-aware parameter recommendation** by default
(`run_config.auto_parameters=true`). Sending a calculation first plans it and
returns a concrete method card; no calculation starts before that card is
confirmed. A configured LLM receives the selected accuracy, scientific task,
structure summary and deterministic parameter candidates. Offline mode uses
the rule planner and the same resolver. User method-card overrides take
priority; unverified model guesses do not replace resolved defaults.

For the built-in SCF, relaxation, bands and DOS graphs, the recommendation
resolves cutoff, density cutoff, SCF k mesh and the applicable band parameters
from the actual material/structure and pseudopotentials. The card records
per-parameter reasons, pseudo SHA-256 hashes and whether any sensitivity scan
has run. UPF v1/v2 and QE GTH valence headers are supported. `nbnd` means the
**total number of bands**, occupied plus empty. For the built-in two-atom Si
cell, the balanced initial settings are 25/200 Ry, a 4×4×4 SCF mesh, 12 bands
and 100 band-path points; these are initial settings, not convergence claims.
An explicit cutoff override also resolves a matching density cutoff. Uploaded
structures use reciprocal-space spacing and one unshifted point along
non-periodic directions. DOS NSCF sampling doubles the confirmed SCF mesh.

Confirmation freezes the plan rather than invoking the LLM again. Invalid
tokens cannot consume a valid pending confirmation, and changed configuration,
structure, template or pseudopotentials require a fresh preview. Each QE node
checks the confirmed source hashes. Automatic-parameter graph runs keep their
confirmed values on failure; parameter-changing repair requires a new preview.
Legacy API requests with `auto_parameters=false` retain their existing policy.

The optional **先检查收敛** setting (`converge_first=true`) supports a single
SCF, bands or DOS graph with a known structure. It runs a bounded 3×3
cutoff/k-mesh scan (only cutoff for isolated non-periodic systems), keeping
density cutoff, band count and occupation/smearing choices fixed. The density
cutoff is displayed at the value required by the upper end of the scan and
retained by the target calculation. A candidate must lie inside a stable
rectangle of higher-cutoff/denser-grid points, within 1 meV/atom of the scan
reference; the uppermost reference point cannot pass by comparing to itself.
Failed/cancelled points or absence of that plateau stop the target calculation.
Both scan and target retain separate run evidence and simulated scans remain
labelled simulated. This is a bounded total-energy sensitivity check, not proof
of band-gap, force, phonon or out-of-grid convergence.
Multi-graph requests must be split to keep the nine-point budget per request.
In automatic mode, the legacy standalone `workflow.convergence` is rejected
with guidance to use this setting on one SCF/bands/DOS task; it does not offer
the same fixed-method and interior-plateau guarantees.

For structures produced by an earlier step, the card explicitly confirms an
automatic resolution policy; actual numbers are resolved from that step's
successful output and saved before calculation. Generate the structure first
to preview a concrete convergence grid. Composite vacancy/magnetic/phonon
workflows retain their step-specific parameter policies and are labelled
deferred rather than being assigned parameters for a guessed primitive cell.
Custom QE templates must bind `_parameters_locked` and `_source_lock` inputs
to their QE nodes before opting into automatic-parameter previews.

Scientific result payloads carry a normalized `provenance` alongside charts
and metrics. Recorded graph execution is distinguished from parsed files,
user-entered numbers, and derived results; changing the current backend does
not rewrite historical provenance. Chained thermodynamic calculations and
comparisons preserve their input sources. Simulation warnings are added after
model narration, so the narrator cannot remove them. A real compute source is
still not a claim of independent scientific validation.

The artifact panel exposes scientific inputs and outputs, including filenames
with Unicode or URL-reserved characters. Session history, hidden files,
symbolic links and live SQLite state are not downloadable artifacts. Use the
run APIs and reproducibility manifest to inspect state; downloading an active
database file would not provide a consistent backup.

Run-center API:

- `GET /api/sessions/{session_id}/runs` — list this session's graphs.
- `GET /api/sessions/{session_id}/runs/{run_id}` — inspect nodes and allowed actions.
- `GET /api/sessions/{session_id}/runs/{run_id}/evidence` — download a scoped JSON evidence snapshot: original template, stored resolved parameters, scalar results, execution provenance and this run's artifact hashes. No other runs or chat history are included. Legacy missing templates remain explicitly unavailable; live runs and non-atomic file hashes are marked. Execution success is **not** scientific validation, and exports should be reviewed before sharing.
- `POST /api/sessions/{session_id}/runs/action` with `action`, `run_id`,
  and `background: true` — start recovery and return `202`; omit `background`
  to retain the synchronous response.
- `GET /api/sessions/{session_id}/running` — inspect the current operation.
- `POST /api/sessions/{session_id}/cancel` — request a stop for this session.

## LLM Configuration

By default DFT-Forge runs fully offline with the deterministic `dummy` planner.
To let a real LLM turn natural-language requests into validated workflows,
point it at any OpenAI-compatible API (OpenAI, StepFun, DeepSeek, Moonshot,
or a local vLLM/Ollama server):

```bash
export DFT_FORGE_LLM_PROVIDER=openai
export DFT_FORGE_LLM_BASE_URL=https://api.openai.com/v1   # or your provider
export DFT_FORGE_LLM_API_KEY=sk-...                        # keep out of git
export DFT_FORGE_LLM_MODEL=gpt-4o-mini                     # any chat model
```

The LLM is only asked to emit a JSON `WorkflowIR`; the reply is schema-validated
(materials, step types, dependencies) before anything is executed, and an invalid
reply triggers one repair round. If planning fails, nothing runs.

## Material Library

57 ready-to-run materials (3 hand-tuned built-ins + 54 MP-verified library
entries) covering elemental semiconductors, III-V / II-VI compounds, alkali
halides, simple and transition metals, and simple oxides:

- Structures: Materials Project ground-state cells (cell parameters + fractional
  sites), rebuilt exactly via ASE — including hexagonal (wurtzite/hcp) and
  low-symmetry cells.
- Pseudopotentials: GBRV USPP PBE v1.5 (65 elements, SHA256-verified). Element
  symlinks live at `assets/pseudos/<Element>.upf`.
- Any library material works from natural language end to end, e.g.
  `agent.solve_from_prompt("optimize NaCl structure")` plans, registers a task
  on the fly, runs pw.x, and verifies convergence.
- Regenerate the library after updating the source YAML:
  `python scripts/gen_material_library.py <m1_resolved.yaml>`.

Cutoffs/k-points defaults are heuristic per material family (harder first-row
elements get higher ecut; metals get denser k-meshes and stronger smearing).

## License and Third-Party Notices

- DFT-Forge code in this repository is provided under the repository root license.
- Quantum ESPRESSO is licensed under the GPL. This project does not redistribute QE binaries; users must provide their own installation.
- Pseudopotentials in `assets/pseudos/` are subject to their original distribution licenses. Check individual file headers before redistribution.

## Notes

- This is a local research/engineering project, not a benchmark runner.
- The LLM only decides what to do; deterministic code decides how.
- LLM-generated outputs are validated before execution.
- SSH credentials are expected to come from system SSH config, SSH agent, or environment variables; they are not stored in code, logs, or reports.
- Do not push/commit without explicit authorization.
