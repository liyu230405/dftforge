# DFT-Forge

**Deterministic Quantum ESPRESSO Scientific Computing Agent**

> "LLM 只决定做什么；确定性程序决定输入文件如何生成、程序如何运行、结果如何解析和是否可信。"

## What is DFT-Forge?

DFT-Forge is an agent-friendly CLI and web toolchain for Quantum ESPRESSO (QE) DFT calculations. It is designed to be used by coding agents, chat UIs, and researchers who want structured, inspectable, and recoverable materials-computation workflows:

- **Agent-friendly CLI**: every command does one thing and outputs JSON.
- **Deterministic compiler**: generates QE input files using ASE/pymatgen/spglib/seekpath.
- **Deterministic executor**: runs QE with resource limits and full output preservation.
- **Deterministic parser**: extracts physical quantities from stdout/XML.
- **Deterministic verifier**: checks physical convergence, not just exit codes.
- **Evidence ledger**: SQLite-backed audit trail of runs, failures, and recovery actions.
- **Tool registry + MCP server**: structured tools that external agents can call.

This project is independent and written from scratch. It does not copy benchmark runners from other repos as its core identity; those repos are used only as external references.

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
│   ├── protocol/        # TaskSpec, WorkflowIR, PlanPatch, EvidenceBundle (schemas)
│   ├── compiler/        # Deterministic QE input generation
│   ├── executor.py      # Pluggable executor abstraction
│   ├── parser.py        # QE stdout/XML parsing
│   ├── verifier.py      # Physical convergence checks
│   ├── catalog/         # Material profiles and task definitions
│   ├── runner.py        # Deterministic task execution
│   ├── recovery.py      # Failure classification and bounded recovery
│   ├── ledger.py        # SQLite evidence ledger
│   ├── structure.py     # Structure import/validation
│   ├── llm.py           # LLM provider abstraction
│   ├── planner.py       # WorkflowIR generation
│   ├── agent.py         # Agent solve loop
│   ├── tools/           # Tool registry and tool definitions
│   ├── mcp/             # MCP server for external agents
│   ├── web/             # FastAPI chat backend + frontend
│   └── cli.py           # Agent-friendly CLI entry points
├── assets/pseudos/      # Pseudopotential files
├── tests/               # Pytest test suite
├── scripts/             # Utility scripts
├── outputs/             # Experiment outputs
└── work/                # Working files
```

## Supported Ranges

### Materials
- Built-in materials: Si, Al, MgO
- User-provided structures: CIF, POSCAR/CONTCAR, XYZ, QE input, explicit lattice+coordinates

### Calculations
- T1 vc-relax / relax
- T2 bands and DOS
- Generic SCF / NSCF

### Structure Operations
- Import and validate structures
- Normalize to primitive cell
- Nearest-neighbor sanity checks
- Space-group detection via spglib

### Executors
- Local executor
- SSH executor with scheduler support: Slurm, PBS
- Fake executor for offline testing

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

### MCP Server
```bash
python -m dft_forge.mcp.server
```

This exposes the same tools via MCP for Claude Code, Codex, Gemini, or other compatible agents.

### Web Chat
```bash
python -m dft_forge.web
```

Opens a FastAPI backend and chat UI for interactive use.

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
- This project does not redistribute or embed CatGo code. CatGo is used only as an external reference for architecture and UX ideas.

## Notes

- This is a local research/engineering project, not a benchmark runner.
- The LLM only decides what to do; deterministic code decides how.
- LLM-generated outputs are validated before execution.
- SSH credentials are expected to come from system SSH config, SSH agent, or environment variables; they are not stored in code, logs, or reports.
- Do not push/commit without explicit authorization.
