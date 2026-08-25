"""Graph workflow runtime (CatGo-style kernel, clean-room Python).

Layers:
- states: node/run state machines with legal transition table
- graph: declarative templates + DAG validation (3-color DFS cycle detection)
- run: mutable run instances (GraphRun/NodeRun)
- store: SQLite state persistence with crash recovery
- scheduler: dependency-driven dispatch loop with concurrency limit + repair
- engine: GraphEngine facade (create/start/pause/resume/retry)
"""

from dft_forge.runtime.states import (
    NodeState,
    RunState,
    TERMINAL_NODE_STATES,
    TERMINAL_RUN_STATES,
    can_transition,
    determine_run_status,
)
from dft_forge.runtime.graph import GraphTemplate, NodeSpec, load_template
from dft_forge.runtime.run import ExecutionContext, GraphRun, NodeRun, ToolError, create_run
from dft_forge.runtime.store import SQLiteStateStore
from dft_forge.runtime.scheduler import GraphScheduler, find_ready_nodes, process_blocked
from dft_forge.runtime.engine import GraphEngine

__all__ = [
    "NodeState", "RunState", "TERMINAL_NODE_STATES", "TERMINAL_RUN_STATES",
    "can_transition", "determine_run_status",
    "GraphTemplate", "NodeSpec", "load_template",
    "GraphRun", "NodeRun", "ExecutionContext", "ToolError", "create_run",
    "SQLiteStateStore", "GraphScheduler", "find_ready_nodes", "process_blocked",
    "GraphEngine",
]
