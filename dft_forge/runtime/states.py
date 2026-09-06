"""Node and run state machines with legal-transition enforcement."""

from __future__ import annotations

from enum import Enum
from typing import Dict, Iterable, Set


class NodeState(str, Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    REPAIRING = "repairing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class RunState(str, Enum):
    CREATED = "created"
    VALIDATED = "validated"
    RUNNING = "running"
    PAUSED = "paused"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PARTIALLY_SUCCEEDED = "partially_succeeded"


TERMINAL_NODE_STATES: Set[NodeState] = {
    NodeState.SUCCEEDED,
    NodeState.FAILED,
    NodeState.BLOCKED,
    NodeState.SKIPPED,
    NodeState.CANCELLED,
}

TERMINAL_RUN_STATES: Set[RunState] = {
    RunState.SUCCEEDED,
    RunState.FAILED,
    RunState.CANCELLED,
    RunState.PARTIALLY_SUCCEEDED,
}

ALLOWED_NODE_TRANSITIONS: Dict[NodeState, Set[NodeState]] = {
    NodeState.PENDING: {NodeState.READY, NodeState.BLOCKED, NodeState.SKIPPED, NodeState.CANCELLED},
    NodeState.READY: {NodeState.RUNNING, NodeState.SKIPPED, NodeState.CANCELLED},
    NodeState.RUNNING: {NodeState.SUCCEEDED, NodeState.FAILED, NodeState.REPAIRING, NodeState.READY, NodeState.CANCELLED},
    NodeState.REPAIRING: {NodeState.READY, NodeState.FAILED, NodeState.CANCELLED},
    NodeState.SUCCEEDED: set(),
    NodeState.FAILED: {NodeState.PENDING},  # explicit retry() only
    NodeState.BLOCKED: {NodeState.PENDING},  # explicit retry() only
    NodeState.SKIPPED: set(),
    NodeState.CANCELLED: {NodeState.PENDING},  # resume after pause
}


def can_transition(source: NodeState, target: NodeState) -> bool:
    return target in ALLOWED_NODE_TRANSITIONS.get(source, set())


def determine_run_status(states: Iterable[NodeState]) -> RunState:
    """Final run status from node states (priority order).

    all succeeded > cancelled-without-success > any succeeded > failed.
    """
    node_states = list(states)
    if not node_states:
        return RunState.FAILED
    all_succeeded = all(s == NodeState.SUCCEEDED for s in node_states)
    any_succeeded = any(s == NodeState.SUCCEEDED for s in node_states)
    any_cancelled = any(s == NodeState.CANCELLED for s in node_states)
    if all_succeeded:
        return RunState.SUCCEEDED
    if any_cancelled and not any_succeeded:
        return RunState.CANCELLED
    if any_succeeded:
        return RunState.PARTIALLY_SUCCEEDED
    return RunState.FAILED
