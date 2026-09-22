"""Core data classes for the tool-first architecture."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

_VALID_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")

VALID_OUTPUT_TYPES = frozenset({
    "text", "json", "structure", "image",
})

VALID_CATEGORIES = frozenset({
    "general", "structure", "input", "job", "result", "ledger",
})

# Tool effects are part of the execution contract, not merely UI metadata.
# Unknown effects fail closed in plan-only mode and in future authorization
# layers, while keeping ordinary registry calls backwards compatible.
VALID_SIDE_EFFECTS = frozenset({
    "read_only", "workspace_write", "external_execution", "unknown",
})


def validate_tool_id(tool_id: str) -> bool:
    return bool(_VALID_ID.match(tool_id))


@dataclass
class ToolResult:
    data: dict
    output_type: str
    tool_id: str
    error: Optional[str] = None
    traceback: Optional[str] = None
    session_id: Optional[str] = None


@dataclass
class ToolEntry:
    id: str
    name: str
    description: str
    version: str = "1.0.0"
    author: str = ""
    category: str = "general"
    input_schema: dict = field(default_factory=dict)
    output_type: str = "json"
    trust: str = "builtin"
    permissions: list[str] = field(default_factory=list)
    side_effect: str = "unknown"
    semantic_contract: str = ""
    deferred_inputs: list[str] = field(default_factory=list)
    source: str = "code"
    path: Optional[Path] = None
    execute_fn: Optional[Callable] = field(default=None, repr=False)
    extra_fns: dict[str, Callable] = field(default_factory=dict, repr=False)
    frontend: Optional[dict] = None
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.side_effect not in VALID_SIDE_EFFECTS:
            raise ValueError(
                f"Invalid side_effect for {self.id}: {self.side_effect!r}; "
                f"expected one of {sorted(VALID_SIDE_EFFECTS)}"
            )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "author": self.author,
            "category": self.category,
            "input_schema": self.input_schema,
            "output_type": self.output_type,
            "trust": self.trust,
            "permissions": self.permissions,
            "side_effect": self.side_effect,
            "semantic_contract": self.semantic_contract,
            "deferred_inputs": self.deferred_inputs,
            "enabled": self.enabled,
            "source": self.source,
        }
