"""Runtime state for active embodied sessions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - Python < 3.11 fallback for local tooling.
    class StrEnum(str, Enum):
        """Fallback for Python versions without enum.StrEnum."""


class RuntimeStatus(StrEnum):
    """Lifecycle state of one active embodied runtime."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"


@dataclass(frozen=True)
class RuntimeTask:
    """A task currently owned by the runtime."""

    id: str
    kind: str
    status: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RuntimeSession:
    """Live runtime view for one active embodied system."""

    id: str
    assembly_id: str
    target_id: str
    deployment_id: str | None = None
    adapter_id: str | None = None
    status: RuntimeStatus = RuntimeStatus.DISCONNECTED
    active_tasks: list[RuntimeTask] = field(default_factory=list)
    last_error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def mark_busy(self, task: RuntimeTask) -> None:
        self.status = RuntimeStatus.BUSY
        self.active_tasks.append(task)

    def clear_tasks(self) -> None:
        self.active_tasks.clear()
        if self.status == RuntimeStatus.BUSY:
            self.status = RuntimeStatus.READY
