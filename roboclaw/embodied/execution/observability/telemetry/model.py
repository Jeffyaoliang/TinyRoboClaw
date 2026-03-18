"""Telemetry and trace records for embodied systems."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - Python < 3.11 fallback for local tooling.
    class StrEnum(str, Enum):
        """Fallback for Python versions without enum.StrEnum."""


class TelemetryKind(StrEnum):
    """Telemetry event kind."""

    STATE = "state"
    ACTION = "action"
    ERROR = "error"
    SENSOR = "sensor"
    DIAGNOSTIC = "diagnostic"


class TelemetrySeverity(StrEnum):
    """Severity level for one telemetry event."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class TelemetryPhase(StrEnum):
    """Lifecycle phase of one telemetry event."""

    START = "start"
    PROGRESS = "progress"
    COMPLETE = "complete"
    FAILURE = "failure"
    SNAPSHOT = "snapshot"


@dataclass(frozen=True)
class RawEvidenceHandle:
    """Handle to raw evidence associated with telemetry."""

    id: str
    uri: str
    media_type: str | None = None
    digest: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("RawEvidenceHandle id cannot be empty.")
        if not self.uri.strip():
            raise ValueError("RawEvidenceHandle uri cannot be empty.")
        if self.media_type is not None and not self.media_type.strip():
            raise ValueError("RawEvidenceHandle media_type cannot be empty when specified.")
        if self.digest is not None and not self.digest.strip():
            raise ValueError("RawEvidenceHandle digest cannot be empty when specified.")


@dataclass(frozen=True)
class TelemetryEvent:
    """One structured telemetry event."""

    timestamp: datetime
    correlation_id: str
    source_component: str
    kind: TelemetryKind
    severity: TelemetrySeverity
    message: str
    payload: dict[str, Any] = field(default_factory=dict)
    raw_evidence: tuple[RawEvidenceHandle, ...] = field(default_factory=tuple)
    subject: str | None = None
    phase: TelemetryPhase | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)
    replay_handle: str | None = None

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("TelemetryEvent timestamp must be timezone-aware.")
        if not self.correlation_id.strip():
            raise ValueError("TelemetryEvent correlation_id cannot be empty.")
        if not self.source_component.strip():
            raise ValueError("TelemetryEvent source_component cannot be empty.")
        if not self.message.strip():
            raise ValueError("TelemetryEvent message cannot be empty.")
        if self.subject is not None and not self.subject.strip():
            raise ValueError("TelemetryEvent subject cannot be empty when specified.")
        if self.replay_handle is not None and not self.replay_handle.strip():
            raise ValueError("TelemetryEvent replay_handle cannot be empty when specified.")

        for key in self.payload:
            if not key.strip():
                raise ValueError("TelemetryEvent payload keys cannot be empty.")
        for tag in self.tags:
            if not tag.strip():
                raise ValueError("TelemetryEvent tags cannot contain empty values.")


@dataclass(frozen=True)
class StateSnapshot:
    """Normalized runtime state snapshot."""

    timestamp: datetime
    correlation_id: str
    source_component: str
    assembly_id: str
    target_id: str
    status: str
    joints: dict[str, float] = field(default_factory=dict)
    sensors: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("StateSnapshot timestamp must be timezone-aware.")
        if not self.correlation_id.strip():
            raise ValueError("StateSnapshot correlation_id cannot be empty.")
        if not self.source_component.strip():
            raise ValueError("StateSnapshot source_component cannot be empty.")
        if not self.assembly_id.strip():
            raise ValueError("StateSnapshot assembly_id cannot be empty.")
        if not self.target_id.strip():
            raise ValueError("StateSnapshot target_id cannot be empty.")
        if not self.status.strip():
            raise ValueError("StateSnapshot status cannot be empty.")


@dataclass(frozen=True)
class ActionTrace:
    """Action execution trace."""

    timestamp: datetime
    correlation_id: str
    source_component: str
    action: str
    status: str
    severity: TelemetrySeverity = TelemetrySeverity.INFO
    phase: TelemetryPhase = TelemetryPhase.COMPLETE
    request: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)
    raw_evidence: tuple[RawEvidenceHandle, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("ActionTrace timestamp must be timezone-aware.")
        if not self.correlation_id.strip():
            raise ValueError("ActionTrace correlation_id cannot be empty.")
        if not self.source_component.strip():
            raise ValueError("ActionTrace source_component cannot be empty.")
        if not self.action.strip():
            raise ValueError("ActionTrace action cannot be empty.")
        if not self.status.strip():
            raise ValueError("ActionTrace status cannot be empty.")


def utcnow() -> datetime:
    """Return timezone-aware UTC timestamp for telemetry defaults."""

    return datetime.now(timezone.utc)
