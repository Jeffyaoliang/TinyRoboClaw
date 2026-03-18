"""Sensor manifests."""

from __future__ import annotations

from dataclasses import dataclass, field

from roboclaw.embodied.definition.foundation.schema import SensorKind


@dataclass(frozen=True)
class SensorManifest:
    """Static sensor definition independent from robots and carriers."""

    id: str
    kind: SensorKind
    description: str
    mount_points: tuple[str, ...] = field(default_factory=tuple)
    default_topic_name: str | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)
