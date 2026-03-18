"""Deployment profiles for embodied systems."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DeploymentProfile:
    """Concrete environment profile for one assembly."""

    id: str
    assembly_id: str
    target_id: str
    connection: dict[str, Any] = field(default_factory=dict)
    robots: dict[str, dict[str, Any]] = field(default_factory=dict)
    sensors: dict[str, dict[str, Any]] = field(default_factory=dict)
    safety_overrides: dict[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = field(default_factory=tuple)
