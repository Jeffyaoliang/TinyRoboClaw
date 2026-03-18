"""Simulation world and scenario definitions."""

from __future__ import annotations

from dataclasses import dataclass, field

from roboclaw.embodied.definition.foundation.schema import SimulatorKind


@dataclass(frozen=True)
class SimulatorWorld:
    """Reusable simulation world definition."""

    id: str
    simulator: SimulatorKind
    description: str
    assets: tuple[str, ...] = field(default_factory=tuple)
    reset_modes: tuple[str, ...] = field(default_factory=lambda: ("default",))
    notes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class SimulatorScenario:
    """Concrete scenario binding a world to one assembly target."""

    id: str
    assembly_id: str
    target_id: str
    world_id: str
    description: str
    notes: tuple[str, ...] = field(default_factory=tuple)
