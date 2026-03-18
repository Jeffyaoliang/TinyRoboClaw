"""Execution target models."""

from __future__ import annotations

from dataclasses import dataclass, field

from roboclaw.embodied.definition.foundation.schema import CarrierKind, SimulatorKind, TransportKind
from roboclaw.embodied.execution.integration.transports.ros2 import Ros2InterfaceBundle


@dataclass(frozen=True)
class ExecutionTarget:
    """A concrete real or simulated carrier reached through one transport."""

    id: str
    carrier: CarrierKind
    transport: TransportKind
    description: str
    simulator: SimulatorKind = SimulatorKind.NONE
    ros2: Ros2InterfaceBundle | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)
