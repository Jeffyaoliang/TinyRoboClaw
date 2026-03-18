"""Builders for simulated execution targets."""

from __future__ import annotations

from roboclaw.embodied.execution.integration.carriers.model import ExecutionTarget
from roboclaw.embodied.definition.foundation.schema import CarrierKind, SimulatorKind, TransportKind
from roboclaw.embodied.execution.integration.transports.ros2 import Ros2InterfaceBundle


def build_sim_ros2_target(
    *,
    target_id: str,
    simulator: SimulatorKind,
    description: str,
    ros2: Ros2InterfaceBundle,
    notes: tuple[str, ...] = (),
) -> ExecutionTarget:
    """Build a simulation carrier target reached through ROS2."""

    return ExecutionTarget(
        id=target_id,
        carrier=CarrierKind.SIM,
        transport=TransportKind.ROS2,
        description=description,
        simulator=simulator,
        ros2=ros2,
        notes=notes,
    )
