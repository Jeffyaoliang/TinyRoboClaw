"""Builders for real execution targets."""

from __future__ import annotations

from roboclaw.embodied.execution.integration.carriers.model import ExecutionTarget
from roboclaw.embodied.definition.foundation.schema import CarrierKind, TransportKind
from roboclaw.embodied.execution.integration.transports.ros2 import Ros2InterfaceBundle


def build_real_ros2_target(
    *,
    target_id: str,
    description: str,
    ros2: Ros2InterfaceBundle,
    notes: tuple[str, ...] = (),
) -> ExecutionTarget:
    """Build a real carrier target reached through ROS2."""

    return ExecutionTarget(
        id=target_id,
        carrier=CarrierKind.REAL,
        transport=TransportKind.ROS2,
        description=description,
        ros2=ros2,
        notes=notes,
    )
