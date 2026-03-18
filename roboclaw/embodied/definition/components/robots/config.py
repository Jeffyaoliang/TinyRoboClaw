"""Robot configs inspired by instance-oriented robot definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, kw_only=True)
class RobotConfig:
    """Backend-agnostic robot instance config."""

    instance_id: str | None = None
    calibration_dir: str | None = None
    base_frame: str | None = None
    tool_frame: str | None = None
    extra: tuple[tuple[str, Any], ...] = field(default_factory=tuple)


@dataclass(frozen=True, kw_only=True)
class Ros2RobotConfig(RobotConfig):
    """Robot config for ROS2-exposed robot instances."""

    namespace: str | None = None
    joint_state_topic: str | None = None
    command_action: str | None = None
