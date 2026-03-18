"""Domain bridge contracts for execution integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from roboclaw.embodied.definition.foundation.schema import CapabilityFamily, RobotType

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - Python < 3.11 fallback for local tooling.
    class StrEnum(str, Enum):
        """Fallback for Python versions without enum.StrEnum."""


class BridgeDomain(StrEnum):
    """Top-level bridge domains to avoid arm-centric integration."""

    ARM_HAND = "arm_hand"
    HUMANOID_WHOLE_BODY = "humanoid_whole_body"
    MOBILE_BASE_FLEET = "mobile_base_fleet"
    DRONE = "drone"
    SIMULATOR = "simulator"


class BridgeKind(StrEnum):
    """Execution bridge kind within one domain."""

    ROS2_CONTROL = "ros2_control"
    WHOLE_BODY_CONTROLLER = "whole_body_controller"
    NAV2_RMF = "nav2_rmf"
    MAVSDK_MAVLINK = "mavsdk_mavlink"
    SIM_RUNTIME = "sim_runtime"
    CUSTOM = "custom"


@dataclass(frozen=True)
class ControlSurfaceSpec:
    """Control command surface exposed by one bridge."""

    id: str
    mode: str
    interface: str
    description: str

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("Control surface id cannot be empty.")
        if not self.mode.strip():
            raise ValueError(f"Control surface '{self.id}' mode cannot be empty.")
        if not self.interface.strip():
            raise ValueError(f"Control surface '{self.id}' interface cannot be empty.")
        if not self.description.strip():
            raise ValueError(f"Control surface '{self.id}' description cannot be empty.")


@dataclass(frozen=True)
class ObservationSurfaceSpec:
    """Observation stream surface exposed by one bridge."""

    id: str
    stream: str
    interface: str
    description: str
    expected_rate_hz: float | None = None

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("Observation surface id cannot be empty.")
        if not self.stream.strip():
            raise ValueError(f"Observation surface '{self.id}' stream cannot be empty.")
        if not self.interface.strip():
            raise ValueError(f"Observation surface '{self.id}' interface cannot be empty.")
        if not self.description.strip():
            raise ValueError(f"Observation surface '{self.id}' description cannot be empty.")
        if self.expected_rate_hz is not None and self.expected_rate_hz <= 0:
            raise ValueError(
                f"Observation surface '{self.id}' expected_rate_hz must be > 0 when specified."
            )


@dataclass(frozen=True)
class DomainBridgeContract:
    """Contract describing one domain bridge profile."""

    id: str
    domain: BridgeDomain
    kind: BridgeKind
    description: str
    supported_robot_types: tuple[RobotType, ...]
    supported_capabilities: tuple[CapabilityFamily, ...]
    control_surfaces: tuple[ControlSurfaceSpec, ...] = field(default_factory=tuple)
    observation_surfaces: tuple[ObservationSurfaceSpec, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("Bridge id cannot be empty.")
        if not self.description.strip():
            raise ValueError(f"Bridge '{self.id}' description cannot be empty.")
        if not self.supported_robot_types:
            raise ValueError(f"Bridge '{self.id}' must declare supported robot types.")
        if not self.supported_capabilities:
            raise ValueError(f"Bridge '{self.id}' must declare supported capabilities.")
        if not (self.control_surfaces or self.observation_surfaces):
            raise ValueError(
                f"Bridge '{self.id}' must define at least one control or observation surface."
            )
        control_ids = [surface.id for surface in self.control_surfaces]
        if len(set(control_ids)) != len(control_ids):
            raise ValueError(f"Bridge '{self.id}' has duplicate control surface ids.")
        observation_ids = [surface.id for surface in self.observation_surfaces]
        if len(set(observation_ids)) != len(observation_ids):
            raise ValueError(f"Bridge '{self.id}' has duplicate observation surface ids.")

    def supports_robot_type(self, robot_type: RobotType) -> bool:
        """Return whether this bridge supports the given robot category."""

        return robot_type in set(self.supported_robot_types)

    def supports_capability(self, capability: CapabilityFamily) -> bool:
        """Return whether this bridge supports the given capability family."""

        return capability in set(self.supported_capabilities)

