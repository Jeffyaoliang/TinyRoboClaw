"""Domain bridge exports."""

from roboclaw.embodied.execution.integration.bridges.library import (
    ARM_HAND_BRIDGE,
    DEFAULT_DOMAIN_BRIDGES,
    DRONE_BRIDGE,
    HUMANOID_WHOLE_BODY_BRIDGE,
    MOBILE_BASE_FLEET_BRIDGE,
    SIMULATOR_BRIDGE,
)
from roboclaw.embodied.execution.integration.bridges.model import (
    BridgeDomain,
    BridgeKind,
    ControlSurfaceSpec,
    DomainBridgeContract,
    ObservationSurfaceSpec,
)
from roboclaw.embodied.execution.integration.bridges.registry import BridgeRegistry

__all__ = [
    "ARM_HAND_BRIDGE",
    "BridgeDomain",
    "BridgeKind",
    "BridgeRegistry",
    "ControlSurfaceSpec",
    "DEFAULT_DOMAIN_BRIDGES",
    "DRONE_BRIDGE",
    "DomainBridgeContract",
    "HUMANOID_WHOLE_BODY_BRIDGE",
    "MOBILE_BASE_FLEET_BRIDGE",
    "ObservationSurfaceSpec",
    "SIMULATOR_BRIDGE",
]

