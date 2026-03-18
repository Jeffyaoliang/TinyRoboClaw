"""Domain bridge registry."""

from __future__ import annotations

from roboclaw.embodied.definition.foundation.schema import CapabilityFamily, RobotType
from roboclaw.embodied.execution.integration.bridges.model import BridgeDomain, DomainBridgeContract


class BridgeRegistry:
    """Register reusable domain bridge contracts."""

    def __init__(self) -> None:
        self._entries: dict[str, DomainBridgeContract] = {}

    def register(self, bridge: DomainBridgeContract) -> None:
        if bridge.id in self._entries:
            raise ValueError(f"Bridge '{bridge.id}' is already registered.")
        self._entries[bridge.id] = bridge

    def get(self, bridge_id: str) -> DomainBridgeContract:
        try:
            return self._entries[bridge_id]
        except KeyError as exc:
            raise KeyError(f"Unknown bridge '{bridge_id}'.") from exc

    def list(self) -> tuple[DomainBridgeContract, ...]:
        return tuple(self._entries.values())

    def for_domain(self, domain: BridgeDomain) -> tuple[DomainBridgeContract, ...]:
        return tuple(entry for entry in self._entries.values() if entry.domain == domain)

    def for_robot_type(self, robot_type: RobotType) -> tuple[DomainBridgeContract, ...]:
        return tuple(
            entry for entry in self._entries.values() if entry.supports_robot_type(robot_type)
        )

    def for_capability(self, capability: CapabilityFamily) -> tuple[DomainBridgeContract, ...]:
        return tuple(
            entry for entry in self._entries.values() if entry.supports_capability(capability)
        )

