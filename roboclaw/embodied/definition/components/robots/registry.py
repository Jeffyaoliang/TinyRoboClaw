"""Robot registry."""

from __future__ import annotations

from roboclaw.embodied.definition.components.robots.model import RobotManifest
from roboclaw.embodied.definition.foundation.schema import RobotType


class RobotRegistry:
    """Simple in-memory robot registry."""

    def __init__(self) -> None:
        self._entries: dict[str, RobotManifest] = {}

    def register(self, manifest: RobotManifest) -> None:
        if manifest.id in self._entries:
            raise ValueError(f"Robot '{manifest.id}' is already registered.")
        self._entries[manifest.id] = manifest

    def get(self, robot_id: str) -> RobotManifest:
        try:
            return self._entries[robot_id]
        except KeyError as exc:
            raise KeyError(f"Unknown robot '{robot_id}'.") from exc

    def list(self) -> tuple[RobotManifest, ...]:
        return tuple(sorted(self._entries.values(), key=lambda item: item.id))

    def by_type(self, robot_type: RobotType) -> tuple[RobotManifest, ...]:
        return tuple(item for item in self.list() if item.robot_type == robot_type)
