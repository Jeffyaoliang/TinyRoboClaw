"""Sensor registry."""

from __future__ import annotations

from roboclaw.embodied.definition.components.sensors.model import SensorManifest


class SensorRegistry:
    """Simple in-memory sensor registry."""

    def __init__(self) -> None:
        self._entries: dict[str, SensorManifest] = {}

    def register(self, manifest: SensorManifest) -> None:
        if manifest.id in self._entries:
            raise ValueError(f"Sensor '{manifest.id}' is already registered.")
        self._entries[manifest.id] = manifest

    def get(self, sensor_id: str) -> SensorManifest:
        try:
            return self._entries[sensor_id]
        except KeyError as exc:
            raise KeyError(f"Unknown sensor '{sensor_id}'.") from exc

    def list(self) -> tuple[SensorManifest, ...]:
        return tuple(sorted(self._entries.values(), key=lambda item: item.id))
