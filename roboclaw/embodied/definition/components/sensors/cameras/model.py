"""Camera sensor manifests."""

from __future__ import annotations

from dataclasses import dataclass, field

from roboclaw.embodied.definition.components.sensors.cameras.config import CameraDriver
from roboclaw.embodied.definition.components.sensors.model import SensorManifest


@dataclass(frozen=True)
class CameraSensorManifest(SensorManifest):
    """Reusable camera type independent from robot-specific mounting."""

    supported_drivers: tuple[CameraDriver, ...] = field(default_factory=tuple)
    supports_intrinsics: bool = True
    supports_depth: bool = False
