"""Sensor exports."""

from roboclaw.embodied.definition.components.sensors.cameras import (
    CameraColorMode,
    CameraConfig,
    CameraDriver,
    CameraRotation,
    CameraSensorManifest,
    OpenCVCameraConfig,
    RGB_CAMERA,
    RealSenseCameraConfig,
    Ros2CameraConfig,
)
from roboclaw.embodied.definition.components.sensors.model import SensorManifest
from roboclaw.embodied.definition.components.sensors.registry import SensorRegistry

__all__ = [
    "CameraColorMode",
    "CameraConfig",
    "CameraDriver",
    "CameraRotation",
    "CameraSensorManifest",
    "OpenCVCameraConfig",
    "RGB_CAMERA",
    "RealSenseCameraConfig",
    "Ros2CameraConfig",
    "SensorManifest",
    "SensorRegistry",
]
