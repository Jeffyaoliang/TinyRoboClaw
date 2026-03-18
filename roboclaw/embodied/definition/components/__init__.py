"""Component-layer exports for embodied definitions."""

from roboclaw.embodied.definition.components.robots import (
    PrimitiveSpec,
    RobotConfig,
    RobotManifest,
    RobotRegistry,
    Ros2RobotConfig,
    SO101_ROBOT,
)
from roboclaw.embodied.definition.components.sensors import (
    CameraColorMode,
    CameraConfig,
    CameraDriver,
    CameraRotation,
    CameraSensorManifest,
    OpenCVCameraConfig,
    RGB_CAMERA,
    RealSenseCameraConfig,
    Ros2CameraConfig,
    SensorManifest,
    SensorRegistry,
)

__all__ = [
    "CameraColorMode",
    "CameraConfig",
    "CameraDriver",
    "CameraRotation",
    "CameraSensorManifest",
    "OpenCVCameraConfig",
    "PrimitiveSpec",
    "RGB_CAMERA",
    "RobotConfig",
    "RealSenseCameraConfig",
    "RobotManifest",
    "RobotRegistry",
    "Ros2RobotConfig",
    "Ros2CameraConfig",
    "SO101_ROBOT",
    "SensorManifest",
    "SensorRegistry",
]
