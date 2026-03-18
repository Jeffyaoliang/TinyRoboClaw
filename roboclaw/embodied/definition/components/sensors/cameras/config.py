"""Camera configs inspired by backend-first camera definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CameraDriver(str, Enum):
    """Execution backend for one camera attachment."""

    ROS2 = "ros2"
    OPENCV = "opencv"
    REALSENSE = "intelrealsense"
    ZMQ = "zmq"


class CameraColorMode(str, Enum):
    """Color ordering for frames."""

    RGB = "rgb"
    BGR = "bgr"


class CameraRotation(int, Enum):
    """Image rotation applied by the backend."""

    NO_ROTATION = 0
    ROTATE_90 = 90
    ROTATE_180 = 180
    ROTATE_270 = -90


@dataclass(frozen=True, kw_only=True)
class CameraConfig:
    """Backend-agnostic camera initialization config."""

    driver: CameraDriver
    fps: int | None = None
    width: int | None = None
    height: int | None = None
    color_mode: CameraColorMode = CameraColorMode.RGB
    rotation: CameraRotation = CameraRotation.NO_ROTATION
    warmup_s: int = 1
    frame_id: str | None = None
    topic_namespace: str = "camera"
    image_topic_name: str = "image_raw"
    info_topic_name: str = "camera_info"
    extra: tuple[tuple[str, Any], ...] = field(default_factory=tuple)


@dataclass(frozen=True, kw_only=True)
class OpenCVCameraConfig(CameraConfig):
    """Camera config for OpenCV-backed devices or files."""

    index_or_path: int | str
    fourcc: str | None = None
    backend: str | None = None
    driver: CameraDriver = field(default=CameraDriver.OPENCV, init=False)


@dataclass(frozen=True, kw_only=True)
class RealSenseCameraConfig(CameraConfig):
    """Camera config for Intel RealSense devices."""

    serial_number_or_name: str
    use_depth: bool = False
    driver: CameraDriver = field(default=CameraDriver.REALSENSE, init=False)


@dataclass(frozen=True, kw_only=True)
class Ros2CameraConfig(CameraConfig):
    """Camera config for cameras exposed through ROS2 topics."""

    driver: CameraDriver = field(default=CameraDriver.ROS2, init=False)
