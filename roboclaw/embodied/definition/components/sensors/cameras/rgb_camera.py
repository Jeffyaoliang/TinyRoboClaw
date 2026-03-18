"""Generic RGB camera manifest."""

from __future__ import annotations

from roboclaw.embodied.definition.components.sensors.cameras.config import CameraDriver
from roboclaw.embodied.definition.components.sensors.cameras.model import CameraSensorManifest
from roboclaw.embodied.definition.foundation.schema import SensorKind

RGB_CAMERA = CameraSensorManifest(
    id="rgb_camera",
    kind=SensorKind.CAMERA,
    description="Reusable RGB camera definition independent from robot, mount, and deployment.",
    mount_points=("wrist", "head", "external", "overhead"),
    default_topic_name="image_raw",
    supported_drivers=(
        CameraDriver.ROS2,
        CameraDriver.OPENCV,
        CameraDriver.REALSENSE,
        CameraDriver.ZMQ,
    ),
    supports_intrinsics=True,
    supports_depth=False,
    notes=(
        "Camera type stays generic; mount point, namespace, and device initialization belong to attachments or deployments.",
    ),
)
