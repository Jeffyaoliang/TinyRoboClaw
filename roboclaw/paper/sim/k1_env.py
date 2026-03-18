"""K1 real robot environment via ROS2 interface.

Interfaces with the Unitree K1 dual-arm robot through ROS2 services and topics:
- Joint control: /ctrl/move_to_qpos (ArrayCommand service)
- Gripper control: /ctrl/set_gripper (ArrayCommand service)
- Joint state: /qpos_0, /qpos_1 (Joint7 messages)
- Camera: head camera image topics
- End-effector: /ee_xquat_0, /ee_xquat_1 (Xquat messages)

Requires: ros2 environment sourced, knowin_controller nodes running.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from roboclaw.paper.sim.base_env import ActionChunk, BaseEnvironment, Observation


@dataclass
class K1Config:
    """Configuration for K1 robot environment."""

    # Which arm to use: 0 = left, 1 = right
    arm_id: int = 0
    num_joints: int = 7

    # Home joint configuration (radians) — safe starting pose
    home_qpos: list[float] = field(default_factory=lambda: [0.0, 0.5, 0.0, -1.0, 0.0, 0.5, 0.0])

    # Control timing
    move_time_ms: int = 1000  # milliseconds per joint command
    step_time_ms: int = 200  # milliseconds per action step (faster for chunk execution)

    # Camera topic namespace: "head", "left_hand", or "right_hand"
    camera_namespace: str = "head"
    image_topic_suffix: str = "image_left_raw"
    image_size: tuple[int, int] = (480, 640)  # (H, W) from camera

    # Gripper
    gripper_open_angle: float = 1000.0  # motor units (fully open)
    gripper_close_angle: float = 0.0
    gripper_max_current: float = 200.0
    gripper_time_ms: int = 500

    # Safety
    max_joint_delta: float = 0.3  # max radians per step (safety clamp)

    # ROS2 service/topic names (auto-generated from arm_id)
    @property
    def qpos_topic(self) -> str:
        return f"/qpos_{self.arm_id}"

    @property
    def ee_xquat_topic(self) -> str:
        return f"/ee_xquat_{self.arm_id}"

    @property
    def image_topic(self) -> str:
        return f"/{self.camera_namespace}/{self.image_topic_suffix}"


class K1Env(BaseEnvironment):
    """Real robot environment for Unitree K1 via ROS2.

    Usage:
        # Ensure ROS2 nodes are running:
        #   ros2 launch knowin_controller launcher.py
        #
        # Then:
        env = K1Env(K1Config(arm_id=0))
        obs = env.reset()
        action = ActionChunk(joint_targets=..., gripper_targets=...)
        obs, info = env.step(action)
    """

    def __init__(self, config: K1Config | None = None):
        self.config = config or K1Config()
        self._node = None
        self._qpos_sub = None
        self._image_sub = None
        self._ctrl_client = None
        self._gripper_client = None

        # Cached state
        self._current_qpos = np.zeros(self.config.num_joints, dtype=np.float32)
        self._current_gripper = 1.0  # normalized: 0=closed, 1=open
        self._current_image: np.ndarray | None = None
        self._objects: dict[str, np.ndarray] = {}

        self._initialized = False

    def _ensure_ros_init(self) -> None:
        """Lazy ROS2 initialization — only when actually needed."""
        if self._initialized:
            return

        try:
            import rclpy
            from rclpy.node import Node

            if not rclpy.ok():
                rclpy.init()

            self._node = rclpy.create_node("k1_roboclaw_env")
            self._setup_subscribers()
            self._setup_clients()
            self._initialized = True

        except ImportError:
            raise RuntimeError(
                "ROS2 (rclpy) not available. "
                "Source your ROS2 workspace before running: "
                "source /opt/ros/humble/setup.bash && source ~/k1-ros2/install/setup.bash"
            )

    def _setup_subscribers(self) -> None:
        """Subscribe to joint state and camera topics."""
        from interfaces.msg import Joint7
        from sensor_msgs.msg import Image

        self._qpos_sub = self._node.create_subscription(
            Joint7,
            self.config.qpos_topic,
            self._qpos_callback,
            10,
        )

        self._image_sub = self._node.create_subscription(
            Image,
            self.config.image_topic,
            self._image_callback,
            1,
        )

    def _setup_clients(self) -> None:
        """Create service clients for control."""
        from interfaces.srv import ArrayCommand

        self._ctrl_client = self._node.create_client(
            ArrayCommand, "/ctrl/move_to_qpos"
        )
        self._gripper_client = self._node.create_client(
            ArrayCommand, "/ctrl/set_gripper"
        )

        # Wait for services
        for client, name in [
            (self._ctrl_client, "/ctrl/move_to_qpos"),
            (self._gripper_client, "/ctrl/set_gripper"),
        ]:
            if not client.wait_for_service(timeout_sec=5.0):
                raise RuntimeError(f"Service {name} not available. Is knowin_controller running?")

    def _qpos_callback(self, msg: Any) -> None:
        """Update cached joint positions from Joint7 message."""
        if msg.arm_id == self.config.arm_id:
            self._current_qpos = np.array(msg.angles[: self.config.num_joints], dtype=np.float32)

    def _image_callback(self, msg: Any) -> None:
        """Update cached image from sensor_msgs/Image."""
        # Convert ROS Image to numpy
        h, w = msg.height, msg.width
        if msg.encoding in ("rgb8", "bgr8"):
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3)
            if msg.encoding == "bgr8":
                img = img[:, :, ::-1].copy()  # BGR → RGB
        else:
            # Fallback for other encodings
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, -1)[:, :, :3]
        self._current_image = img

    def _spin_once(self, timeout_sec: float = 0.1) -> None:
        """Process pending ROS2 callbacks."""
        import rclpy

        rclpy.spin_once(self._node, timeout_sec=timeout_sec)

    async def _call_service(self, client: Any, arm_id: int, data: list[float], time_ms: int) -> Any:
        """Call an ArrayCommand service."""
        from interfaces.srv import ArrayCommand

        req = ArrayCommand.Request()
        req.arm_id = arm_id
        req.data = data
        req.running_time = time_ms

        future = client.call_async(req)

        # Spin until the future completes
        import rclpy

        while not future.done():
            rclpy.spin_once(self._node, timeout_sec=0.05)

        return future.result()

    # ── BaseEnvironment interface ──

    def reset(self) -> Observation:
        """Move arm to home position and open gripper."""
        self._ensure_ros_init()

        import asyncio

        loop = asyncio.get_event_loop()

        # Move to home position
        loop.run_until_complete(
            self._call_service(
                self._ctrl_client,
                self.config.arm_id,
                self.config.home_qpos,
                self.config.move_time_ms,
            )
        )

        # Open gripper
        loop.run_until_complete(
            self._call_service(
                self._gripper_client,
                self.config.arm_id,
                [self.config.gripper_open_angle, self.config.gripper_max_current],
                self.config.gripper_time_ms,
            )
        )
        self._current_gripper = 1.0

        # Wait for motion to complete and state to update
        time.sleep(self.config.move_time_ms / 1000.0 + 0.5)
        self._spin_once(0.5)

        return self.get_observation()

    def step(self, action: ActionChunk) -> tuple[Observation, dict[str, Any]]:
        """Execute an action chunk on the real robot."""
        self._ensure_ros_init()

        import asyncio

        loop = asyncio.get_event_loop()
        info: dict[str, Any] = {"steps_executed": 0}

        for t in range(action.chunk_size):
            target_joints = action.joint_targets[t]
            target_gripper = float(action.gripper_targets[t])

            # Safety: clamp joint deltas
            delta = target_joints - self._current_qpos
            delta = np.clip(delta, -self.config.max_joint_delta, self.config.max_joint_delta)
            safe_target = self._current_qpos + delta

            # Send joint command
            loop.run_until_complete(
                self._call_service(
                    self._ctrl_client,
                    self.config.arm_id,
                    safe_target.tolist(),
                    self.config.step_time_ms,
                )
            )

            # Send gripper command if significantly changed
            if abs(target_gripper - self._current_gripper) > 0.1:
                gripper_angle = (
                    self.config.gripper_open_angle * target_gripper
                    + self.config.gripper_close_angle * (1.0 - target_gripper)
                )
                loop.run_until_complete(
                    self._call_service(
                        self._gripper_client,
                        self.config.arm_id,
                        [gripper_angle, self.config.gripper_max_current],
                        self.config.gripper_time_ms,
                    )
                )
                self._current_gripper = target_gripper

            # Wait for step and update state
            time.sleep(self.config.step_time_ms / 1000.0)
            self._spin_once(0.1)

            info["steps_executed"] += 1

        return self.get_observation(), info

    def get_observation(self) -> Observation:
        """Get current observation from robot sensors."""
        self._ensure_ros_init()
        self._spin_once(0.2)

        image = self.capture_image()
        return Observation(
            image=image,
            joint_positions=self._current_qpos.copy(),
            gripper_open=self._current_gripper,
            timestamp=time.time(),
        )

    def capture_image(self) -> np.ndarray:
        """Capture current camera image."""
        self._ensure_ros_init()
        self._spin_once(0.1)

        if self._current_image is not None:
            return self._current_image.copy()

        # Return placeholder if no image received yet
        h, w = self.config.image_size
        return np.zeros((h, w, 3), dtype=np.uint8)

    def get_joint_positions(self) -> np.ndarray:
        """Get current joint positions."""
        self._ensure_ros_init()
        self._spin_once(0.05)
        return self._current_qpos.copy()

    def get_gripper_state(self) -> float:
        """Get gripper openness: 0.0 = closed, 1.0 = open."""
        return self._current_gripper

    def get_object_positions(self) -> dict[str, np.ndarray]:
        """Object positions — requires external perception (VLM/tracking).

        Returns cached positions if available, otherwise empty.
        Integration with knowin_perception tracking can be added.
        """
        return {k: v.copy() for k, v in self._objects.items()}

    def update_object_positions(self, positions: dict[str, np.ndarray]) -> None:
        """Update tracked object positions (called externally by perception)."""
        self._objects = {k: np.array(v, dtype=np.float32) for k, v in positions.items()}

    # ── Additional K1-specific methods ──

    def move_to_cartesian(self, position: list[float], quaternion: list[float]) -> None:
        """Move end-effector to a Cartesian pose via /ctrl/move_to_xquat."""
        self._ensure_ros_init()

        import asyncio
        from interfaces.srv import ArrayCommand

        xquat_client = self._node.create_client(ArrayCommand, "/ctrl/move_to_xquat")
        if not xquat_client.wait_for_service(timeout_sec=3.0):
            raise RuntimeError("/ctrl/move_to_xquat service not available")

        data = position + quaternion  # [x, y, z, qx, qy, qz, qw]
        loop = asyncio.get_event_loop()
        loop.run_until_complete(
            self._call_service(xquat_client, self.config.arm_id, data, self.config.move_time_ms)
        )
        time.sleep(self.config.move_time_ms / 1000.0 + 0.3)
        self._spin_once(0.2)

    def enable_arm(self) -> None:
        """Enable arm motors."""
        self._ensure_ros_init()

        from std_srvs.srv import Trigger

        client = self._node.create_client(Trigger, "/ctrl/enable_arm")
        if client.wait_for_service(timeout_sec=3.0):
            future = client.call_async(Trigger.Request())
            import rclpy

            rclpy.spin_until_future_complete(self._node, future, timeout_sec=5.0)

    def disable_arm(self) -> None:
        """Disable arm motors (enter teach mode)."""
        self._ensure_ros_init()

        from std_srvs.srv import Trigger

        client = self._node.create_client(Trigger, "/ctrl/disable_arm")
        if client.wait_for_service(timeout_sec=3.0):
            future = client.call_async(Trigger.Request())
            import rclpy

            rclpy.spin_until_future_complete(self._node, future, timeout_sec=5.0)

    def destroy(self) -> None:
        """Clean up ROS2 resources."""
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
            self._initialized = False
