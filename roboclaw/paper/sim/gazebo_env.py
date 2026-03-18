"""Gazebo (ROS2) simulation environment.

Interfaces with Gazebo through ROS2 topics and services,
matching the same interface used by real robots.

This is useful because:
1. Piper already has a Gazebo package (piper_gazebo in piper_ros)
2. K1's ROS2 stack can run identically in Gazebo
3. Same ROS2 topics → same K1Env code works for both sim and real

Dependencies:
    - ROS2 Humble/Iron
    - Gazebo (Ignition/Classic)
    - Robot-specific Gazebo packages (piper_gazebo, etc.)

Usage:
    # Terminal 1: Launch Gazebo with robot
    ros2 launch piper_gazebo piper_gazebo.launch.py

    # Terminal 2: Use this env
    env = GazeboEnv(GazeboConfig(
        joint_command_topic="/joint_position_controller/commands",
        joint_state_topic="/joint_states",
        camera_topic="/camera/image_raw",
    ))
    obs = env.reset()
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from roboclaw.paper.sim.base_env import ActionChunk, BaseEnvironment, Observation


@dataclass
class GazeboConfig:
    """Configuration for Gazebo ROS2 environment."""

    # ROS2 topics
    joint_state_topic: str = "/joint_states"
    joint_command_topic: str = "/joint_position_controller/commands"
    gripper_command_topic: str = "/gripper_controller/commands"
    camera_topic: str = "/camera/image_raw"

    # ROS2 services
    reset_service: str = "/reset_simulation"  # gazebo reset world
    pause_service: str = "/pause_physics"
    unpause_service: str = "/unpause_physics"

    # Robot parameters
    num_joints: int = 6
    joint_names: list[str] = field(default_factory=lambda: [
        "joint1", "joint2", "joint3", "joint4", "joint5", "joint6"
    ])
    home_qpos: list[float] = field(default_factory=lambda: [0.0, 0.3, -0.3, 0.0, 0.0, 0.0])

    # Timing
    step_delay: float = 0.05  # seconds between control steps
    reset_delay: float = 1.0  # seconds after reset to settle

    # Image
    image_size: tuple[int, int] = (480, 640)

    # Node name
    node_name: str = "gazebo_roboclaw_env"


class GazeboEnv(BaseEnvironment):
    """Gazebo simulation environment via ROS2.

    Communicates with Gazebo through standard ROS2 topics:
    - Subscribes to /joint_states for joint feedback
    - Publishes to joint position controller for commands
    - Subscribes to camera topic for images
    - Calls Gazebo services for reset/pause

    This makes it interchangeable with real robot environments
    that use the same ROS2 interface.
    """

    def __init__(self, config: GazeboConfig | None = None):
        self.config = config or GazeboConfig()
        self._node = None
        self._initialized = False

        # Cached state
        self._current_qpos = np.zeros(self.config.num_joints, dtype=np.float32)
        self._current_velocity = np.zeros(self.config.num_joints, dtype=np.float32)
        self._current_gripper = 1.0
        self._current_image: np.ndarray | None = None
        self._objects: dict[str, np.ndarray] = {}

        # ROS2 publishers/subscribers/clients
        self._joint_state_sub = None
        self._camera_sub = None
        self._joint_cmd_pub = None
        self._gripper_cmd_pub = None
        self._reset_client = None

    def _ensure_init(self) -> None:
        if self._initialized:
            return

        try:
            import rclpy
            from rclpy.node import Node

            if not rclpy.ok():
                rclpy.init()

            self._node = rclpy.create_node(self.config.node_name)
            self._setup_ros2()
            self._initialized = True

        except ImportError:
            raise RuntimeError(
                "ROS2 (rclpy) not available. Source your ROS2 workspace:\n"
                "  source /opt/ros/humble/setup.bash"
            )

    def _setup_ros2(self) -> None:
        """Set up ROS2 publishers, subscribers, and service clients."""
        from sensor_msgs.msg import JointState, Image
        from std_msgs.msg import Float64MultiArray
        from std_srvs.srv import Empty

        # Subscribe to joint states
        self._joint_state_sub = self._node.create_subscription(
            JointState,
            self.config.joint_state_topic,
            self._joint_state_callback,
            10,
        )

        # Subscribe to camera
        self._camera_sub = self._node.create_subscription(
            Image,
            self.config.camera_topic,
            self._camera_callback,
            1,
        )

        # Publish joint commands
        self._joint_cmd_pub = self._node.create_publisher(
            Float64MultiArray,
            self.config.joint_command_topic,
            10,
        )

        # Gripper publisher
        self._gripper_cmd_pub = self._node.create_publisher(
            Float64MultiArray,
            self.config.gripper_command_topic,
            10,
        )

        # Reset service client
        self._reset_client = self._node.create_client(Empty, self.config.reset_service)

    def _joint_state_callback(self, msg: Any) -> None:
        """Update joint positions from /joint_states."""
        # Map joint names to indices
        for i, target_name in enumerate(self.config.joint_names):
            if target_name in msg.name:
                idx = msg.name.index(target_name)
                self._current_qpos[i] = msg.position[idx]
                if idx < len(msg.velocity):
                    self._current_velocity[i] = msg.velocity[idx]

    def _camera_callback(self, msg: Any) -> None:
        """Update image from camera topic."""
        h, w = msg.height, msg.width
        if msg.encoding in ("rgb8", "bgr8"):
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3)
            if msg.encoding == "bgr8":
                img = img[:, :, ::-1].copy()
        else:
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, -1)[:, :, :3]
        self._current_image = img

    def _spin_once(self, timeout: float = 0.1) -> None:
        import rclpy
        rclpy.spin_once(self._node, timeout_sec=timeout)

    def _publish_joint_cmd(self, positions: np.ndarray) -> None:
        from std_msgs.msg import Float64MultiArray
        msg = Float64MultiArray()
        msg.data = positions.tolist()
        self._joint_cmd_pub.publish(msg)

    def _publish_gripper_cmd(self, value: float) -> None:
        from std_msgs.msg import Float64MultiArray
        msg = Float64MultiArray()
        msg.data = [value]
        self._gripper_cmd_pub.publish(msg)

    def _call_reset(self) -> None:
        """Call Gazebo reset_simulation service."""
        from std_srvs.srv import Empty

        if self._reset_client is None:
            return

        if not self._reset_client.wait_for_service(timeout_sec=3.0):
            logger_msg = "Gazebo reset service not available, skipping"
            return

        future = self._reset_client.call_async(Empty.Request())
        import rclpy
        rclpy.spin_until_future_complete(self._node, future, timeout_sec=5.0)

    # ── BaseEnvironment interface ──

    def reset(self) -> Observation:
        self._ensure_init()

        # Reset Gazebo simulation
        self._call_reset()
        time.sleep(0.5)

        # Move to home position
        home = np.array(self.config.home_qpos, dtype=np.float32)
        self._publish_joint_cmd(home)

        # Open gripper
        self._publish_gripper_cmd(1.0)
        self._current_gripper = 1.0

        # Wait to settle
        time.sleep(self.config.reset_delay)
        for _ in range(20):
            self._spin_once(0.05)

        return self.get_observation()

    def step(self, action: ActionChunk) -> tuple[Observation, dict[str, Any]]:
        self._ensure_init()
        info: dict[str, Any] = {"steps_executed": 0}

        for t in range(action.chunk_size):
            target = action.joint_targets[t]
            target_gripper = float(action.gripper_targets[t])

            self._publish_joint_cmd(target.astype(np.float32))

            if abs(target_gripper - self._current_gripper) > 0.1:
                self._publish_gripper_cmd(target_gripper)
                self._current_gripper = target_gripper

            time.sleep(self.config.step_delay)
            self._spin_once(0.02)
            info["steps_executed"] += 1

        return self.get_observation(), info

    def get_observation(self) -> Observation:
        self._ensure_init()
        for _ in range(5):
            self._spin_once(0.05)

        return Observation(
            image=self.capture_image(),
            joint_positions=self._current_qpos.copy(),
            gripper_open=self._current_gripper,
            timestamp=time.time(),
        )

    def capture_image(self) -> np.ndarray:
        self._ensure_init()
        self._spin_once(0.1)
        if self._current_image is not None:
            return self._current_image.copy()
        h, w = self.config.image_size
        return np.zeros((h, w, 3), dtype=np.uint8)

    def get_joint_positions(self) -> np.ndarray:
        self._ensure_init()
        self._spin_once(0.05)
        return self._current_qpos.copy()

    def get_gripper_state(self) -> float:
        return self._current_gripper

    def get_object_positions(self) -> dict[str, np.ndarray]:
        return {k: v.copy() for k, v in self._objects.items()}

    def destroy(self) -> None:
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
        self._initialized = False
