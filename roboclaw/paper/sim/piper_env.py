"""Piper real robot environment via piper_sdk.

Agilex Piper: 6-DOF arm + 1-DOF gripper, CAN bus communication.

SDK: pip install piper_sdk
Interface: C_PiperInterface_V2

Units:
- Joint angles: SDK uses 0.001 degrees, we convert to/from radians
- Gripper: SDK uses 0.001mm (0-50000 for 50mm stroke), we normalize to 0-1
- End-effector pose: SDK uses 0.001mm and 0.001 degrees
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from roboclaw.paper.sim.base_env import ActionChunk, BaseEnvironment, Observation


# Conversion factors
_DEG2RAD = math.pi / 180.0
_RAD2DEG = 180.0 / math.pi
_SDK_UNIT = 1000.0  # SDK uses 0.001 degree units
_RAD_TO_SDK = _RAD2DEG * _SDK_UNIT  # radians → SDK units
_SDK_TO_RAD = 1.0 / _RAD_TO_SDK  # SDK units → radians

# Gripper
_GRIPPER_MAX_MM = 50000  # 50mm in 0.001mm units (50mm stroke model)
_GRIPPER_EFFORT_DEFAULT = 1000  # 1 N/m


@dataclass
class PiperConfig:
    """Configuration for Piper robot environment."""

    can_port: str = "can0"
    num_joints: int = 6

    # Home position (radians)
    home_qpos: list[float] = field(default_factory=lambda: [0.0, 0.3, -0.3, 0.0, 0.0, 0.0])

    # Control parameters
    move_speed_rate: int = 50  # 0-100, percentage of max speed
    step_delay: float = 0.05  # seconds between steps

    # Safety
    max_joint_delta: float = 0.2  # max radians per step

    # Gripper
    gripper_max_mm: int = 50000  # max opening in SDK units
    gripper_effort: int = 1000  # N/m * 1000

    # Camera (external, e.g., RealSense)
    camera_device: int | str | None = None  # OpenCV device index or path
    image_size: tuple[int, int] = (480, 640)

    # Joint limits (radians)
    joint_limits_low: list[float] = field(
        default_factory=lambda: [-2.618, 0.0, -2.967, -1.745, -1.22, -2.094]
    )
    joint_limits_high: list[float] = field(
        default_factory=lambda: [2.618, 3.14, 0.0, 1.745, 1.22, 2.094]
    )


class PiperEnv(BaseEnvironment):
    """Real robot environment for Agilex Piper arm.

    Usage:
        # Set up CAN interface first:
        #   sudo ip link set can0 up type can bitrate 1000000
        #
        env = PiperEnv(PiperConfig(can_port="can0"))
        obs = env.reset()
        action = ActionChunk(joint_targets=..., gripper_targets=...)
        obs, info = env.step(action)
    """

    def __init__(self, config: PiperConfig | None = None):
        self.config = config or PiperConfig()
        self._piper = None
        self._camera = None
        self._initialized = False

        # Cached state
        self._current_qpos = np.zeros(self.config.num_joints, dtype=np.float32)
        self._current_gripper = 1.0
        self._current_image: np.ndarray | None = None
        self._objects: dict[str, np.ndarray] = {}

    def _ensure_init(self) -> None:
        """Lazy initialization of Piper SDK and camera."""
        if self._initialized:
            return

        try:
            from piper_sdk import C_PiperInterface_V2
        except ImportError:
            raise RuntimeError("piper_sdk not installed. Run: pip install piper_sdk")

        self._piper = C_PiperInterface_V2(self.config.can_port)
        self._piper.ConnectPort()

        # Enable arm
        retry = 0
        while not self._piper.EnablePiper() and retry < 50:
            time.sleep(0.01)
            retry += 1
        if retry >= 50:
            raise RuntimeError("Failed to enable Piper arm")

        # Set joint control mode
        self._piper.ModeCtrl(
            ctrl_mode=0x01,
            move_mode=0x01,  # MOVE J
            move_spd_rate_ctrl=self.config.move_speed_rate,
            is_mit_mode=0x00,
        )

        # Init camera if configured
        if self.config.camera_device is not None:
            self._init_camera()

        self._initialized = True

    def _init_camera(self) -> None:
        """Initialize external camera via OpenCV."""
        try:
            import cv2
            self._camera = cv2.VideoCapture(self.config.camera_device)
            if not self._camera.isOpened():
                raise RuntimeError(f"Cannot open camera: {self.config.camera_device}")
        except ImportError:
            raise RuntimeError("OpenCV not installed. Run: pip install opencv-python")

    def _read_joints(self) -> np.ndarray:
        """Read current joint positions from Piper, convert to radians."""
        msg = self._piper.GetArmJointMsgs()
        angles = np.zeros(self.config.num_joints, dtype=np.float32)
        for i in range(self.config.num_joints):
            # msg.joint_state.joint_N.angle is in 0.001 degree units
            raw = getattr(msg.joint_state, f"joint_{i + 1}").angle
            angles[i] = raw * _SDK_TO_RAD
        return angles

    def _read_gripper(self) -> float:
        """Read gripper state, normalize to 0-1."""
        msg = self._piper.GetArmGripperMsgs()
        raw_angle = msg.gripper_state.grippers_angle  # 0.001mm units
        return float(np.clip(raw_angle / self.config.gripper_max_mm, 0.0, 1.0))

    def _send_joints(self, qpos: np.ndarray) -> None:
        """Send joint position command to Piper."""
        # Clamp to joint limits
        low = np.array(self.config.joint_limits_low, dtype=np.float32)
        high = np.array(self.config.joint_limits_high, dtype=np.float32)
        qpos = np.clip(qpos, low, high)

        # Convert radians → SDK units (0.001 degrees)
        sdk_angles = [round(float(q) * _RAD_TO_SDK) for q in qpos]
        self._piper.JointCtrl(*sdk_angles)

    def _send_gripper(self, normalized: float) -> None:
        """Send gripper command. normalized: 0=closed, 1=open."""
        angle_sdk = round(float(np.clip(normalized, 0.0, 1.0)) * self.config.gripper_max_mm)
        self._piper.GripperCtrl(
            gripper_angle=angle_sdk,
            gripper_effort=self.config.gripper_effort,
            gripper_code=0x01,  # enable
            set_zero=0,
        )

    # ── BaseEnvironment interface ──

    def reset(self) -> Observation:
        self._ensure_init()

        # Move to home
        home = np.array(self.config.home_qpos, dtype=np.float32)
        self._send_joints(home)
        time.sleep(1.5)  # Wait for motion

        # Open gripper
        self._send_gripper(1.0)
        time.sleep(0.5)

        self._current_qpos = self._read_joints()
        self._current_gripper = self._read_gripper()

        return self.get_observation()

    def step(self, action: ActionChunk) -> tuple[Observation, dict[str, Any]]:
        self._ensure_init()
        info: dict[str, Any] = {"steps_executed": 0}

        for t in range(action.chunk_size):
            target = action.joint_targets[t]
            target_gripper = float(action.gripper_targets[t])

            # Safety clamp
            current = self._read_joints()
            delta = target - current
            delta = np.clip(delta, -self.config.max_joint_delta, self.config.max_joint_delta)
            safe_target = current + delta

            self._send_joints(safe_target)

            if abs(target_gripper - self._current_gripper) > 0.1:
                self._send_gripper(target_gripper)
                self._current_gripper = target_gripper

            time.sleep(self.config.step_delay)
            info["steps_executed"] += 1

        self._current_qpos = self._read_joints()
        self._current_gripper = self._read_gripper()

        return self.get_observation(), info

    def get_observation(self) -> Observation:
        self._ensure_init()
        self._current_qpos = self._read_joints()
        self._current_gripper = self._read_gripper()

        return Observation(
            image=self.capture_image(),
            joint_positions=self._current_qpos.copy(),
            gripper_open=self._current_gripper,
            timestamp=time.time(),
        )

    def capture_image(self) -> np.ndarray:
        if self._camera is not None:
            import cv2
            ret, frame = self._camera.read()
            if ret:
                # BGR → RGB
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        h, w = self.config.image_size
        return np.zeros((h, w, 3), dtype=np.uint8)

    def get_joint_positions(self) -> np.ndarray:
        self._ensure_init()
        return self._read_joints()

    def get_gripper_state(self) -> float:
        self._ensure_init()
        return self._read_gripper()

    def get_object_positions(self) -> dict[str, np.ndarray]:
        return {k: v.copy() for k, v in self._objects.items()}

    # ── Cleanup ──

    def disable(self) -> None:
        """Disable arm motors."""
        if self._piper is not None:
            self._piper.DisablePiper()

    def destroy(self) -> None:
        """Release all resources."""
        self.disable()
        if self._camera is not None:
            self._camera.release()
            self._camera = None
        self._initialized = False
