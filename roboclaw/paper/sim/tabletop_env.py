"""Simulated tabletop pick-and-place environment."""

from __future__ import annotations

from typing import Any

import numpy as np

from roboclaw.paper.config import SimConfig
from roboclaw.paper.sim.base_env import ActionChunk, BaseEnvironment, Observation


class TabletopEnv(BaseEnvironment):
    """Simple tabletop environment with objects and a robot arm.

    Simulates:
    - 6-DOF joint positions (shoulder, elbow, wrist * 3, gripper)
    - Object positions on a bounded workspace
    - Camera image (synthetic: colored rectangles on white background)
    """

    NUM_JOINTS = 6

    def __init__(self, config: SimConfig | None = None):
        self.config = config or SimConfig()
        self._bounds = np.array(self.config.workspace_bounds).reshape(2, 3)
        self._img_h, self._img_w = self.config.image_size
        self._rng = np.random.default_rng(42)

        self._joints = np.zeros(self.NUM_JOINTS, dtype=np.float32)
        self._gripper = 1.0
        self._objects: dict[str, np.ndarray] = {}
        self._step_count = 0
        self._grasped_object: str | None = None

        self.reset()

    def reset(self) -> Observation:
        self._joints = np.zeros(self.NUM_JOINTS, dtype=np.float32)
        self._gripper = 1.0
        self._step_count = 0
        self._grasped_object = None

        self._objects = {}
        for i in range(self.config.num_objects):
            name = f"object_{i}"
            pos = self._rng.uniform(self._bounds[0], self._bounds[1]).astype(np.float32)
            self._objects[name] = pos

        return self.get_observation()

    def step(self, action: ActionChunk) -> tuple[Observation, dict[str, Any]]:
        info: dict[str, Any] = {"steps_executed": 0}

        for t in range(action.chunk_size):
            target_joints = action.joint_targets[t]
            target_gripper = float(action.gripper_targets[t])

            # Simple proportional control toward targets
            self._joints = self._joints + 0.5 * (target_joints - self._joints)
            self._gripper = self._gripper + 0.5 * (target_gripper - self._gripper)

            self._update_grasp()
            self._step_count += 1
            info["steps_executed"] += 1

        return self.get_observation(), info

    def get_observation(self) -> Observation:
        return Observation(
            image=self.capture_image(),
            joint_positions=self._joints.copy(),
            gripper_open=self._gripper,
            timestamp=self._step_count * self.config.dt,
        )

    def capture_image(self) -> np.ndarray:
        """Render a simple synthetic image: white background + colored blocks."""
        img = np.full((self._img_h, self._img_w, 3), 255, dtype=np.uint8)
        colors = [
            (255, 0, 0),
            (0, 180, 0),
            (0, 0, 255),
            (255, 165, 0),
            (128, 0, 128),
        ]
        workspace_range = self._bounds[1] - self._bounds[0]

        for i, (name, pos) in enumerate(self._objects.items()):
            # Map 3D position to 2D image coordinates
            u = int(((pos[0] - self._bounds[0][0]) / workspace_range[0]) * self._img_w)
            v = int(((pos[1] - self._bounds[0][1]) / workspace_range[1]) * self._img_h)
            u = np.clip(u, 10, self._img_w - 10)
            v = np.clip(v, 10, self._img_h - 10)
            color = colors[i % len(colors)]
            img[v - 8 : v + 8, u - 8 : u + 8] = color

        # Draw end-effector as a circle-like cross
        ee_u = int(self._joints[0] / np.pi * self._img_w / 2 + self._img_w / 2)
        ee_v = int(self._joints[1] / np.pi * self._img_h / 2 + self._img_h / 2)
        ee_u = np.clip(ee_u, 5, self._img_w - 5)
        ee_v = np.clip(ee_v, 5, self._img_h - 5)
        img[ee_v - 3 : ee_v + 3, ee_u - 3 : ee_u + 3] = (0, 0, 0)

        return img

    def get_joint_positions(self) -> np.ndarray:
        return self._joints.copy()

    def get_gripper_state(self) -> float:
        return self._gripper

    def get_object_positions(self) -> dict[str, np.ndarray]:
        return {k: v.copy() for k, v in self._objects.items()}

    def _update_grasp(self) -> None:
        """Simple grasp logic: if gripper closes near an object, attach it."""
        ee_pos = self._ee_position()

        if self._gripper < 0.3 and self._grasped_object is None:
            # Try to grasp the nearest object
            for name, pos in self._objects.items():
                dist = float(np.linalg.norm(ee_pos - pos))
                if dist < 0.05:
                    self._grasped_object = name
                    break

        if self._gripper > 0.7 and self._grasped_object is not None:
            self._grasped_object = None

        # Move grasped object with the end effector
        if self._grasped_object is not None:
            self._objects[self._grasped_object] = ee_pos.copy()

    def _ee_position(self) -> np.ndarray:
        """Simplified forward kinematics: map first 3 joints to XYZ."""
        center = (self._bounds[0] + self._bounds[1]) / 2
        return center + self._joints[:3] * 0.05
