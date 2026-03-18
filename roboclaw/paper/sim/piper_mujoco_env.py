"""Piper MuJoCo simulation environment.

Uses MuJoCo physics engine to simulate the Agilex Piper 6-DOF arm.
Can use the URDF/MJCF model from piper_ros or a built-in simplified model.

Dependencies:
    pip install mujoco
    # Optional for rendering: pip install mujoco-python-viewer
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from roboclaw.paper.sim.base_env import ActionChunk, BaseEnvironment, Observation


@dataclass
class PiperMujocoConfig:
    """Configuration for Piper MuJoCo simulation."""

    # Path to MuJoCo XML model.
    # If None, uses a built-in minimal model.
    model_path: str | None = None

    num_joints: int = 6
    image_size: tuple[int, int] = (224, 224)
    dt: float = 0.002  # MuJoCo timestep
    control_dt: float = 0.05  # Control loop period (= dt * n_substeps)

    # Home position (radians)
    home_qpos: list[float] = field(default_factory=lambda: [0.0, 0.3, -0.3, 0.0, 0.0, 0.0])

    # Objects to spawn
    num_objects: int = 2
    object_size: float = 0.025  # meters

    # Rendering
    render: bool = False
    camera_name: str = "overhead"  # camera in MJCF

    # Safety
    max_joint_delta: float = 0.3


_MINIMAL_MJCF = """
<mujoco model="piper_minimal">
  <option timestep="0.002" gravity="0 0 -9.81"/>

  <default>
    <joint damping="0.5" armature="0.1"/>
    <geom condim="3" friction="1 0.5 0.01"/>
  </default>

  <worldbody>
    <!-- Table -->
    <body name="table" pos="0 0 0.4">
      <geom type="box" size="0.4 0.4 0.02" rgba="0.8 0.7 0.6 1"/>
    </body>

    <!-- Piper Arm (simplified 6-DOF) -->
    <body name="base" pos="0 0 0.42">
      <geom type="cylinder" size="0.04 0.02" rgba="0.3 0.3 0.3 1"/>

      <body name="link1" pos="0 0 0.02">
        <joint name="joint1" type="hinge" axis="0 0 1" range="-2.618 2.618"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.12" size="0.03" rgba="0.2 0.2 0.8 1"/>

        <body name="link2" pos="0 0 0.12">
          <joint name="joint2" type="hinge" axis="0 1 0" range="0 3.14"/>
          <geom type="capsule" fromto="0 0 0 0.15 0 0" size="0.025" rgba="0.2 0.6 0.2 1"/>

          <body name="link3" pos="0.15 0 0">
            <joint name="joint3" type="hinge" axis="0 1 0" range="-2.967 0"/>
            <geom type="capsule" fromto="0 0 0 0.12 0 0" size="0.02" rgba="0.8 0.2 0.2 1"/>

            <body name="link4" pos="0.12 0 0">
              <joint name="joint4" type="hinge" axis="1 0 0" range="-1.745 1.745"/>
              <geom type="capsule" fromto="0 0 0 0.08 0 0" size="0.018" rgba="0.6 0.6 0.2 1"/>

              <body name="link5" pos="0.08 0 0">
                <joint name="joint5" type="hinge" axis="0 1 0" range="-1.22 1.22"/>
                <geom type="capsule" fromto="0 0 0 0.06 0 0" size="0.015" rgba="0.2 0.6 0.6 1"/>

                <body name="link6" pos="0.06 0 0">
                  <joint name="joint6" type="hinge" axis="1 0 0" range="-2.094 2.094"/>
                  <geom type="capsule" fromto="0 0 0 0.04 0 0" size="0.012" rgba="0.6 0.2 0.6 1"/>

                  <!-- Gripper (simplified as two fingers) -->
                  <body name="gripper" pos="0.04 0 0">
                    <joint name="gripper_joint" type="slide" axis="0 1 0" range="0 0.025"/>
                    <geom name="finger_left" type="box" pos="0.01 0.012 0" size="0.01 0.002 0.01" rgba="0.4 0.4 0.4 1"/>
                    <geom name="finger_right" type="box" pos="0.01 -0.012 0" size="0.01 0.002 0.01" rgba="0.4 0.4 0.4 1"/>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>

    <!-- Objects (will be added dynamically) -->
    {objects}

    <!-- Cameras -->
    <camera name="overhead" pos="0.2 0 1.0" xyaxes="0 1 0 -1 0 0.5" fovy="60"/>
    <camera name="front" pos="0.8 0 0.7" xyaxes="0 1 0 -0.5 0 0.8" fovy="60"/>

    <!-- Lighting -->
    <light pos="0.2 0 1.5" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
  </worldbody>

  <actuator>
    <position name="act_joint1" joint="joint1" kp="50"/>
    <position name="act_joint2" joint="joint2" kp="50"/>
    <position name="act_joint3" joint="joint3" kp="30"/>
    <position name="act_joint4" joint="joint4" kp="20"/>
    <position name="act_joint5" joint="joint5" kp="15"/>
    <position name="act_joint6" joint="joint6" kp="10"/>
    <position name="act_gripper" joint="gripper_joint" kp="20"/>
  </actuator>
</mujoco>
"""


class PiperMujocoEnv(BaseEnvironment):
    """MuJoCo-simulated Piper arm environment.

    Usage:
        env = PiperMujocoEnv(PiperMujocoConfig(render=True))
        obs = env.reset()
        for _ in range(100):
            action = ActionChunk(...)
            obs, info = env.step(action)
    """

    def __init__(self, config: PiperMujocoConfig | None = None):
        self.config = config or PiperMujocoConfig()
        self._model = None
        self._data = None
        self._renderer = None
        self._rng = np.random.default_rng(42)
        self._step_count = 0
        self._joint_ids: list[int] = []
        self._actuator_ids: list[int] = []
        self._gripper_actuator_id: int = -1
        self._object_body_ids: list[int] = []

        self._init_mujoco()

    def _init_mujoco(self) -> None:
        """Initialize MuJoCo model and data."""
        try:
            import mujoco
        except ImportError:
            raise RuntimeError("MuJoCo not installed. Run: pip install mujoco")

        if self.config.model_path and Path(self.config.model_path).exists():
            self._model = mujoco.MjModel.from_xml_path(self.config.model_path)
        else:
            # Build minimal model with objects
            objects_xml = self._generate_objects_xml()
            xml = _MINIMAL_MJCF.replace("{objects}", objects_xml)
            self._model = mujoco.MjModel.from_xml_string(xml)

        self._data = mujoco.MjData(self._model)
        self._model.opt.timestep = self.config.dt

        # Cache joint and actuator IDs
        for i in range(1, self.config.num_joints + 1):
            jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")
            self._joint_ids.append(jid)
            aid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_joint{i}")
            self._actuator_ids.append(aid)

        self._gripper_actuator_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_gripper"
        )

        # Cache object body IDs
        for i in range(self.config.num_objects):
            bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, f"obj_{i}")
            if bid >= 0:
                self._object_body_ids.append(bid)

        # Renderer
        if self.config.render:
            self._renderer = mujoco.Renderer(self._model, *self.config.image_size)

    def _generate_objects_xml(self) -> str:
        """Generate XML for spawnable objects."""
        xml_parts = []
        colors = [
            "0.9 0.1 0.1 1",
            "0.1 0.7 0.1 1",
            "0.1 0.1 0.9 1",
            "0.9 0.6 0.1 1",
            "0.7 0.1 0.7 1",
        ]
        size = self.config.object_size

        for i in range(self.config.num_objects):
            x = 0.1 + i * 0.12
            y = 0.05 * ((-1) ** i)
            color = colors[i % len(colors)]
            xml_parts.append(
                f'<body name="obj_{i}" pos="{x:.3f} {y:.3f} {0.42 + size:.3f}">'
                f'  <freejoint name="obj_{i}_free"/>'
                f'  <geom type="box" size="{size} {size} {size}" rgba="{color}" mass="0.05"/>'
                f'</body>'
            )
        return "\n    ".join(xml_parts)

    # ── BaseEnvironment interface ──

    def reset(self) -> Observation:
        import mujoco

        mujoco.mj_resetData(self._model, self._data)

        # Set home position
        home = np.array(self.config.home_qpos, dtype=np.float64)
        for i, jid in enumerate(self._joint_ids):
            qpos_addr = self._model.jnt_qposadr[jid]
            self._data.qpos[qpos_addr] = home[i]
            self._data.ctrl[self._actuator_ids[i]] = home[i]

        # Open gripper
        self._data.ctrl[self._gripper_actuator_id] = 0.025

        # Randomize object positions
        for i, bid in enumerate(self._object_body_ids):
            jnt_id = self._model.body_jntadr[bid]
            if jnt_id >= 0:
                qpos_addr = self._model.jnt_qposadr[jnt_id]
                self._data.qpos[qpos_addr] = 0.1 + self._rng.uniform(-0.05, 0.15)
                self._data.qpos[qpos_addr + 1] = self._rng.uniform(-0.1, 0.1)
                self._data.qpos[qpos_addr + 2] = 0.42 + self.config.object_size

        # Forward to settle
        mujoco.mj_forward(self._model, self._data)
        for _ in range(200):
            mujoco.mj_step(self._model, self._data)

        self._step_count = 0
        return self.get_observation()

    def step(self, action: ActionChunk) -> tuple[Observation, dict[str, Any]]:
        import mujoco

        info: dict[str, Any] = {"steps_executed": 0}
        n_substeps = max(1, round(self.config.control_dt / self.config.dt))

        for t in range(action.chunk_size):
            target_joints = action.joint_targets[t].astype(np.float64)
            target_gripper = float(action.gripper_targets[t])

            # Set actuator controls
            for i, aid in enumerate(self._actuator_ids):
                self._data.ctrl[aid] = target_joints[i]
            self._data.ctrl[self._gripper_actuator_id] = target_gripper * 0.025

            # Step simulation
            for _ in range(n_substeps):
                mujoco.mj_step(self._model, self._data)

            self._step_count += 1
            info["steps_executed"] += 1

        return self.get_observation(), info

    def get_observation(self) -> Observation:
        return Observation(
            image=self.capture_image(),
            joint_positions=self.get_joint_positions(),
            gripper_open=self.get_gripper_state(),
            timestamp=self._step_count * self.config.control_dt,
        )

    def capture_image(self) -> np.ndarray:
        import mujoco

        if self._renderer is None:
            self._renderer = mujoco.Renderer(self._model, *self.config.image_size)

        self._renderer.update_scene(self._data, camera=self.config.camera_name)
        img = self._renderer.render()
        return img.copy()

    def get_joint_positions(self) -> np.ndarray:
        positions = np.zeros(self.config.num_joints, dtype=np.float32)
        for i, jid in enumerate(self._joint_ids):
            qpos_addr = self._model.jnt_qposadr[jid]
            positions[i] = self._data.qpos[qpos_addr]
        return positions

    def get_gripper_state(self) -> float:
        """Gripper: 0=closed, 1=open. Joint range 0-0.025m."""
        gid = self._model.jnt_qposadr[
            self._model.actuator_trnid[self._gripper_actuator_id, 0]
        ] if self._gripper_actuator_id >= 0 else -1

        if gid >= 0:
            return float(np.clip(self._data.qpos[gid] / 0.025, 0.0, 1.0))
        return 0.5

    def get_object_positions(self) -> dict[str, np.ndarray]:
        import mujoco

        positions = {}
        for i, bid in enumerate(self._object_body_ids):
            pos = self._data.xpos[bid].copy().astype(np.float32)
            positions[f"obj_{i}"] = pos
        return positions

    def close(self) -> None:
        """Clean up renderer."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
