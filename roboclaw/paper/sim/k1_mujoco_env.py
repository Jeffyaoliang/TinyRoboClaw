"""K1 MuJoCo simulation environment.

Loads the K1 URDF from the k1-ros2 repository and runs physics in MuJoCo.
Supports single-arm or dual-arm control.

URDF location: k1-ros2/knowin_controller/urdfs/k1u_v2_26w07_2r/urdf/robot.urdf
Meshes:        k1-ros2/knowin_controller/urdfs/k1u_v2_26w07_2r/meshes/*.STL

Dependencies:
    pip install mujoco
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from roboclaw.paper.sim.base_env import ActionChunk, BaseEnvironment, Observation


# K1 joint structure:
#   lifting_joint (prismatic, Z)
#   l_joint1..l_joint7 (left arm, revolute)
#   r_joint1..r_joint7 (right arm, revolute)

_LEFT_JOINTS = [f"l_joint{i}" for i in range(1, 8)]
_RIGHT_JOINTS = [f"r_joint{i}" for i in range(1, 8)]


@dataclass
class K1MujocoConfig:
    """Configuration for K1 MuJoCo simulation."""

    # Path to K1 URDF
    urdf_path: str = "~/Desktop/k1-ros 2/knowin_controller/urdfs/k1u_v2_26w07_2r/urdf/robot.urdf"

    # Which arm(s) to control: "left", "right", or "both"
    arm: str = "left"
    num_joints: int = 7  # per arm

    # Simulation
    dt: float = 0.002
    control_dt: float = 0.05
    image_size: tuple[int, int] = (224, 224)

    # Home position (radians, 7-DOF per arm)
    home_qpos_left: list[float] = field(
        default_factory=lambda: [0.0, 0.5, 0.0, -1.0, 0.0, 0.5, 0.0]
    )
    home_qpos_right: list[float] = field(
        default_factory=lambda: [0.0, -0.5, 0.0, 1.0, 0.0, -0.5, 0.0]
    )
    lifting_height: float = -0.2  # prismatic joint value

    # Rendering
    render: bool = False
    camera_distance: float = 1.5
    camera_elevation: float = -30.0
    camera_azimuth: float = 135.0

    # Objects
    num_objects: int = 2
    object_size: float = 0.03


class K1MujocoEnv(BaseEnvironment):
    """MuJoCo simulation of the K1 dual-arm humanoid upper body.

    Loads the real K1 URDF with STL meshes for accurate physics.
    """

    def __init__(self, config: K1MujocoConfig | None = None):
        self.config = config or K1MujocoConfig()
        self._model = None
        self._data = None
        self._renderer = None
        self._rng = np.random.default_rng(42)
        self._step_count = 0

        # Joint/actuator mappings
        self._joint_ids: list[int] = []
        self._actuator_ids: list[int] = []
        self._joint_names: list[str] = []
        self._object_body_ids: list[int] = []

        self._init_mujoco()

    def _init_mujoco(self) -> None:
        """Load K1 URDF into MuJoCo."""
        try:
            import mujoco
        except ImportError:
            raise RuntimeError("MuJoCo not installed. Run: pip install mujoco")

        urdf_path = Path(self.config.urdf_path).expanduser()
        if not urdf_path.exists():
            raise FileNotFoundError(
                f"K1 URDF not found: {urdf_path}\n"
                f"Expected at: ~/Desktop/k1-ros 2/knowin_controller/urdfs/k1u_v2_26w07_2r/urdf/robot.urdf"
            )

        # MuJoCo can load URDF directly, but we need to wrap it in MJCF
        # to add actuators, cameras, ground plane, and objects
        mjcf_xml = self._build_mjcf(urdf_path)

        # Write temp MJCF file (MuJoCo needs filesystem access for mesh paths)
        self._temp_dir = tempfile.mkdtemp(prefix="k1_mujoco_")
        mjcf_path = Path(self._temp_dir) / "k1_scene.xml"
        mjcf_path.write_text(mjcf_xml)

        self._model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        self._data = mujoco.MjData(self._model)
        self._model.opt.timestep = self.config.dt

        # Resolve joint/actuator IDs
        self._resolve_ids(mujoco)

    def _build_mjcf(self, urdf_path: Path) -> str:
        """Build MJCF XML that includes the K1 URDF + scene elements."""
        meshdir = str(urdf_path.parent.parent / "meshes")

        # Determine which joints to actuate
        if self.config.arm == "left":
            self._joint_names = _LEFT_JOINTS
        elif self.config.arm == "right":
            self._joint_names = _RIGHT_JOINTS
        else:  # both
            self._joint_names = _LEFT_JOINTS + _RIGHT_JOINTS

        # Build actuator block
        actuators = []
        for jname in self._joint_names:
            actuators.append(f'    <position name="act_{jname}" joint="{jname}" kp="50"/>')
        actuator_xml = "\n".join(actuators)

        # Build objects
        objects = []
        for i in range(self.config.num_objects):
            x = 0.3 + i * 0.1
            y = 0.05 * ((-1) ** i)
            colors = ["0.9 0.1 0.1 1", "0.1 0.7 0.1 1", "0.1 0.1 0.9 1"]
            color = colors[i % len(colors)]
            s = self.config.object_size
            objects.append(
                f'    <body name="obj_{i}" pos="{x:.3f} {y:.3f} {0.82 + s:.3f}">\n'
                f'      <freejoint name="obj_{i}_free"/>\n'
                f'      <geom type="box" size="{s} {s} {s}" rgba="{color}" mass="0.05"/>\n'
                f'    </body>'
            )
        objects_xml = "\n".join(objects)

        return f"""<mujoco model="k1_scene">
  <compiler meshdir="{meshdir}" balanceinertia="true" strippath="true"/>

  <option timestep="{self.config.dt}" gravity="0 0 -9.81"/>

  <default>
    <joint damping="1.0" armature="0.1"/>
    <geom condim="3" friction="1 0.5 0.01"/>
  </default>

  <worldbody>
    <!-- Ground plane -->
    <geom type="plane" size="2 2 0.01" rgba="0.9 0.9 0.9 1"/>

    <!-- Table -->
    <body name="table" pos="0.3 0 0.4">
      <geom type="box" size="0.4 0.4 0.02" rgba="0.7 0.6 0.5 1"/>
    </body>

    <!-- K1 Robot (from URDF) -->
    <include file="{urdf_path}"/>

    <!-- Manipulable objects -->
{objects_xml}

    <!-- Lighting -->
    <light pos="0.3 0 2.0" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <light pos="-0.5 0.5 1.5" dir="0.5 -0.5 -1" diffuse="0.4 0.4 0.4"/>
  </worldbody>

  <actuator>
    <position name="act_lifting" joint="lifting_joint" kp="200"/>
{actuator_xml}
  </actuator>
</mujoco>
"""

    def _resolve_ids(self, mujoco_module: Any) -> None:
        """Resolve MuJoCo IDs for joints and actuators."""
        mujoco = mujoco_module

        for jname in self._joint_names:
            jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            aid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_{jname}")
            if jid < 0:
                raise RuntimeError(f"Joint '{jname}' not found in model")
            self._joint_ids.append(jid)
            self._actuator_ids.append(aid)

        for i in range(self.config.num_objects):
            bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, f"obj_{i}")
            if bid >= 0:
                self._object_body_ids.append(bid)

    # ── BaseEnvironment interface ──

    def reset(self) -> Observation:
        import mujoco

        mujoco.mj_resetData(self._model, self._data)

        # Set lifting height
        lift_jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "lifting_joint")
        if lift_jid >= 0:
            self._data.qpos[self._model.jnt_qposadr[lift_jid]] = self.config.lifting_height
            lift_aid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_lifting")
            if lift_aid >= 0:
                self._data.ctrl[lift_aid] = self.config.lifting_height

        # Set arm home positions
        home = self._get_home_qpos()
        for i, jid in enumerate(self._joint_ids):
            qaddr = self._model.jnt_qposadr[jid]
            self._data.qpos[qaddr] = home[i]
            if self._actuator_ids[i] >= 0:
                self._data.ctrl[self._actuator_ids[i]] = home[i]

        # Randomize objects
        for i, bid in enumerate(self._object_body_ids):
            jnt_id = self._model.body_jntadr[bid]
            if jnt_id >= 0:
                qaddr = self._model.jnt_qposadr[jnt_id]
                self._data.qpos[qaddr] = 0.3 + self._rng.uniform(-0.05, 0.15)
                self._data.qpos[qaddr + 1] = self._rng.uniform(-0.1, 0.1)
                self._data.qpos[qaddr + 2] = 0.82 + self.config.object_size

        # Settle
        mujoco.mj_forward(self._model, self._data)
        for _ in range(500):
            mujoco.mj_step(self._model, self._data)

        self._step_count = 0
        return self.get_observation()

    def _get_home_qpos(self) -> np.ndarray:
        """Get home qpos for the controlled joints."""
        if self.config.arm == "left":
            return np.array(self.config.home_qpos_left, dtype=np.float64)
        elif self.config.arm == "right":
            return np.array(self.config.home_qpos_right, dtype=np.float64)
        else:
            return np.concatenate([
                self.config.home_qpos_left,
                self.config.home_qpos_right,
            ]).astype(np.float64)

    def step(self, action: ActionChunk) -> tuple[Observation, dict[str, Any]]:
        import mujoco

        info: dict[str, Any] = {"steps_executed": 0}
        n_substeps = max(1, round(self.config.control_dt / self.config.dt))

        for t in range(action.chunk_size):
            target = action.joint_targets[t].astype(np.float64)
            target_gripper = float(action.gripper_targets[t])

            # Set actuator controls for arm joints
            n_arm = len(self._actuator_ids)
            for i in range(min(n_arm, len(target))):
                if self._actuator_ids[i] >= 0:
                    self._data.ctrl[self._actuator_ids[i]] = target[i]

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
            gripper_open=0.5,  # No gripper in URDF, placeholder
            timestamp=self._step_count * self.config.control_dt,
        )

    def capture_image(self) -> np.ndarray:
        import mujoco

        if self._renderer is None:
            h, w = self.config.image_size
            self._renderer = mujoco.Renderer(self._model, h, w)

        self._renderer.update_scene(self._data)
        return self._renderer.render().copy()

    def get_joint_positions(self) -> np.ndarray:
        positions = np.zeros(len(self._joint_ids), dtype=np.float32)
        for i, jid in enumerate(self._joint_ids):
            positions[i] = self._data.qpos[self._model.jnt_qposadr[jid]]
        return positions

    def get_gripper_state(self) -> float:
        return 0.5  # K1 URDF 不含夹爪关节，后续可扩展

    def get_object_positions(self) -> dict[str, np.ndarray]:
        positions = {}
        for i, bid in enumerate(self._object_body_ids):
            positions[f"obj_{i}"] = self._data.xpos[bid].copy().astype(np.float32)
        return positions

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

        # Clean up temp directory
        import shutil
        if hasattr(self, "_temp_dir") and Path(self._temp_dir).exists():
            shutil.rmtree(self._temp_dir, ignore_errors=True)
