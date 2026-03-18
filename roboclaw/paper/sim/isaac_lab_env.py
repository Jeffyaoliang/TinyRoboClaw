"""Isaac Lab GPU-parallel simulation environment.

Wraps NVIDIA Isaac Lab (formerly Isaac Gym / Orbit) for massively parallel
data collection on GPU. Supports running hundreds of envs on A100s.

Architecture:
    IsaacLabEnv (BaseEnvironment API)
        └── wraps Isaac Lab ManagerBasedEnv / DirectRLEnv
            └── GPU-parallel articulations + cameras

Dependencies:
    - Isaac Sim (NVIDIA Omniverse)
    - Isaac Lab: pip install isaaclab
    - GPU with CUDA support

Usage:
    # Launch headless (on server with A100s)
    env = IsaacLabEnv(IsaacLabConfig(
        urdf_path="path/to/robot.urdf",
        num_envs=128,
        headless=True,
    ))
    obs = env.reset()
    obs, info = env.step(action)  # Steps ALL 128 envs in parallel

Note:
    This env wraps the GPU-parallel nature into the BaseEnvironment interface.
    Internally all computation is on GPU tensors, but the interface returns
    numpy arrays for compatibility with the rest of the framework.
    For maximum throughput, use the batch API directly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from roboclaw.paper.sim.base_env import ActionChunk, BaseEnvironment, Observation


@dataclass
class IsaacLabConfig:
    """Configuration for Isaac Lab environment."""

    # Robot
    urdf_path: str = ""
    usd_path: str = ""  # Preferred over URDF if available
    num_joints: int = 7
    robot_name: str = "robot"

    # Parallel simulation
    num_envs: int = 64
    device: str = "cuda:0"
    headless: bool = True

    # Simulation
    dt: float = 1.0 / 120.0  # Isaac Lab default: 120 Hz
    control_dt: float = 0.05  # Control frequency: 20 Hz
    decimation: int = 6  # control_dt / dt

    # Camera
    image_size: tuple[int, int] = (224, 224)
    enable_camera: bool = True
    camera_position: tuple[float, float, float] = (0.5, 0.0, 1.0)
    camera_target: tuple[float, float, float] = (0.0, 0.0, 0.5)

    # Scene
    table_height: float = 0.4
    num_objects: int = 2
    object_size: float = 0.03

    # Home position
    home_qpos: list[float] = field(default_factory=lambda: [0.0] * 7)

    # Safety
    max_joint_delta: float = 0.3

    # Which parallel env to use for BaseEnvironment interface (single-env view)
    primary_env_idx: int = 0


class IsaacLabEnv(BaseEnvironment):
    """GPU-parallel environment via NVIDIA Isaac Lab.

    Provides two interfaces:
    1. BaseEnvironment (single-env): Uses primary_env_idx for compatibility
    2. Batch API: step_batch() / reset_batch() for parallel data collection

    The batch API is ~100x faster for data collection as it processes
    all envs on GPU simultaneously.
    """

    def __init__(self, config: IsaacLabConfig | None = None):
        self.config = config or IsaacLabConfig()
        self._sim_app = None
        self._env = None
        self._robot = None
        self._camera = None
        self._initialized = False
        self._step_count = 0

        # Cached state for primary env
        self._current_qpos: np.ndarray | None = None
        self._current_gripper = 1.0
        self._current_image: np.ndarray | None = None

    def _ensure_init(self) -> None:
        """Initialize Isaac Lab simulation."""
        if self._initialized:
            return

        try:
            # Isaac Lab requires AppLauncher before any other imports
            from omni.isaac.lab.app import AppLauncher

            launcher = AppLauncher(headless=self.config.headless)
            self._sim_app = launcher.app

            self._setup_scene()
            self._initialized = True
        except ImportError as e:
            raise RuntimeError(
                f"Isaac Lab not available: {e}\n"
                "Install Isaac Sim + Isaac Lab first:\n"
                "  https://isaac-sim.github.io/IsaacLab/source/setup/installation.html"
            )

    def _setup_scene(self) -> None:
        """Create the simulation scene with robot, table, objects, camera."""
        import torch
        import omni.isaac.lab.sim as sim_utils
        from omni.isaac.lab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
        from omni.isaac.lab.sensors import Camera, CameraCfg
        from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
        from omni.isaac.lab.sim import SimulationCfg, SimulationContext

        # Simulation context
        sim_cfg = SimulationCfg(dt=self.config.dt, device=self.config.device)
        sim = SimulationContext(sim_cfg)

        # Ground plane
        sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())

        # Light
        sim_utils.DistantLightCfg(intensity=3000.0).func("/World/light", sim_utils.DistantLightCfg(intensity=3000.0))

        # Table
        table_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/table",
            spawn=sim_utils.CuboidCfg(
                size=(0.8, 0.8, self.config.table_height),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.6, 0.5)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.3, 0.0, self.config.table_height / 2),
            ),
        )

        # Robot
        robot_spawn_cfg = self._get_robot_spawn_cfg()
        robot_cfg = ArticulationCfg(
            prim_path="/World/envs/env_.*/robot",
            spawn=robot_spawn_cfg,
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos={f"joint{i+1}": self.config.home_qpos[i] for i in range(self.config.num_joints)},
            ),
            actuators={
                "arm": sim_utils.ImplicitActuatorCfg(
                    joint_names_expr=["joint[1-7]"],
                    stiffness=50.0,
                    damping=5.0,
                ),
            },
        )

        # Objects
        objects_cfg = {}
        colors = [(0.9, 0.1, 0.1), (0.1, 0.7, 0.1), (0.1, 0.1, 0.9)]
        for i in range(self.config.num_objects):
            obj_cfg = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/obj_{i}",
                spawn=sim_utils.CuboidCfg(
                    size=(self.config.object_size * 2,) * 3,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=colors[i % len(colors)]
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.3 + i * 0.1, 0.05 * ((-1) ** i), self.config.table_height + self.config.object_size + 0.01),
                ),
            )
            objects_cfg[f"obj_{i}"] = obj_cfg

        # Camera
        camera_cfg = None
        if self.config.enable_camera:
            camera_cfg = CameraCfg(
                prim_path="/World/envs/env_.*/camera",
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0,
                    focus_distance=400.0,
                    horizontal_aperture=20.955,
                ),
                height=self.config.image_size[0],
                width=self.config.image_size[1],
                data_types=["rgb"],
                update_period=self.config.control_dt,
            )

        # Build scene config
        scene_cfg = InteractiveSceneCfg(num_envs=self.config.num_envs, env_spacing=2.0)

        # Create scene
        self._scene = InteractiveScene(scene_cfg)
        self._robot = Articulation(robot_cfg)
        self._sim = sim

        if camera_cfg:
            self._camera = Camera(camera_cfg)

        # Reset simulation
        sim.reset()

    def _get_robot_spawn_cfg(self) -> Any:
        """Get spawn config based on URDF or USD path."""
        import omni.isaac.lab.sim as sim_utils

        if self.config.usd_path:
            return sim_utils.UsdFileCfg(usd_path=self.config.usd_path)

        if self.config.urdf_path:
            urdf_path = Path(self.config.urdf_path).expanduser()
            return sim_utils.UrdfFileCfg(
                asset_path=str(urdf_path),
                fix_base=True,
            )

        raise ValueError("Must provide either urdf_path or usd_path")

    # ── Batch API (GPU-parallel, high throughput) ──

    def reset_batch(self, env_ids: list[int] | None = None) -> dict[str, Any]:
        """Reset specified envs (or all). Returns batched observations as torch tensors."""
        import torch

        self._ensure_init()

        if env_ids is None:
            env_ids = list(range(self.config.num_envs))

        idx = torch.tensor(env_ids, device=self.config.device)

        # Reset joint positions to home
        home = torch.tensor(self.config.home_qpos, device=self.config.device, dtype=torch.float32)
        home_batch = home.unsqueeze(0).expand(len(env_ids), -1)
        self._robot.write_joint_state_to_sim(home_batch, torch.zeros_like(home_batch), env_ids=idx)

        # Step simulation to settle
        for _ in range(10):
            self._sim.step()

        self._step_count = 0
        return self._get_batch_obs(env_ids)

    def step_batch(
        self,
        actions: Any,  # torch.Tensor (num_envs, num_joints)
        gripper_actions: Any | None = None,  # torch.Tensor (num_envs,)
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Step all envs in parallel. Returns (observations, info).

        Args:
            actions: Joint position targets, shape (num_envs, num_joints)
            gripper_actions: Gripper targets, shape (num_envs,)

        Returns:
            (obs_dict, info_dict) with torch tensors on GPU
        """
        self._ensure_init()

        # Apply actions
        self._robot.set_joint_position_target(actions)

        # Step simulation
        for _ in range(self.config.decimation):
            self._sim.step()

        self._step_count += 1

        obs = self._get_batch_obs()
        info = {"steps": self._step_count}
        return obs, info

    def _get_batch_obs(self, env_ids: list[int] | None = None) -> dict[str, Any]:
        """Get batched observations as torch tensors."""
        import torch

        obs = {
            "joint_positions": self._robot.data.joint_pos,  # (num_envs, num_joints)
            "joint_velocities": self._robot.data.joint_vel,
        }

        if self._camera is not None:
            self._camera.update(self.config.control_dt)
            obs["images"] = self._camera.data.output["rgb"]  # (num_envs, H, W, 3)

        if env_ids is not None:
            idx = torch.tensor(env_ids, device=self.config.device)
            obs = {k: v[idx] for k, v in obs.items()}

        return obs

    def get_batch_images(self) -> Any:
        """Get camera images from all envs. Returns (num_envs, H, W, 3) torch tensor."""
        if self._camera is not None:
            self._camera.update(self.config.control_dt)
            return self._camera.data.output["rgb"]
        return None

    # ── BaseEnvironment interface (single-env view) ──

    def reset(self) -> Observation:
        self._ensure_init()
        obs_dict = self.reset_batch([self.config.primary_env_idx])
        return self._dict_to_observation(obs_dict, 0)

    def step(self, action: ActionChunk) -> tuple[Observation, dict[str, Any]]:
        import torch

        self._ensure_init()
        info: dict[str, Any] = {"steps_executed": 0}

        for t in range(action.chunk_size):
            # Broadcast single action to all envs (only primary_env_idx matters)
            target = torch.tensor(
                action.joint_targets[t],
                device=self.config.device,
                dtype=torch.float32,
            ).unsqueeze(0).expand(self.config.num_envs, -1)

            obs_dict, _ = self.step_batch(target)
            info["steps_executed"] += 1

        return self._dict_to_observation(obs_dict, self.config.primary_env_idx), info

    def get_observation(self) -> Observation:
        self._ensure_init()
        obs_dict = self._get_batch_obs([self.config.primary_env_idx])
        return self._dict_to_observation(obs_dict, 0)

    def capture_image(self) -> np.ndarray:
        self._ensure_init()
        if self._camera is not None:
            self._camera.update(self.config.control_dt)
            img = self._camera.data.output["rgb"][self.config.primary_env_idx]
            return img.cpu().numpy().astype(np.uint8)
        h, w = self.config.image_size
        return np.zeros((h, w, 3), dtype=np.uint8)

    def get_joint_positions(self) -> np.ndarray:
        self._ensure_init()
        return self._robot.data.joint_pos[self.config.primary_env_idx].cpu().numpy().astype(np.float32)

    def get_gripper_state(self) -> float:
        return self._current_gripper

    def get_object_positions(self) -> dict[str, np.ndarray]:
        return {}  # TODO: track objects via scene

    def _dict_to_observation(self, obs_dict: dict, idx: int) -> Observation:
        """Convert batch obs dict to single Observation."""
        joints = obs_dict["joint_positions"][idx].cpu().numpy().astype(np.float32)

        if "images" in obs_dict:
            image = obs_dict["images"][idx].cpu().numpy().astype(np.uint8)
        else:
            h, w = self.config.image_size
            image = np.zeros((h, w, 3), dtype=np.uint8)

        return Observation(
            image=image,
            joint_positions=joints,
            gripper_open=self._current_gripper,
            timestamp=time.time(),
        )

    def close(self) -> None:
        """Shutdown simulation."""
        if self._sim_app is not None:
            self._sim_app.close()
            self._sim_app = None
        self._initialized = False

    # ── Parallel data collection helper ──

    @property
    def num_envs(self) -> int:
        return self.config.num_envs
