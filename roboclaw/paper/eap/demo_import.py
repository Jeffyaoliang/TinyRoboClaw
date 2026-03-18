"""Import human demonstrations into TrajectoryDataset.

Paper workflow: human demos → seed dataset → EAP online collection → flywheel.
Supports multiple formats: recorded joint trajectories, ROS2 bags, HDF5, npz.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from roboclaw.paper.eap.trajectory import TimeStep, Trajectory, TrajectoryDataset


class DemoImporter:
    """Import human demonstrations as seed data for the training dataset."""

    def __init__(self, dataset: TrajectoryDataset):
        self.dataset = dataset

    def import_from_npz(
        self,
        npz_path: str | Path,
        task: str,
        direction: str = "forward",
        image_key: str = "images",
        joints_key: str = "joint_positions",
        actions_key: str = "actions",
        gripper_key: str = "gripper",
    ) -> int:
        """Import demo from a NumPy npz file.

        Expected npz contents:
        - images: (T, H, W, 3) or None
        - joint_positions: (T, num_joints)
        - actions: (T, num_joints) optional
        - gripper: (T,) optional

        Returns trajectory index.
        """
        data = np.load(npz_path, allow_pickle=True)

        joints = data[joints_key]
        T = joints.shape[0]

        images = data.get(image_key)
        if images is None:
            images = np.zeros((T, 8, 8, 3), dtype=np.uint8)

        actions = data.get(actions_key)
        grippers = data.get(gripper_key)
        if grippers is None:
            grippers = np.full(T, 0.5, dtype=np.float32)

        traj = Trajectory(
            task=task,
            direction=direction,
            success=True,
            metadata={"source": "human_demo", "file": str(npz_path)},
        )

        for t in range(T):
            ts = TimeStep(
                image=images[t].astype(np.uint8),
                joint_positions=joints[t].astype(np.float32),
                gripper_open=float(grippers[t]),
                action_joints=actions[t].astype(np.float32) if actions is not None else None,
                action_gripper=float(grippers[min(t + 1, T - 1)]) if actions is not None else None,
                timestamp=t * 0.05,
            )
            traj.add_step(ts)

        idx = self.dataset.add(traj)
        logger.info(f"Imported demo from {npz_path}: {T} steps → index {idx}")
        return idx

    def import_from_hdf5(
        self,
        hdf5_path: str | Path,
        task: str,
        direction: str = "forward",
        obs_group: str = "observations",
        action_group: str = "actions",
    ) -> int:
        """Import demo from HDF5 file (LeRobot/robomimic format)."""
        try:
            import h5py
        except ImportError:
            raise RuntimeError("h5py required for HDF5 import. Run: pip install h5py")

        with h5py.File(hdf5_path, "r") as f:
            obs = f[obs_group]
            joints = np.array(obs.get("joint_positions", obs.get("qpos", [])))
            images = np.array(obs.get("images", obs.get("image", [])))
            grippers = np.array(obs.get("gripper", np.full(len(joints), 0.5)))

            acts = f.get(action_group)
            actions = np.array(acts) if acts is not None else None

        T = len(joints)
        if images.ndim < 4 or len(images) == 0:
            images = np.zeros((T, 8, 8, 3), dtype=np.uint8)

        traj = Trajectory(
            task=task,
            direction=direction,
            success=True,
            metadata={"source": "human_demo_hdf5", "file": str(hdf5_path)},
        )

        for t in range(T):
            ts = TimeStep(
                image=images[t].astype(np.uint8),
                joint_positions=joints[t].astype(np.float32),
                gripper_open=float(grippers[t]),
                action_joints=actions[t].astype(np.float32) if actions is not None else None,
                timestamp=t * 0.05,
            )
            traj.add_step(ts)

        idx = self.dataset.add(traj)
        logger.info(f"Imported HDF5 demo from {hdf5_path}: {T} steps → index {idx}")
        return idx

    def import_from_joint_trajectory(
        self,
        joints_array: np.ndarray,
        task: str,
        direction: str = "forward",
        gripper_array: np.ndarray | None = None,
        images: np.ndarray | None = None,
        dt: float = 0.05,
    ) -> int:
        """Import from raw numpy arrays (e.g., from K1's qpos_recorder).

        Args:
            joints_array: (T, num_joints) joint positions.
            task: Task name.
            gripper_array: (T,) gripper values, optional.
            images: (T, H, W, 3) images, optional.
            dt: Timestep interval.
        """
        T = joints_array.shape[0]

        if gripper_array is None:
            gripper_array = np.full(T, 0.5, dtype=np.float32)
        if images is None:
            images = np.zeros((T, 8, 8, 3), dtype=np.uint8)

        traj = Trajectory(
            task=task,
            direction=direction,
            success=True,
            metadata={"source": "joint_trajectory"},
        )

        for t in range(T):
            # Compute action as next joint position (for training)
            next_joints = joints_array[min(t + 1, T - 1)]

            ts = TimeStep(
                image=images[t].astype(np.uint8),
                joint_positions=joints_array[t].astype(np.float32),
                gripper_open=float(gripper_array[t]),
                action_joints=next_joints.astype(np.float32),
                action_gripper=float(gripper_array[min(t + 1, T - 1)]),
                timestamp=t * dt,
            )
            traj.add_step(ts)

        idx = self.dataset.add(traj)
        logger.info(f"Imported joint trajectory: {T} steps → index {idx}")
        return idx

    def import_directory(
        self,
        demo_dir: str | Path,
        task: str,
        direction: str = "forward",
        pattern: str = "*.npz",
    ) -> list[int]:
        """Import all demos from a directory.

        Args:
            demo_dir: Directory containing demo files.
            task: Task name.
            pattern: Glob pattern for demo files.

        Returns list of trajectory indices.
        """
        demo_dir = Path(demo_dir)
        indices = []

        for demo_file in sorted(demo_dir.glob(pattern)):
            if demo_file.suffix == ".npz":
                idx = self.import_from_npz(demo_file, task, direction)
            elif demo_file.suffix in (".hdf5", ".h5"):
                idx = self.import_from_hdf5(demo_file, task, direction)
            else:
                continue
            indices.append(idx)

        logger.info(f"Imported {len(indices)} demos from {demo_dir}")
        return indices
