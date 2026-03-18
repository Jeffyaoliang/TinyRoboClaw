"""Trajectory data structures and dataset management (paper Sec 3.2)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np


@dataclass
class TimeStep:
    """A single timestep in a trajectory."""

    image: np.ndarray  # (H, W, 3) uint8
    joint_positions: np.ndarray  # (num_joints,)
    gripper_open: float
    action_joints: np.ndarray | None = None  # (num_joints,) target
    action_gripper: float | None = None
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "joint_positions": self.joint_positions.tolist(),
            "gripper_open": self.gripper_open,
            "timestamp": self.timestamp,
        }
        if self.action_joints is not None:
            d["action_joints"] = self.action_joints.tolist()
        if self.action_gripper is not None:
            d["action_gripper"] = self.action_gripper
        return d


@dataclass
class Trajectory:
    """A full trajectory: sequence of timesteps with metadata."""

    task: str
    direction: str  # "forward" | "inverse"
    success: bool = False
    steps: list[TimeStep] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.steps)

    def add_step(self, step: TimeStep) -> None:
        self.steps.append(step)

    def get_images(self) -> np.ndarray:
        """Stack all images into (T, H, W, 3) array."""
        return np.stack([s.image for s in self.steps])

    def get_joint_trajectory(self) -> np.ndarray:
        """Get (T, num_joints) array of joint positions."""
        return np.stack([s.joint_positions for s in self.steps])

    def get_action_trajectory(self) -> np.ndarray | None:
        """Get (T, num_joints) array of action targets, if available."""
        actions = [s.action_joints for s in self.steps if s.action_joints is not None]
        return np.stack(actions) if actions else None

    def to_meta_dict(self) -> dict[str, Any]:
        """Serializable metadata (without images)."""
        return {
            "task": self.task,
            "direction": self.direction,
            "success": self.success,
            "length": self.length,
            "metadata": self.metadata,
            "steps": [s.to_dict() for s in self.steps],
        }


class TrajectoryDataset:
    """Persistent trajectory dataset: JSONL metadata + npz image arrays.

    Storage layout:
        data_dir/
            index.jsonl        # One JSON line per trajectory
            images/
                traj_0000.npz  # Compressed image arrays
                traj_0001.npz
                ...
    """

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "images").mkdir(exist_ok=True)
        self._index_path = self.data_dir / "index.jsonl"
        self._count = self._load_count()

    def _load_count(self) -> int:
        if not self._index_path.exists():
            return 0
        with open(self._index_path) as f:
            return sum(1 for _ in f)

    def add(self, trajectory: Trajectory) -> int:
        """Add a trajectory to the dataset. Returns its index."""
        idx = self._count
        traj_id = f"traj_{idx:04d}"

        # Save images as npz
        images = trajectory.get_images()
        np.savez_compressed(self.data_dir / "images" / f"{traj_id}.npz", images=images)

        # Append metadata to index
        meta = trajectory.to_meta_dict()
        meta["traj_id"] = traj_id
        with open(self._index_path, "a") as f:
            f.write(json.dumps(meta) + "\n")

        self._count += 1
        return idx

    def get_meta(self, idx: int) -> dict[str, Any]:
        """Get metadata for a trajectory by index."""
        if not self._index_path.exists():
            raise IndexError(f"Trajectory index {idx} out of range (dataset is empty)")
        with open(self._index_path) as f:
            for i, line in enumerate(f):
                if i == idx:
                    return json.loads(line)
        raise IndexError(f"Trajectory index {idx} out of range")

    def load_images(self, idx: int) -> np.ndarray:
        """Load images for a trajectory."""
        traj_id = f"traj_{idx:04d}"
        data = np.load(self.data_dir / "images" / f"{traj_id}.npz")
        return data["images"]

    def filter(
        self,
        task: str | None = None,
        direction: str | None = None,
        success: bool | None = None,
    ) -> list[dict[str, Any]]:
        """Filter trajectories by criteria."""
        results = []
        if not self._index_path.exists():
            return results
        with open(self._index_path) as f:
            for line in f:
                meta = json.loads(line)
                if task is not None and meta["task"] != task:
                    continue
                if direction is not None and meta["direction"] != direction:
                    continue
                if success is not None and meta["success"] != success:
                    continue
                results.append(meta)
        return results

    def iter_all(self) -> Iterator[dict[str, Any]]:
        """Iterate over all trajectory metadata."""
        if not self._index_path.exists():
            return
        with open(self._index_path) as f:
            for line in f:
                yield json.loads(line)

    @property
    def size(self) -> int:
        return self._count

    def stats(self) -> dict[str, Any]:
        """Get dataset statistics."""
        total = 0
        success_fwd = 0
        success_inv = 0
        total_fwd = 0
        total_inv = 0

        for meta in self.iter_all():
            total += 1
            if meta["direction"] == "forward":
                total_fwd += 1
                if meta["success"]:
                    success_fwd += 1
            else:
                total_inv += 1
                if meta["success"]:
                    success_inv += 1

        return {
            "total": total,
            "forward": total_fwd,
            "inverse": total_inv,
            "success_rate_forward": success_fwd / total_fwd if total_fwd else 0.0,
            "success_rate_inverse": success_inv / total_inv if total_inv else 0.0,
        }

    def export_for_training(self, output_dir: str | Path, format: str = "lerobot") -> Path:
        """Export dataset in training-ready format.

        Args:
            output_dir: Directory to write exported data.
            format: Export format ("lerobot" or "raw").

        Returns:
            Path to the exported dataset.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if format == "lerobot":
            return self._export_lerobot(output_path)
        return self._export_raw(output_path)

    def _export_lerobot(self, output_path: Path) -> Path:
        """Export in LeRobot HDF5-like format (simplified as npz + json)."""
        all_joints = []
        all_actions = []
        all_episode_ids = []
        meta_records = []

        for meta in self.iter_all():
            # Only export successful forward trajectories for training
            if not meta["success"] or meta["direction"] != "forward":
                continue

            ep_id = len(meta_records)
            for step in meta["steps"]:
                all_joints.append(step["joint_positions"])
                if "action_joints" in step:
                    all_actions.append(step["action_joints"])
                else:
                    all_actions.append(step["joint_positions"])
                all_episode_ids.append(ep_id)

            meta_records.append({
                "episode_id": ep_id,
                "task": meta["task"],
                "length": meta["length"],
            })

        if all_joints:
            np.savez_compressed(
                output_path / "dataset.npz",
                observations=np.array(all_joints, dtype=np.float32),
                actions=np.array(all_actions, dtype=np.float32),
                episode_ids=np.array(all_episode_ids, dtype=np.int32),
            )

        with open(output_path / "meta.json", "w") as f:
            json.dump({"episodes": meta_records, "total_steps": len(all_joints)}, f, indent=2)

        return output_path

    def _export_raw(self, output_path: Path) -> Path:
        """Export as raw npz files (images + actions per trajectory)."""
        for meta in self.iter_all():
            traj_id = meta["traj_id"]
            # Copy image npz
            src = self.data_dir / "images" / f"{traj_id}.npz"
            if src.exists():
                import shutil
                shutil.copy2(src, output_path / f"{traj_id}_images.npz")

            # Save actions
            actions = []
            for step in meta["steps"]:
                if "action_joints" in step:
                    actions.append(step["action_joints"])
            if actions:
                np.savez_compressed(
                    output_path / f"{traj_id}_actions.npz",
                    actions=np.array(actions, dtype=np.float32),
                )

        return output_path
