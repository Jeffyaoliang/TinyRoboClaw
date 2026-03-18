"""Deployment data recycler — feeds deployment trajectories back to training.

Closes the loop: deployment → collect trajectories → filter → add to dataset → retrain.
Paper Sec 3.3: "deployment trajectories are also recycled into D."
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from roboclaw.paper.eap.trajectory import TrajectoryDataset


class DeploymentRecycler:
    """Recycles deployment trajectories into the training dataset.

    Monitors a deployment dataset and periodically merges successful
    trajectories into the main training dataset.
    """

    def __init__(
        self,
        training_dataset: TrajectoryDataset,
        deployment_dataset: TrajectoryDataset,
        success_only: bool = True,
        forward_only: bool = True,
    ):
        self.training_dataset = training_dataset
        self.deployment_dataset = deployment_dataset
        self.success_only = success_only
        self.forward_only = forward_only
        self._last_recycled_idx = 0

    def recycle(self) -> int:
        """Merge new deployment trajectories into training dataset.

        Returns number of trajectories recycled.
        """
        recycled = 0
        current_size = self.deployment_dataset.size

        if current_size <= self._last_recycled_idx:
            return 0

        for idx in range(self._last_recycled_idx, current_size):
            try:
                meta = self.deployment_dataset.get_meta(idx)
            except IndexError:
                break

            # Filter criteria
            if self.success_only and not meta.get("success", False):
                continue
            if self.forward_only and meta.get("direction") != "forward":
                continue

            # Load images and rebuild trajectory for the training dataset
            images = self.deployment_dataset.load_images(idx)

            from roboclaw.paper.eap.trajectory import Trajectory, TimeStep
            import numpy as np

            traj = Trajectory(
                task=meta["task"],
                direction=meta["direction"],
                success=meta["success"],
                metadata={**meta.get("metadata", {}), "source": "deployment"},
            )

            for step_idx, step_data in enumerate(meta.get("steps", [])):
                img = images[step_idx] if step_idx < len(images) else np.zeros((8, 8, 3), dtype=np.uint8)
                ts = TimeStep(
                    image=img,
                    joint_positions=np.array(step_data["joint_positions"], dtype=np.float32),
                    gripper_open=step_data["gripper_open"],
                    action_joints=np.array(step_data["action_joints"], dtype=np.float32) if "action_joints" in step_data else None,
                    action_gripper=step_data.get("action_gripper"),
                    timestamp=step_data.get("timestamp", 0.0),
                )
                traj.add_step(ts)

            self.training_dataset.add(traj)
            recycled += 1

        self._last_recycled_idx = current_size
        if recycled > 0:
            logger.info(f"Recycled {recycled} deployment trajectories into training dataset")

        return recycled

    @property
    def pending_count(self) -> int:
        """Number of deployment trajectories not yet recycled."""
        return max(0, self.deployment_dataset.size - self._last_recycled_idx)
