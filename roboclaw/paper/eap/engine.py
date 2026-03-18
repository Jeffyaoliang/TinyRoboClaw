"""EAP (Entangled Action Pair) engine (paper Sec 3.2, formulas 6-8).

Core loop:
  for each episode:
    τ_forward = run forward policy
    success_fwd = judge(forward)
    τ_inverse = run inverse policy
    success_inv = judge(inverse)
    dataset.add(τ_forward, τ_inverse)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from loguru import logger

from roboclaw.paper.config import EAPConfig, PaperConfig
from roboclaw.paper.eap.judge import SuccessJudge
from roboclaw.paper.eap.trajectory import TimeStep, Trajectory, TrajectoryDataset
from roboclaw.paper.policy.base import PolicyInterface
from roboclaw.paper.sim.base_env import BaseEnvironment


@dataclass
class EAPBatchResult:
    """Result of running a batch of EAP episodes."""

    num_episodes: int = 0
    success_forward: int = 0
    success_inverse: int = 0
    total_steps: int = 0

    @property
    def success_rate_forward(self) -> float:
        return self.success_forward / self.num_episodes if self.num_episodes else 0.0

    @property
    def success_rate_inverse(self) -> float:
        return self.success_inverse / self.num_episodes if self.num_episodes else 0.0


class EAPEngine:
    """Entangled Action Pair data collection engine."""

    def __init__(
        self,
        config: PaperConfig,
        env: BaseEnvironment,
        forward_policy: PolicyInterface,
        inverse_policy: PolicyInterface,
        judge: SuccessJudge,
        dataset: TrajectoryDataset,
    ):
        self.config = config
        self.env = env
        self.forward_policy = forward_policy
        self.inverse_policy = inverse_policy
        self.judge = judge
        self.dataset = dataset
        self._eap_config = config.eap

    async def run_batch(self, num_episodes: int | None = None) -> EAPBatchResult:
        """Run a batch of EAP episodes."""
        n = num_episodes or self._eap_config.episodes_per_batch
        result = EAPBatchResult()

        for ep in range(n):
            logger.info(f"EAP episode {ep + 1}/{n}")

            # Reset environment
            self.env.reset()
            self.forward_policy.reset()
            self.inverse_policy.reset()

            # --- Forward pass ---
            traj_fwd = await self._run_policy(
                policy=self.forward_policy,
                instruction=self.config.task_instruction,
                direction="forward",
            )

            # Judge forward success
            image = self.env.capture_image()
            success_fwd, reason_fwd = await self.judge.judge(
                image=image,
                instruction=self.config.task_instruction,
                direction="forward",
            )
            traj_fwd.success = success_fwd
            traj_fwd.metadata["judge_reason"] = reason_fwd
            self.dataset.add(traj_fwd)

            if success_fwd:
                result.success_forward += 1

            logger.info(f"  Forward: success={success_fwd}, reason={reason_fwd}")

            # --- Inverse pass (reset the scene) ---
            traj_inv = await self._run_policy(
                policy=self.inverse_policy,
                instruction=self.config.inverse_instruction or f"undo: {self.config.task_instruction}",
                direction="inverse",
            )

            # Judge inverse success (scene reset)
            image = self.env.capture_image()
            success_inv, reason_inv = await self.judge.judge(
                image=image,
                instruction=self.config.task_instruction,
                direction="inverse",
            )
            traj_inv.success = success_inv
            traj_inv.metadata["judge_reason"] = reason_inv
            self.dataset.add(traj_inv)

            if success_inv:
                result.success_inverse += 1

            logger.info(f"  Inverse: success={success_inv}, reason={reason_inv}")

            result.num_episodes += 1
            result.total_steps += traj_fwd.length + traj_inv.length

        logger.info(
            f"Batch done: {result.num_episodes} episodes, "
            f"fwd_rate={result.success_rate_forward:.2%}, "
            f"inv_rate={result.success_rate_inverse:.2%}"
        )
        return result

    async def _run_policy(
        self,
        policy: PolicyInterface,
        instruction: str,
        direction: str,
    ) -> Trajectory:
        """Run a single policy until max steps, collecting a trajectory."""
        traj = Trajectory(
            task=self.config.task_name,
            direction=direction,
        )
        max_steps = self._eap_config.max_steps_per_episode

        for step_idx in range(max_steps):
            obs = self.env.get_observation()

            # Get action from policy
            action = await policy.infer(obs, instruction)

            # Record timestep
            ts = TimeStep(
                image=obs.image.copy(),
                joint_positions=obs.joint_positions.copy(),
                gripper_open=obs.gripper_open,
                action_joints=action.joint_targets[0].copy() if action.chunk_size > 0 else None,
                action_gripper=float(action.gripper_targets[0]) if action.chunk_size > 0 else None,
                timestamp=obs.timestamp,
            )
            traj.add_step(ts)

            # Execute action in environment
            self.env.step(action)

        return traj
