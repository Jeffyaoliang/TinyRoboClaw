"""Deployment supervisor: process supervision + failure detection (paper Sec 3.3)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from loguru import logger

from roboclaw.paper.config import DeploymentConfig
from roboclaw.paper.eap.judge import SuccessJudge
from roboclaw.paper.eap.trajectory import TimeStep, Trajectory, TrajectoryDataset
from roboclaw.paper.policy.manager import PolicyManager
from roboclaw.paper.sim.base_env import BaseEnvironment


class FailureType(str, Enum):
    NON_DEGRADING = "non_degrading"  # Scene unchanged → retry same policy
    DEGRADING = "degrading"  # Scene worsened → switch to recovery policy


@dataclass
class SupervisorState:
    """Current supervisor monitoring state."""

    subtask: str = ""
    step_count: int = 0
    retry_count: int = 0
    failure_type: FailureType | None = None
    observations: list[str] = field(default_factory=list)


class DeploymentSupervisor:
    """Monitors policy execution and handles failures during deployment.

    Periodically checks environment state via:
    - Camera observations (VLM-based scene description)
    - Robot joint/gripper stats

    Failure handling:
    - Non-degrading: retry with same policy (up to max_retries)
    - Degrading: switch to recovery/inverse policy
    """

    def __init__(
        self,
        config: DeploymentConfig,
        env: BaseEnvironment,
        policy_manager: PolicyManager,
        judge: SuccessJudge,
        dataset: TrajectoryDataset | None = None,
    ):
        self.config = config
        self.env = env
        self.policy_manager = policy_manager
        self.judge = judge
        self.dataset = dataset
        self._state = SupervisorState()
        self._monitoring = False

    async def supervise_subtask(
        self,
        subtask: str,
        instruction: str,
        max_steps: int = 200,
    ) -> tuple[bool, str]:
        """Supervise execution of a single subtask.

        Returns:
            (success, reason) tuple.
        """
        self._state = SupervisorState(subtask=subtask)
        self._monitoring = True

        logger.info(f"Supervising subtask: {subtask}")

        # Start collecting deployment trajectory
        trajectory = Trajectory(task=subtask, direction="forward")

        while self._state.retry_count <= self.config.max_retries:
            # Start policy execution
            result = await self.policy_manager.start_policy(
                instruction=instruction,
                direction="forward",
                max_steps=max_steps,
            )

            if result.startswith("error:"):
                return False, result

            # Monitor loop
            success, reason = await self._monitor_execution(
                instruction=instruction,
                trajectory=trajectory,
            )

            if success:
                trajectory.success = True
                if self.config.collect_deployment_data and self.dataset:
                    self.dataset.add(trajectory)
                return True, reason

            # Classify failure
            failure_type = await self._classify_failure(instruction)
            self._state.failure_type = failure_type

            if failure_type == FailureType.NON_DEGRADING:
                self._state.retry_count += 1
                logger.info(
                    f"Non-degrading failure, retrying ({self._state.retry_count}/{self.config.max_retries})"
                )
                continue
            else:
                # Degrading: attempt recovery
                logger.warning("Degrading failure, attempting recovery")
                recovered = await self._attempt_recovery(instruction)
                if not recovered:
                    if self.config.collect_deployment_data and self.dataset:
                        self.dataset.add(trajectory)
                    return False, "degrading_failure: recovery unsuccessful"
                self._state.retry_count += 1

        if self.config.collect_deployment_data and self.dataset:
            self.dataset.add(trajectory)
        return False, f"max_retries_exceeded ({self.config.max_retries})"

    async def _monitor_execution(
        self,
        instruction: str,
        trajectory: Trajectory,
    ) -> tuple[bool, str]:
        """Monitor policy execution, collecting observations periodically."""
        interval = self.config.monitor_interval

        while self.policy_manager.is_running:
            await asyncio.sleep(interval)

            # Capture observation
            obs = self.env.get_observation()
            ts = TimeStep(
                image=obs.image.copy(),
                joint_positions=obs.joint_positions.copy(),
                gripper_open=obs.gripper_open,
                timestamp=obs.timestamp,
            )
            trajectory.add_step(ts)
            self._state.step_count += 1

        # Judge success after policy finishes
        image = self.env.capture_image()
        success, reason = await self.judge.judge(
            image=image,
            instruction=instruction,
            direction="forward",
        )
        return success, reason

    async def _classify_failure(self, instruction: str) -> FailureType:
        """Classify failure type by comparing scene state."""
        image = self.env.capture_image()
        success, reason = await self.judge.judge(
            image=image,
            instruction=instruction,
            direction="forward",
        )

        # If the scene looks similar to start → non-degrading
        # If the scene has changed adversely → degrading
        if "no change" in reason.lower() or "same" in reason.lower():
            return FailureType.NON_DEGRADING
        return FailureType.DEGRADING

    async def _attempt_recovery(self, instruction: str) -> bool:
        """Attempt recovery using inverse policy."""
        await self.policy_manager.switch_direction("inverse")
        result = await self.policy_manager.start_policy(
            instruction=f"undo: {instruction}",
            direction="inverse",
            max_steps=100,
        )

        if result.startswith("error:"):
            return False

        # Wait for inverse policy to finish
        while self.policy_manager.is_running:
            await asyncio.sleep(self.config.monitor_interval)

        # Judge if recovery was successful
        image = self.env.capture_image()
        success, _ = await self.judge.judge(
            image=image,
            instruction=instruction,
            direction="inverse",
        )

        # Switch back to forward
        await self.policy_manager.switch_direction("forward")
        return success
