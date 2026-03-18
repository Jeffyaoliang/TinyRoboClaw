"""Policy lifecycle manager: load, start, stop, switch policies."""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
from loguru import logger

from roboclaw.paper.config import PaperConfig
from roboclaw.paper.policy.base import PolicyInterface
from roboclaw.paper.sim.base_env import ActionChunk, BaseEnvironment


class PolicyManager:
    """Manages forward/inverse policy pairs and their lifecycle.

    Responsibilities:
    - Load/unload policies
    - Start/stop policy execution
    - Switch between forward and inverse directions
    - Track execution state (running, steps, trajectory)
    """

    def __init__(
        self,
        config: PaperConfig,
        env: BaseEnvironment,
        forward_policy: PolicyInterface | None = None,
        inverse_policy: PolicyInterface | None = None,
    ):
        self.config = config
        self.env = env
        self._policies: dict[str, PolicyInterface] = {}
        self._active_direction: str = "forward"
        self._running = False
        self._step_count = 0
        self._execution_task: asyncio.Task[Any] | None = None

        if forward_policy:
            self._policies["forward"] = forward_policy
        if inverse_policy:
            self._policies["inverse"] = inverse_policy

    @property
    def active_policy(self) -> PolicyInterface | None:
        return self._policies.get(self._active_direction)

    @property
    def active_direction(self) -> str:
        return self._active_direction

    @property
    def is_running(self) -> bool:
        return self._running

    def set_policy(self, direction: str, policy: PolicyInterface) -> None:
        """Register a policy for a given direction."""
        self._policies[direction] = policy
        logger.info(f"Registered {direction} policy: {policy.name}")

    async def start_policy(
        self,
        instruction: str,
        direction: str = "forward",
        max_steps: int = 200,
    ) -> str:
        """Start executing the specified policy."""
        if self._running:
            return "error: policy already running"

        policy = self._policies.get(direction)
        if policy is None:
            return f"error: no {direction} policy registered"

        self._active_direction = direction
        self._running = True
        self._step_count = 0
        policy.reset()

        logger.info(f"Starting {direction} policy: {policy.name}")

        # Run policy execution in background
        self._execution_task = asyncio.create_task(
            self._execute_policy(policy, instruction, max_steps)
        )

        return f"started:{direction}:{policy.name}"

    async def _execute_policy(
        self,
        policy: PolicyInterface,
        instruction: str,
        max_steps: int,
    ) -> None:
        """Execute policy loop until stopped or max steps reached."""
        try:
            for step in range(max_steps):
                if not self._running:
                    break

                obs = self.env.get_observation()
                action = await policy.infer(obs, instruction)
                self.env.step(action)
                self._step_count += 1

                # Yield control
                await asyncio.sleep(0)

        except Exception as e:
            logger.error(f"Policy execution error: {e}")
        finally:
            self._running = False
            logger.info(f"Policy execution ended after {self._step_count} steps")

    async def stop_policy(self) -> None:
        """Stop the currently running policy."""
        self._running = False
        if self._execution_task is not None:
            await self._execution_task
            self._execution_task = None
        logger.info("Policy stopped")

    async def switch_direction(self, direction: str) -> None:
        """Switch to a different policy direction."""
        if self._running:
            await self.stop_policy()
        self._active_direction = direction
        logger.info(f"Switched to {direction} direction")

    async def reload_policy(
        self,
        direction: str,
        policy: PolicyInterface,
    ) -> None:
        """Replace and reload a policy (e.g., after retraining)."""
        if self._running and self._active_direction == direction:
            await self.stop_policy()
        self._policies[direction] = policy
        logger.info(f"Reloaded {direction} policy: {policy.name}")
