"""Mock policy for testing without a real VLA server."""

from __future__ import annotations

import numpy as np

from roboclaw.paper.sim.base_env import ActionChunk, Observation
from roboclaw.paper.policy.base import PolicyInterface


class MockPolicy(PolicyInterface):
    """Random/rule-based policy for testing flywheel logic.

    Modes:
    - "random": Gaussian noise actions
    - "reach": Move toward a fixed target position
    - "replay": Replay a recorded trajectory in reverse (for inverse policy testing)
    """

    def __init__(
        self,
        name: str = "mock_forward",
        mode: str = "random",
        chunk_size: int = 50,
        num_joints: int = 6,
        seed: int = 0,
    ):
        self._name = name
        self.mode = mode
        self.chunk_size = chunk_size
        self.num_joints = num_joints
        self._rng = np.random.default_rng(seed)
        self._step = 0

    @property
    def name(self) -> str:
        return self._name

    def reset(self) -> None:
        self._step = 0

    async def infer(self, obs: Observation, instruction: str) -> ActionChunk:
        if self.mode == "reach":
            return self._reach_action(obs)
        return self._random_action()

    def _random_action(self) -> ActionChunk:
        joints = self._rng.normal(0, 0.1, (self.chunk_size, self.num_joints)).astype(
            np.float32
        )
        gripper = self._rng.uniform(0, 1, self.chunk_size).astype(np.float32)
        self._step += 1
        return ActionChunk(joint_targets=joints, gripper_targets=gripper)

    def _reach_action(self, obs: Observation) -> ActionChunk:
        """Move joints toward zero (home position) as a simple reach target."""
        current = obs.joint_positions
        delta = -current * 0.1  # Move 10% toward zero each step
        joints = np.tile(
            (current + delta).astype(np.float32), (self.chunk_size, 1)
        )
        # Add small noise
        joints += self._rng.normal(0, 0.01, joints.shape).astype(np.float32)
        gripper = np.full(self.chunk_size, 0.5, dtype=np.float32)
        self._step += 1
        return ActionChunk(joint_targets=joints, gripper_targets=gripper)
