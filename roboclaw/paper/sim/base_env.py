"""Abstract base environment for robot simulation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Observation:
    """A single observation from the environment."""

    image: np.ndarray  # (H, W, 3) uint8
    joint_positions: np.ndarray  # (num_joints,)
    gripper_open: float  # 0.0 = closed, 1.0 = open
    timestamp: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionChunk:
    """A chunk of actions to execute (VLA output)."""

    joint_targets: np.ndarray  # (chunk_size, num_joints)
    gripper_targets: np.ndarray  # (chunk_size,)

    @property
    def chunk_size(self) -> int:
        return self.joint_targets.shape[0]


class BaseEnvironment(ABC):
    """Abstract base for robot environments."""

    @abstractmethod
    def reset(self) -> Observation:
        """Reset environment to initial state."""

    @abstractmethod
    def step(self, action: ActionChunk) -> tuple[Observation, dict[str, Any]]:
        """Execute an action chunk. Returns (observation, info)."""

    @abstractmethod
    def get_observation(self) -> Observation:
        """Get current observation without stepping."""

    @abstractmethod
    def capture_image(self) -> np.ndarray:
        """Capture current camera image as (H, W, 3) uint8."""

    @abstractmethod
    def get_joint_positions(self) -> np.ndarray:
        """Get current joint positions."""

    @abstractmethod
    def get_gripper_state(self) -> float:
        """Get gripper openness: 0.0 = closed, 1.0 = open."""

    @abstractmethod
    def get_object_positions(self) -> dict[str, np.ndarray]:
        """Get positions of all tracked objects."""
