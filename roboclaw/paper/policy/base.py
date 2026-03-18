"""Abstract policy interface for VLA models."""

from __future__ import annotations

from abc import ABC, abstractmethod

from roboclaw.paper.sim.base_env import ActionChunk, Observation


class PolicyInterface(ABC):
    """Abstract interface for a VLA (Vision-Language-Action) policy."""

    @abstractmethod
    async def infer(self, obs: Observation, instruction: str) -> ActionChunk:
        """Given observation + language instruction, produce an action chunk."""

    @abstractmethod
    def reset(self) -> None:
        """Reset any internal state (e.g., action queue, hidden states)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable policy name."""

    @property
    def is_ready(self) -> bool:
        """Whether the policy is loaded and ready for inference."""
        return True
