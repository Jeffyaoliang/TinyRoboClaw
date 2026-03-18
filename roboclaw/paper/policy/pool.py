"""Policy pool: multi-policy management for long-horizon tasks (paper Sec 3.3).

Manages a pool of forward/inverse policy pairs, one per subtask.
E.g., "pick lipstick" has its own forward+inverse policies,
"place in holder" has another pair, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from roboclaw.paper.policy.base import PolicyInterface


@dataclass
class PolicyPair:
    """A forward + inverse policy pair for a subtask."""

    subtask: str
    forward: PolicyInterface
    inverse: PolicyInterface
    checkpoint_forward: str | None = None
    checkpoint_inverse: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


class PolicyPool:
    """Manages a pool of policy pairs for multiple subtasks.

    Usage:
        pool = PolicyPool()
        pool.register("pick lipstick", fwd_policy, inv_policy)
        pool.register("place in holder", fwd_policy2, inv_policy2)

        pair = pool.get("pick lipstick")
        action = await pair.forward.infer(obs, "pick lipstick")
    """

    def __init__(self) -> None:
        self._pairs: dict[str, PolicyPair] = {}
        self._default_pair: PolicyPair | None = None

    def register(
        self,
        subtask: str,
        forward: PolicyInterface,
        inverse: PolicyInterface,
        checkpoint_forward: str | None = None,
        checkpoint_inverse: str | None = None,
    ) -> None:
        """Register a policy pair for a subtask."""
        pair = PolicyPair(
            subtask=subtask,
            forward=forward,
            inverse=inverse,
            checkpoint_forward=checkpoint_forward,
            checkpoint_inverse=checkpoint_inverse,
        )
        self._pairs[subtask] = pair

        # First registered pair becomes default
        if self._default_pair is None:
            self._default_pair = pair

        logger.info(
            f"Registered policy pair for '{subtask}': "
            f"fwd={forward.name}, inv={inverse.name}"
        )

    def get(self, subtask: str) -> PolicyPair | None:
        """Get the policy pair for a subtask."""
        return self._pairs.get(subtask)

    def get_or_default(self, subtask: str) -> PolicyPair | None:
        """Get policy pair for subtask, falling back to default."""
        return self._pairs.get(subtask) or self._default_pair

    def get_forward(self, subtask: str) -> PolicyInterface | None:
        """Get the forward policy for a subtask."""
        pair = self.get_or_default(subtask)
        return pair.forward if pair else None

    def get_inverse(self, subtask: str) -> PolicyInterface | None:
        """Get the inverse policy for a subtask."""
        pair = self.get_or_default(subtask)
        return pair.inverse if pair else None

    def set_default(self, subtask: str) -> None:
        """Set a registered pair as the default."""
        pair = self._pairs.get(subtask)
        if pair:
            self._default_pair = pair

    def update_policy(
        self,
        subtask: str,
        direction: str,
        policy: PolicyInterface,
        checkpoint: str | None = None,
    ) -> None:
        """Update a single policy within a pair (e.g., after retraining)."""
        pair = self._pairs.get(subtask)
        if pair is None:
            raise KeyError(f"No policy pair registered for '{subtask}'")

        if direction == "forward":
            pair.forward = policy
            pair.checkpoint_forward = checkpoint
        elif direction == "inverse":
            pair.inverse = policy
            pair.checkpoint_inverse = checkpoint
        else:
            raise ValueError(f"Invalid direction: {direction}")

        logger.info(f"Updated {direction} policy for '{subtask}': {policy.name}")

    def update_metrics(self, subtask: str, metrics: dict[str, Any]) -> None:
        """Update metrics for a policy pair."""
        pair = self._pairs.get(subtask)
        if pair:
            pair.metrics.update(metrics)

    @property
    def subtasks(self) -> list[str]:
        """List all registered subtask names."""
        return list(self._pairs.keys())

    @property
    def size(self) -> int:
        return len(self._pairs)

    def summary(self) -> list[dict[str, Any]]:
        """Get summary of all policy pairs."""
        return [
            {
                "subtask": pair.subtask,
                "forward": pair.forward.name,
                "inverse": pair.inverse.name,
                "checkpoint_forward": pair.checkpoint_forward,
                "checkpoint_inverse": pair.checkpoint_inverse,
                "metrics": pair.metrics,
            }
            for pair in self._pairs.values()
        ]
