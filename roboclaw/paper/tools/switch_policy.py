"""SwitchPolicy tool: switch between forward and inverse VLA policies."""

from __future__ import annotations

from typing import Any

from roboclaw.agent.tools.base import Tool


class SwitchPolicyTool(Tool):
    """Switch the active VLA policy direction (forward ↔ inverse)."""

    def __init__(self, policy_manager: Any) -> None:
        self._policy_manager = policy_manager

    @property
    def name(self) -> str:
        return "switch_policy"

    @property
    def description(self) -> str:
        return (
            "Switch the active policy direction between forward and inverse. "
            "Forward policies execute the task; inverse policies undo/reset it."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["forward", "inverse"],
                    "description": "Target direction to switch to.",
                },
            },
            "required": ["direction"],
        }

    async def execute(self, **kwargs: Any) -> str:
        direction = kwargs["direction"]
        try:
            await self._policy_manager.switch_direction(direction)
            return f"Switched to {direction} policy."
        except Exception as e:
            return f"Failed to switch policy: {e}"
