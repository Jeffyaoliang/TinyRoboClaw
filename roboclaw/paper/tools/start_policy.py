"""StartPolicy tool: launch a VLA policy for execution."""

from __future__ import annotations

from typing import Any

from roboclaw.agent.tools.base import Tool


class StartPolicyTool(Tool):
    """Start a VLA policy to execute a manipulation task."""

    def __init__(self, policy_manager: Any) -> None:
        self._policy_manager = policy_manager

    @property
    def name(self) -> str:
        return "start_policy"

    @property
    def description(self) -> str:
        return (
            "Start executing a VLA policy on the robot. "
            "Specify the instruction and optionally the direction (forward/inverse) "
            "and maximum number of steps."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "instruction": {
                    "type": "string",
                    "description": "Natural language instruction for the policy.",
                },
                "direction": {
                    "type": "string",
                    "enum": ["forward", "inverse"],
                    "description": "Policy direction: forward (do task) or inverse (undo/reset).",
                },
                "max_steps": {
                    "type": "integer",
                    "description": "Maximum action chunks to execute.",
                    "minimum": 1,
                },
            },
            "required": ["instruction"],
        }

    async def execute(self, **kwargs: Any) -> str:
        instruction = kwargs["instruction"]
        direction = kwargs.get("direction", "forward")
        max_steps = kwargs.get("max_steps", 200)

        try:
            result = await self._policy_manager.start_policy(
                instruction=instruction,
                direction=direction,
                max_steps=max_steps,
            )
            return f"Policy started: direction={direction}, instruction='{instruction}'. Result: {result}"
        except Exception as e:
            return f"Failed to start policy: {e}"
