"""TerminatePolicy tool: stop the currently running VLA policy."""

from __future__ import annotations

from typing import Any

from roboclaw.agent.tools.base import Tool


class TerminatePolicyTool(Tool):
    """Terminate the currently running VLA policy."""

    def __init__(self, policy_manager: Any) -> None:
        self._policy_manager = policy_manager

    @property
    def name(self) -> str:
        return "terminate_policy"

    @property
    def description(self) -> str:
        return "Stop the currently running VLA policy execution."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Reason for termination.",
                },
            },
            "required": [],
        }

    async def execute(self, **kwargs: Any) -> str:
        reason = kwargs.get("reason", "agent_decision")
        try:
            await self._policy_manager.stop_policy()
            return f"Policy terminated. Reason: {reason}"
        except Exception as e:
            return f"Failed to terminate policy: {e}"
