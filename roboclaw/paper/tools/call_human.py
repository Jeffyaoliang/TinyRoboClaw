"""CallHuman tool: request human intervention."""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Awaitable

from roboclaw.agent.tools.base import Tool


class CallHumanTool(Tool):
    """Request human intervention when the robot is stuck."""

    def __init__(self, callback: Callable[[str], Awaitable[str]] | None = None) -> None:
        self._callback = callback

    @property
    def name(self) -> str:
        return "call_human"

    @property
    def description(self) -> str:
        return (
            "Request human help when the robot is stuck or in a situation "
            "that cannot be resolved autonomously. Use as a last resort."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Description of the situation and what help is needed.",
                },
            },
            "required": ["message"],
        }

    async def execute(self, **kwargs: Any) -> str:
        message = kwargs["message"]

        if self._callback is not None:
            response = await self._callback(message)
            return f"Human response: {response}"

        # Default: stdin-based interaction
        print(f"\n[HUMAN HELP REQUESTED] {message}")
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, lambda: input("Your response (or press Enter to skip): ")
            )
            return f"Human response: {response}" if response else "Human acknowledged (no response)."
        except EOFError:
            return "Human help requested but no interactive input available."
