"""FetchRobotStats tool: get current robot joint/gripper state."""

from __future__ import annotations

import json
from typing import Any

from roboclaw.agent.tools.base import Tool


class FetchRobotStatsTool(Tool):
    """Fetch current robot joint positions and gripper state."""

    def __init__(self, env: Any) -> None:
        self._env = env

    @property
    def name(self) -> str:
        return "fetch_robot_stats"

    @property
    def description(self) -> str:
        return "Get the current robot joint positions, gripper state, and object positions."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

    async def execute(self, **kwargs: Any) -> str:
        try:
            joints = self._env.get_joint_positions()
            gripper = self._env.get_gripper_state()
            objects = self._env.get_object_positions()

            stats = {
                "joint_positions": [round(float(j), 4) for j in joints],
                "gripper_open": round(float(gripper), 4),
                "objects": {
                    name: [round(float(v), 4) for v in pos]
                    for name, pos in objects.items()
                },
            }
            return json.dumps(stats, indent=2)
        except Exception as e:
            return f"Failed to fetch robot stats: {e}"
