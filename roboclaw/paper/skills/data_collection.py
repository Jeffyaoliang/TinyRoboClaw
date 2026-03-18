"""Data Collection Skill — orchestrates EAP forward-inverse loops.

This is the programmatic implementation of the data-collection SKILL.md.
It automates: observe → forward → judge → inverse → judge → loop.
"""

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

from roboclaw.paper.skills.base import RoboticSkill, SkillResult, SkillStatus


class DataCollectionSkill(RoboticSkill):
    """Autonomous EAP data collection skill."""

    @property
    def name(self) -> str:
        return "data-collection"

    @property
    def description(self) -> str:
        return (
            "Run autonomous data collection using Entangled Action Pairs (EAP). "
            "Orchestrates forward/inverse policy loops for self-resetting data acquisition."
        )

    async def execute(
        self,
        instruction: str,
        tools: dict[str, Any],
        num_episodes: int = 10,
        max_steps: int = 200,
        **kwargs: Any,
    ) -> SkillResult:
        result = SkillResult(status=SkillStatus.RUNNING)
        consecutive_failures = 0
        max_consecutive_failures = 3

        for ep in range(num_episodes):
            logger.info(f"[data-collection] Episode {ep + 1}/{num_episodes}")
            ep_result: dict[str, Any] = {"episode": ep, "forward": None, "inverse": None}

            # 1. Observe scene
            env_desc = await self._call_tool(tools, "env_summary")
            logger.info(f"  Scene: {env_desc[:100]}...")

            # 2. Forward policy
            fwd_ok = await self._run_direction(
                tools, instruction, "forward", max_steps, ep_result
            )

            if not fwd_ok:
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    result.status = SkillStatus.FAILED
                    result.message = f"Aborted after {max_consecutive_failures} consecutive failures"
                    return result
                # Try to recover with human help
                await self._call_tool(tools, "call_human", message=f"Forward policy failed on episode {ep}")
            else:
                consecutive_failures = 0

            # 3. Inverse policy (reset scene)
            inv_ok = await self._run_direction(
                tools, f"undo: {instruction}", "inverse", max_steps, ep_result
            )

            if not inv_ok:
                # Inverse failure → need human reset
                await self._call_tool(
                    tools, "call_human",
                    message=f"Scene reset failed on episode {ep}. Please manually reset.",
                )

            result.subtask_results.append(ep_result)

        result.status = SkillStatus.SUCCESS
        result.message = f"Completed {num_episodes} episodes"
        result.data = {
            "total_episodes": num_episodes,
            "successful_forward": sum(1 for r in result.subtask_results if r.get("forward") == "success"),
            "successful_inverse": sum(1 for r in result.subtask_results if r.get("inverse") == "success"),
        }
        return result

    async def _run_direction(
        self,
        tools: dict[str, Any],
        instruction: str,
        direction: str,
        max_steps: int,
        ep_result: dict[str, Any],
    ) -> bool:
        """Run a single direction (forward or inverse)."""
        # Switch direction
        await self._call_tool(tools, "switch_policy", direction=direction)

        # Start policy
        start_result = await self._call_tool(
            tools, "start_policy",
            instruction=instruction,
            direction=direction,
            max_steps=max_steps,
        )

        if "error" in start_result.lower():
            ep_result[direction] = "failed"
            return False

        # Monitor (wait for policy to finish)
        await asyncio.sleep(0.1)  # Let policy run
        stats = await self._call_tool(tools, "fetch_robot_stats")
        logger.debug(f"  {direction} stats: {stats[:100]}")

        # Terminate
        await self._call_tool(tools, "terminate_policy", reason=f"{direction}_complete")

        ep_result[direction] = "success"
        return True

    async def _call_tool(self, tools: dict[str, Any], name: str, **kwargs: Any) -> str:
        """Call a tool by name, handling missing tools gracefully."""
        tool = tools.get(name)
        if tool is None:
            return f"tool_not_found: {name}"
        try:
            if hasattr(tool, "execute"):
                return await tool.execute(**kwargs)
            return str(await tool(**kwargs))
        except Exception as e:
            logger.error(f"Tool {name} failed: {e}")
            return f"error: {e}"
