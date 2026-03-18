"""VLM Agent main loop (paper Sec 3.1).

Implements: observe → reason(CoT) → select_tool → execute → update_memory
"""

from __future__ import annotations

import json
from typing import Any

from loguru import logger

from roboclaw.paper.agent.memory import StructuredMemory
from roboclaw.paper.agent.prompts import (
    COT_REASONING_PROMPT,
    DATA_COLLECTION_SYSTEM,
    DEPLOYMENT_SYSTEM,
)
from roboclaw.paper.config import PaperConfig
from roboclaw.paper.sim.base_env import BaseEnvironment
from roboclaw.providers.base import LLMProvider


class VLMAgentLoop:
    """Core VLM agent loop for data collection or deployment.

    Modes:
    - "data_collection": Runs EAP forward/inverse episodes
    - "deployment": Supervises long-horizon task execution
    """

    def __init__(
        self,
        config: PaperConfig,
        provider: LLMProvider,
        env: BaseEnvironment,
        tools: dict[str, Any] | None = None,
        mode: str = "data_collection",
    ):
        self.config = config
        self.provider = provider
        self.env = env
        self.tools = tools or {}
        self.mode = mode
        self.memory = StructuredMemory()
        self._step = 0
        self._max_iterations = config.vlm.max_iterations
        self._running = False

        # Initialize role identity
        system_prompt = DATA_COLLECTION_SYSTEM if mode == "data_collection" else DEPLOYMENT_SYSTEM
        self.memory.role.system_prompt = system_prompt
        self.memory.role.tool_descriptions = [
            {"name": name, "description": getattr(tool, "description", str(tool))}
            for name, tool in self.tools.items()
        ]

    async def run_episode(self, task_instruction: str) -> list[dict[str, Any]]:
        """Run one full agent episode. Returns list of step records."""
        self.memory.task.task_instruction = task_instruction
        self._step = 0
        self._running = True
        step_records: list[dict[str, Any]] = []

        logger.info(f"Starting episode: {task_instruction} (mode={self.mode})")

        while self._running and self._step < self._max_iterations:
            record = await self._step_once()
            step_records.append(record)

            if record.get("done"):
                break

            self._step += 1

        logger.info(f"Episode finished after {self._step + 1} steps")
        return step_records

    async def _step_once(self) -> dict[str, Any]:
        """Execute one observe→reason→act cycle."""
        self.memory.reset_working()

        # 1. Observe
        obs = self.env.get_observation()
        self.memory.working.current_image = obs.image

        # 2. Reason (CoT via VLM)
        task_context = ""
        if self.memory.task.subtask_history:
            last = self.memory.task.subtask_history[-1]
            task_context = f"Last subtask: [{last.status}] {last.subtask}"

        cot_prompt = COT_REASONING_PROMPT.format(
            task_instruction=self.memory.task.task_instruction,
            mode=self.mode,
            step_number=self._step,
            task_context=task_context,
        )

        messages = self.memory.to_messages()
        messages.append({"role": "user", "content": cot_prompt})

        response = await self.provider.chat(
            messages=messages,
            model=self.config.vlm.model,
            max_tokens=self.config.vlm.max_tokens,
            temperature=self.config.vlm.temperature,
        )

        # 3. Parse reasoning and tool selection
        reasoning, tool_name, tool_args = self._parse_response(response.content or "")
        self.memory.working.cot_reasoning = reasoning
        self.memory.working.tool_name = tool_name
        self.memory.working.tool_args = tool_args

        logger.debug(f"Step {self._step}: tool={tool_name}, reasoning={reasoning[:100]}...")

        # 4. Execute tool
        tool_result = await self._execute_tool(tool_name, tool_args)
        self.memory.working.tool_result = tool_result

        # 5. Update task memory
        self.memory.task.add_observation(
            f"Step {self._step}: {tool_name} → {tool_result[:200]}"
        )

        done = tool_name == "terminate" or "task_complete" in tool_result.lower()

        return {
            "step": self._step,
            "reasoning": reasoning,
            "tool": tool_name,
            "tool_args": tool_args,
            "tool_result": tool_result,
            "joint_positions": obs.joint_positions.tolist(),
            "gripper_open": obs.gripper_open,
            "done": done,
        }

    async def _execute_tool(self, tool_name: str, tool_args: dict[str, Any]) -> str:
        """Execute a tool by name."""
        if not tool_name:
            return "no_tool_selected"

        tool = self.tools.get(tool_name)
        if tool is None:
            return f"error: unknown tool '{tool_name}'"

        try:
            if hasattr(tool, "execute"):
                result = await tool.execute(**tool_args)
            elif callable(tool):
                result = await tool(**tool_args)
            else:
                result = f"error: tool '{tool_name}' is not callable"
            return str(result)
        except Exception as e:
            logger.error(f"Tool '{tool_name}' failed: {e}")
            return f"error: {e}"

    def _parse_response(self, text: str) -> tuple[str, str, dict[str, Any]]:
        """Parse VLM response to extract reasoning, tool name, and args."""
        reasoning = text
        tool_name = ""
        tool_args: dict[str, Any] = {}

        # Try to parse as JSON
        try:
            # Find JSON object in the response
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
                reasoning = data.get("reasoning", reasoning)
                tool_name = data.get("tool", "")
                tool_args = data.get("args", {})
        except (json.JSONDecodeError, KeyError):
            pass

        return reasoning, tool_name, tool_args

    def stop(self) -> None:
        """Signal the agent loop to stop."""
        self._running = False
