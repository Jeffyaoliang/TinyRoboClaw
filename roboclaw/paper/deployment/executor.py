"""Long-horizon task executor (paper Sec 3.3).

VLM decomposes task → executes subtasks sequentially → monitors → handles failures.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from roboclaw.paper.agent.memory import _encode_image
from roboclaw.paper.agent.prompts import TASK_DECOMPOSITION_PROMPT
from roboclaw.paper.config import PaperConfig
from roboclaw.paper.deployment.supervisor import DeploymentSupervisor
from roboclaw.paper.sim.base_env import BaseEnvironment
from roboclaw.providers.base import LLMProvider


@dataclass
class SubtaskResult:
    """Result of executing a single subtask."""

    subtask: str
    success: bool
    reason: str = ""


@dataclass
class ExecutionResult:
    """Result of executing a full long-horizon task."""

    task: str
    subtasks: list[SubtaskResult] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return all(s.success for s in self.subtasks)

    @property
    def num_completed(self) -> int:
        return sum(1 for s in self.subtasks if s.success)


class LongHorizonExecutor:
    """Executes long-horizon tasks by decomposing into subtasks.

    Flow:
    1. VLM decomposes task into subtasks
    2. For each subtask: execute with supervisor
    3. On failure: retry or recover
    4. Return overall result
    """

    def __init__(
        self,
        config: PaperConfig,
        provider: LLMProvider,
        env: BaseEnvironment,
        supervisor: DeploymentSupervisor,
    ):
        self.config = config
        self.provider = provider
        self.env = env
        self.supervisor = supervisor

    async def execute(self, task_instruction: str) -> ExecutionResult:
        """Execute a long-horizon task."""
        logger.info(f"Executing long-horizon task: {task_instruction}")
        result = ExecutionResult(task=task_instruction)

        # 1. Decompose task into subtasks
        subtasks = await self._decompose_task(task_instruction)
        logger.info(f"Decomposed into {len(subtasks)} subtasks: {subtasks}")

        # 2. Execute each subtask
        for i, subtask in enumerate(subtasks):
            logger.info(f"Executing subtask {i + 1}/{len(subtasks)}: {subtask}")

            success, reason = await self.supervisor.supervise_subtask(
                subtask=subtask,
                instruction=subtask,
            )

            sub_result = SubtaskResult(subtask=subtask, success=success, reason=reason)
            result.subtasks.append(sub_result)

            if not success:
                logger.warning(f"Subtask failed: {subtask} ({reason})")
                # Try to continue with remaining subtasks
                # (some failures may be recoverable)
                continue

            logger.info(f"Subtask completed: {subtask}")

        completed = result.num_completed
        total = len(result.subtasks)
        logger.info(
            f"Task execution finished: {completed}/{total} subtasks completed"
        )
        return result

    async def _decompose_task(self, task_instruction: str) -> list[str]:
        """Use VLM to decompose a task into ordered subtasks."""
        # Get current environment description
        image = self.env.capture_image()
        img_b64 = _encode_image(image)

        prompt = TASK_DECOMPOSITION_PROMPT.format(
            task_instruction=task_instruction,
            env_description="(see image)",
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        response = await self.provider.chat(
            messages=messages,
            model=self.config.vlm.model,
            max_tokens=1024,
            temperature=0.3,
        )

        return self._parse_subtasks(response.content or "")

    def _parse_subtasks(self, text: str) -> list[str]:
        """Parse VLM response to extract subtask list."""
        try:
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                subtasks = json.loads(text[start:end])
                if isinstance(subtasks, list) and all(isinstance(s, str) for s in subtasks):
                    return subtasks
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: split by newlines/numbers
        lines = text.strip().split("\n")
        subtasks = []
        for line in lines:
            line = line.strip().lstrip("0123456789.-) ")
            if line:
                subtasks.append(line)
        return subtasks if subtasks else [text.strip()]
