"""Robotic Skill base class — paper's Skills layer (Sec 3.1).

Skills are reusable procedures that orchestrate Tools to accomplish workflows.
Hierarchy: Skills → Tools → Policies

A Skill encapsulates a complete workflow (e.g., "data-collection" or
"long-horizon-execution") and can be invoked by the VLM agent as a
single high-level action.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from loguru import logger


class SkillStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    ABORTED = "aborted"


@dataclass
class SkillResult:
    """Result of executing a skill."""

    status: SkillStatus = SkillStatus.PENDING
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    subtask_results: list[dict[str, Any]] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.status == SkillStatus.SUCCESS


class RoboticSkill(ABC):
    """Abstract base for robotic skills.

    A skill orchestrates tools to accomplish a complex workflow.
    It receives a dict of tool instances and uses them to execute steps.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Skill name (e.g., 'data-collection')."""

    @property
    @abstractmethod
    def description(self) -> str:
        """What this skill does."""

    @abstractmethod
    async def execute(
        self,
        instruction: str,
        tools: dict[str, Any],
        **kwargs: Any,
    ) -> SkillResult:
        """Execute the skill.

        Args:
            instruction: Task instruction from the user/agent.
            tools: Dict of tool instances (name → Tool).
            **kwargs: Skill-specific parameters.

        Returns:
            SkillResult with status and data.
        """

    def to_tool_schema(self) -> dict[str, Any]:
        """Convert skill to a tool schema for the VLM agent."""
        return {
            "type": "function",
            "function": {
                "name": f"skill_{self.name.replace('-', '_')}",
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "instruction": {
                            "type": "string",
                            "description": "Task instruction to execute.",
                        },
                    },
                    "required": ["instruction"],
                },
            },
        }
