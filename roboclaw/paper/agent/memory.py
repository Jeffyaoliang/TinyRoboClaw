"""Structured memory for the VLM agent (paper Sec 3.1).

Three-layer memory:
- RoleIdentity (r_t): System prompt + available tool descriptions
- TaskMemory (g_t): Task goal + subtask history + observation records
- WorkingMemory (w_t): Current frame + CoT reasoning + tool results (reset each step)
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SubtaskRecord:
    """Record of a completed subtask."""

    subtask: str
    status: str  # "success" | "failed" | "skipped"
    observations: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class RoleIdentity:
    """r_t: System-level identity and capabilities."""

    system_prompt: str = ""
    tool_descriptions: list[dict[str, str]] = field(default_factory=list)

    def to_messages(self) -> list[dict[str, Any]]:
        tools_text = ""
        if self.tool_descriptions:
            tools_text = "\n\nAvailable tools:\n"
            for td in self.tool_descriptions:
                tools_text += f"- {td['name']}: {td['description']}\n"
        return [{"role": "system", "content": self.system_prompt + tools_text}]


@dataclass
class TaskMemory:
    """g_t: Task-level persistent memory."""

    task_instruction: str = ""
    subtask_history: list[SubtaskRecord] = field(default_factory=list)
    observation_log: list[str] = field(default_factory=list)

    def to_messages(self) -> list[dict[str, Any]]:
        parts = [f"Task: {self.task_instruction}"]

        if self.subtask_history:
            parts.append("\nCompleted subtasks:")
            for rec in self.subtask_history:
                parts.append(f"  [{rec.status}] {rec.subtask}")

        if self.observation_log:
            parts.append(f"\nRecent observations ({len(self.observation_log)}):")
            # Keep only last 5 observations to avoid context bloat
            for obs in self.observation_log[-5:]:
                parts.append(f"  - {obs}")

        return [{"role": "user", "content": "\n".join(parts)}]

    def add_observation(self, text: str) -> None:
        self.observation_log.append(text)

    def add_subtask_result(self, subtask: str, status: str, observations: list[str] | None = None) -> None:
        self.subtask_history.append(
            SubtaskRecord(subtask=subtask, status=status, observations=observations or [])
        )


@dataclass
class WorkingMemory:
    """w_t: Per-step working memory (reset each reasoning step)."""

    current_image: np.ndarray | None = None
    cot_reasoning: str = ""
    tool_name: str = ""
    tool_args: dict[str, Any] = field(default_factory=dict)
    tool_result: str = ""

    def to_messages(self) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []

        if self.current_image is not None:
            img_b64 = _encode_image(self.current_image)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64}"},
            })

        text_parts = []
        if self.cot_reasoning:
            text_parts.append(f"Reasoning: {self.cot_reasoning}")
        if self.tool_name:
            text_parts.append(f"Tool call: {self.tool_name}({self.tool_args})")
        if self.tool_result:
            text_parts.append(f"Result: {self.tool_result}")

        if text_parts:
            content.append({"type": "text", "text": "\n".join(text_parts)})

        if not content:
            return []
        return [{"role": "user", "content": content}]

    def reset(self) -> None:
        self.current_image = None
        self.cot_reasoning = ""
        self.tool_name = ""
        self.tool_args = {}
        self.tool_result = ""


class StructuredMemory:
    """Composite memory: r_t + g_t + w_t → message list for VLM."""

    def __init__(self) -> None:
        self.role = RoleIdentity()
        self.task = TaskMemory()
        self.working = WorkingMemory()

    def to_messages(self) -> list[dict[str, Any]]:
        msgs: list[dict[str, Any]] = []
        msgs.extend(self.role.to_messages())
        msgs.extend(self.task.to_messages())
        msgs.extend(self.working.to_messages())
        return msgs

    def reset_working(self) -> None:
        self.working.reset()

    def reset_all(self) -> None:
        self.role = RoleIdentity()
        self.task = TaskMemory()
        self.working = WorkingMemory()


def _encode_image(img: np.ndarray) -> str:
    """Encode numpy image to base64 PNG string."""
    try:
        from PIL import Image
        import io

        pil_img = Image.fromarray(img)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    except ImportError:
        # Fallback: raw bytes (won't render but won't crash)
        return base64.b64encode(img.tobytes()[:1000]).decode()
