"""EnvSummary tool: VLM-based environment scene description."""

from __future__ import annotations

from typing import Any

from roboclaw.agent.tools.base import Tool
from roboclaw.paper.agent.memory import _encode_image
from roboclaw.paper.agent.prompts import ENV_SUMMARY_PROMPT


class EnvSummaryTool(Tool):
    """Use VLM to describe the current environment scene from camera images."""

    def __init__(self, env: Any, provider: Any, model: str = "gpt-4o") -> None:
        self._env = env
        self._provider = provider
        self._model = model

    @property
    def name(self) -> str:
        return "env_summary"

    @property
    def description(self) -> str:
        return "Get a VLM-generated description of the current robot workspace scene."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

    async def execute(self, **kwargs: Any) -> str:
        try:
            image = self._env.capture_image()
            img_b64 = _encode_image(image)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                        },
                        {"type": "text", "text": ENV_SUMMARY_PROMPT},
                    ],
                }
            ]

            response = await self._provider.chat(
                messages=messages,
                model=self._model,
                max_tokens=512,
                temperature=0.3,
            )
            return response.content or "No description generated."
        except Exception as e:
            return f"Failed to generate environment summary: {e}"
