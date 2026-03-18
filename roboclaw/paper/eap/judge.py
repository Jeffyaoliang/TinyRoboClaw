"""VLM-based success/failure judge (paper Sec 3.2)."""

from __future__ import annotations

import json
from typing import Any

from loguru import logger

from roboclaw.paper.agent.memory import _encode_image
from roboclaw.paper.agent.prompts import JUDGE_SUCCESS_PROMPT
from roboclaw.providers.base import LLMProvider


class SuccessJudge:
    """Uses a VLM to judge whether a task/reset was completed successfully."""

    def __init__(self, provider: LLMProvider, model: str = "gpt-4o"):
        self._provider = provider
        self._model = model

    async def judge(
        self,
        image: Any,
        instruction: str,
        direction: str = "forward",
    ) -> tuple[bool, str]:
        """Judge whether the task was completed successfully.

        Args:
            image: Current camera image (numpy array).
            instruction: The task instruction.
            direction: "forward" or "inverse".

        Returns:
            (success, reason) tuple.
        """
        import numpy as np

        img_b64 = _encode_image(np.asarray(image))

        prompt = JUDGE_SUCCESS_PROMPT.format(
            instruction=instruction,
            direction=direction,
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

        try:
            response = await self._provider.chat(
                messages=messages,
                model=self._model,
                max_tokens=256,
                temperature=0.1,
            )

            text = response.content or ""
            return self._parse_judgment(text)
        except Exception as e:
            logger.error(f"Judge failed: {e}")
            return False, f"judge_error: {e}"

    def _parse_judgment(self, text: str) -> tuple[bool, str]:
        """Parse VLM response to extract success/reason."""
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
                return bool(data.get("success", False)), data.get("reason", "")
        except (json.JSONDecodeError, KeyError):
            pass

        # Fallback: keyword-based parsing
        text_lower = text.lower()
        if "success" in text_lower and "not" not in text_lower:
            return True, text[:200]
        return False, text[:200]
