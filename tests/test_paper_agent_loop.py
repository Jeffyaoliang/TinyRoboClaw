"""Tests for roboclaw.paper.agent.loop (VLMAgentLoop)."""

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from roboclaw.paper.agent.loop import VLMAgentLoop
from roboclaw.paper.config import PaperConfig, SimConfig, VLMConfig
from roboclaw.paper.sim.tabletop_env import TabletopEnv
from roboclaw.providers.base import LLMResponse


@pytest.fixture
def mock_provider():
    provider = MagicMock()
    provider.chat = AsyncMock()
    return provider


@pytest.fixture
def env():
    return TabletopEnv(SimConfig(image_size=(8, 8), num_objects=1))


class TestVLMAgentLoop:
    def test_init_data_collection_mode(self, mock_provider, env):
        config = PaperConfig(vlm=VLMConfig(max_iterations=5))
        loop = VLMAgentLoop(config, mock_provider, env, mode="data_collection")
        assert loop.mode == "data_collection"
        assert "data collection" in loop.memory.role.system_prompt.lower() or \
               "training data" in loop.memory.role.system_prompt.lower()

    def test_init_deployment_mode(self, mock_provider, env):
        config = PaperConfig()
        loop = VLMAgentLoop(config, mock_provider, env, mode="deployment")
        assert loop.mode == "deployment"
        assert "long-horizon" in loop.memory.role.system_prompt.lower() or \
               "supervising" in loop.memory.role.system_prompt.lower()

    def test_tool_descriptions_in_memory(self, mock_provider, env):
        mock_tool = MagicMock()
        mock_tool.description = "A test tool"
        config = PaperConfig()
        loop = VLMAgentLoop(config, mock_provider, env, tools={"test_tool": mock_tool})
        assert len(loop.memory.role.tool_descriptions) == 1
        assert loop.memory.role.tool_descriptions[0]["name"] == "test_tool"

    @pytest.mark.asyncio
    async def test_run_episode_basic(self, mock_provider, env):
        config = PaperConfig(vlm=VLMConfig(max_iterations=3))
        loop = VLMAgentLoop(config, mock_provider, env)

        # VLM returns a "no tool" response each time
        mock_provider.chat.return_value = LLMResponse(
            content='{"reasoning": "nothing to do", "tool": "", "args": {}}',
            finish_reason="stop",
        )

        records = await loop.run_episode("pick up block")
        assert len(records) == 3  # max_iterations
        assert records[0]["step"] == 0
        assert records[0]["tool"] == ""

    @pytest.mark.asyncio
    async def test_run_episode_with_tool(self, mock_provider, env):
        config = PaperConfig(vlm=VLMConfig(max_iterations=5))
        mock_tool = AsyncMock(return_value="tool_executed_ok")
        mock_tool.execute = AsyncMock(return_value="tool_executed_ok")

        loop = VLMAgentLoop(config, mock_provider, env, tools={"my_tool": mock_tool})

        # First call: use tool. Second call: terminate
        responses = [
            LLMResponse(
                content='{"reasoning": "use tool", "tool": "my_tool", "args": {"x": 1}}',
                finish_reason="stop",
            ),
            LLMResponse(
                content='{"reasoning": "done", "tool": "terminate", "args": {}}',
                finish_reason="stop",
            ),
        ]
        mock_provider.chat.side_effect = responses

        records = await loop.run_episode("test")
        assert len(records) == 2
        assert records[0]["tool"] == "my_tool"
        assert records[0]["tool_result"] == "tool_executed_ok"
        assert records[1]["done"] is True

    @pytest.mark.asyncio
    async def test_run_episode_unknown_tool(self, mock_provider, env):
        config = PaperConfig(vlm=VLMConfig(max_iterations=2))
        loop = VLMAgentLoop(config, mock_provider, env)

        mock_provider.chat.return_value = LLMResponse(
            content='{"reasoning": "try", "tool": "nonexistent", "args": {}}',
            finish_reason="stop",
        )

        records = await loop.run_episode("test")
        assert "unknown tool" in records[0]["tool_result"]

    @pytest.mark.asyncio
    async def test_stop(self, mock_provider, env):
        config = PaperConfig(vlm=VLMConfig(max_iterations=100))
        loop = VLMAgentLoop(config, mock_provider, env)

        call_count = 0

        async def stop_on_third_call(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count >= 3:
                loop.stop()
            return LLMResponse(
                content='{"reasoning": "wait", "tool": "", "args": {}}',
                finish_reason="stop",
            )

        mock_provider.chat = AsyncMock(side_effect=stop_on_third_call)
        records = await loop.run_episode("test")
        # Should have stopped at step 3 (after stop() called)
        assert len(records) == 3

    def test_parse_response_valid_json(self, mock_provider, env):
        config = PaperConfig()
        loop = VLMAgentLoop(config, mock_provider, env)

        reasoning, tool, args = loop._parse_response(
            '{"reasoning": "I see a block", "tool": "pick", "args": {"obj": "red"}}'
        )
        assert reasoning == "I see a block"
        assert tool == "pick"
        assert args == {"obj": "red"}

    def test_parse_response_invalid_json(self, mock_provider, env):
        config = PaperConfig()
        loop = VLMAgentLoop(config, mock_provider, env)

        reasoning, tool, args = loop._parse_response("Some free text with no JSON")
        assert reasoning == "Some free text with no JSON"
        assert tool == ""
        assert args == {}

    def test_parse_response_json_embedded_in_text(self, mock_provider, env):
        config = PaperConfig()
        loop = VLMAgentLoop(config, mock_provider, env)

        text = 'Let me think...\n{"reasoning": "found it", "tool": "grab", "args": {}}\nDone.'
        reasoning, tool, args = loop._parse_response(text)
        assert reasoning == "found it"
        assert tool == "grab"

    @pytest.mark.asyncio
    async def test_memory_updated_per_step(self, mock_provider, env):
        config = PaperConfig(vlm=VLMConfig(max_iterations=2))
        loop = VLMAgentLoop(config, mock_provider, env)

        mock_provider.chat.return_value = LLMResponse(
            content='{"reasoning": "step", "tool": "", "args": {}}',
            finish_reason="stop",
        )

        await loop.run_episode("test task")
        # Task memory should have observations
        assert len(loop.memory.task.observation_log) == 2
        assert loop.memory.task.task_instruction == "test task"
