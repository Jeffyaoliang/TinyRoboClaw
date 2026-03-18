"""Tests for roboclaw.paper.tools (6 MCP tools)."""

import json
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from roboclaw.paper.sim.tabletop_env import TabletopEnv
from roboclaw.paper.config import SimConfig
from roboclaw.paper.tools.start_policy import StartPolicyTool
from roboclaw.paper.tools.terminate_policy import TerminatePolicyTool
from roboclaw.paper.tools.switch_policy import SwitchPolicyTool
from roboclaw.paper.tools.env_summary import EnvSummaryTool
from roboclaw.paper.tools.fetch_robot_stats import FetchRobotStatsTool
from roboclaw.paper.tools.call_human import CallHumanTool
from roboclaw.providers.base import LLMResponse


@pytest.fixture
def env():
    return TabletopEnv(SimConfig(num_objects=2, image_size=(32, 32)))


@pytest.fixture
def mock_policy_manager():
    pm = MagicMock()
    pm.start_policy = AsyncMock(return_value="started:forward:mock")
    pm.stop_policy = AsyncMock()
    pm.switch_direction = AsyncMock()
    return pm


@pytest.fixture
def mock_provider():
    provider = MagicMock()
    provider.chat = AsyncMock(
        return_value=LLMResponse(content="The scene shows objects on a table.", finish_reason="stop")
    )
    return provider


class TestStartPolicyTool:
    def test_schema(self, mock_policy_manager):
        tool = StartPolicyTool(mock_policy_manager)
        assert tool.name == "start_policy"
        assert "instruction" in tool.parameters["properties"]
        schema = tool.to_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "start_policy"

    @pytest.mark.asyncio
    async def test_execute(self, mock_policy_manager):
        tool = StartPolicyTool(mock_policy_manager)
        result = await tool.execute(instruction="pick up block", direction="forward", max_steps=50)
        assert "started" in result.lower() or "Policy started" in result
        mock_policy_manager.start_policy.assert_awaited_once_with(
            instruction="pick up block", direction="forward", max_steps=50
        )

    @pytest.mark.asyncio
    async def test_execute_default_direction(self, mock_policy_manager):
        tool = StartPolicyTool(mock_policy_manager)
        await tool.execute(instruction="test")
        mock_policy_manager.start_policy.assert_awaited_once_with(
            instruction="test", direction="forward", max_steps=200
        )

    @pytest.mark.asyncio
    async def test_execute_failure(self, mock_policy_manager):
        mock_policy_manager.start_policy = AsyncMock(side_effect=RuntimeError("connection lost"))
        tool = StartPolicyTool(mock_policy_manager)
        result = await tool.execute(instruction="test")
        assert "Failed" in result


class TestTerminatePolicyTool:
    def test_schema(self, mock_policy_manager):
        tool = TerminatePolicyTool(mock_policy_manager)
        assert tool.name == "terminate_policy"

    @pytest.mark.asyncio
    async def test_execute(self, mock_policy_manager):
        tool = TerminatePolicyTool(mock_policy_manager)
        result = await tool.execute(reason="stuck")
        assert "terminated" in result.lower()
        mock_policy_manager.stop_policy.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_execute_no_reason(self, mock_policy_manager):
        tool = TerminatePolicyTool(mock_policy_manager)
        result = await tool.execute()
        assert "agent_decision" in result


class TestSwitchPolicyTool:
    def test_schema(self, mock_policy_manager):
        tool = SwitchPolicyTool(mock_policy_manager)
        assert tool.name == "switch_policy"
        assert "direction" in tool.parameters["properties"]
        assert tool.parameters["properties"]["direction"]["enum"] == ["forward", "inverse"]

    @pytest.mark.asyncio
    async def test_execute(self, mock_policy_manager):
        tool = SwitchPolicyTool(mock_policy_manager)
        result = await tool.execute(direction="inverse")
        assert "inverse" in result.lower()
        mock_policy_manager.switch_direction.assert_awaited_once_with("inverse")


class TestEnvSummaryTool:
    def test_schema(self, env, mock_provider):
        tool = EnvSummaryTool(env, mock_provider)
        assert tool.name == "env_summary"

    @pytest.mark.asyncio
    async def test_execute(self, env, mock_provider):
        tool = EnvSummaryTool(env, mock_provider, model="test-model")
        result = await tool.execute()
        assert "objects on a table" in result
        mock_provider.chat.assert_awaited_once()
        # Verify image was sent
        call_args = mock_provider.chat.call_args
        messages = call_args.kwargs.get("messages") or call_args[1].get("messages") or call_args[0][0]
        content = messages[0]["content"]
        assert any(p["type"] == "image_url" for p in content)


class TestFetchRobotStatsTool:
    def test_schema(self, env):
        tool = FetchRobotStatsTool(env)
        assert tool.name == "fetch_robot_stats"

    @pytest.mark.asyncio
    async def test_execute(self, env):
        env.reset()
        tool = FetchRobotStatsTool(env)
        result = await tool.execute()
        data = json.loads(result)
        assert "joint_positions" in data
        assert "gripper_open" in data
        assert "objects" in data
        assert len(data["joint_positions"]) == 6
        assert len(data["objects"]) == 2


class TestCallHumanTool:
    def test_schema(self):
        tool = CallHumanTool()
        assert tool.name == "call_human"
        assert "message" in tool.parameters["properties"]

    @pytest.mark.asyncio
    async def test_execute_with_callback(self):
        callback = AsyncMock(return_value="I fixed it")
        tool = CallHumanTool(callback=callback)
        result = await tool.execute(message="Robot is stuck")
        assert "I fixed it" in result
        callback.assert_awaited_once_with("Robot is stuck")

    @pytest.mark.asyncio
    async def test_execute_no_interactive(self):
        # Without callback and with no stdin, should handle gracefully
        tool = CallHumanTool(callback=None)
        # Patch stdin to simulate EOF
        import io
        import sys
        from unittest.mock import patch

        with patch("builtins.input", side_effect=EOFError):
            result = await tool.execute(message="help")
        assert "no interactive input" in result.lower() or "help" in result.lower()
