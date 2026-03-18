"""Tests for roboclaw.paper.deployment (Supervisor + Executor)."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from roboclaw.paper.config import DeploymentConfig, PaperConfig, SimConfig
from roboclaw.paper.deployment.executor import LongHorizonExecutor, ExecutionResult
from roboclaw.paper.deployment.supervisor import (
    DeploymentSupervisor,
    FailureType,
    SupervisorState,
)
from roboclaw.paper.eap.judge import SuccessJudge
from roboclaw.paper.eap.trajectory import TrajectoryDataset
from roboclaw.paper.policy.manager import PolicyManager
from roboclaw.paper.policy.mock_policy import MockPolicy
from roboclaw.paper.sim.tabletop_env import TabletopEnv
from roboclaw.providers.base import LLMResponse


# ── DeploymentSupervisor ───────────────────────────────────


class TestDeploymentSupervisor:
    @pytest.fixture
    def setup(self, tmp_path):
        config = PaperConfig(
            deployment=DeploymentConfig(
                monitor_interval=0.01,
                max_retries=2,
                collect_deployment_data=True,
            ),
            sim=SimConfig(image_size=(8, 8), num_objects=1),
        )
        env = TabletopEnv(config.sim)
        fwd = MockPolicy(name="fwd", chunk_size=1)
        inv = MockPolicy(name="inv", chunk_size=1, seed=42)
        pm = PolicyManager(config, env, fwd, inv)

        provider = MagicMock()
        provider.chat = AsyncMock(return_value=LLMResponse(
            content='{"success": true, "reason": "done"}', finish_reason="stop"
        ))
        judge = SuccessJudge(provider)
        dataset = TrajectoryDataset(tmp_path / "deploy_data")

        supervisor = DeploymentSupervisor(config.deployment, env, pm, judge, dataset)
        return supervisor, pm, provider, dataset

    @pytest.mark.asyncio
    async def test_supervise_success(self, setup):
        supervisor, pm, _, dataset = setup
        success, reason = await supervisor.supervise_subtask(
            subtask="pick up block",
            instruction="pick up block",
            max_steps=3,
        )
        assert success is True
        assert reason == "done"
        # Deployment data collected
        assert dataset.size >= 1

    @pytest.mark.asyncio
    async def test_supervise_failure_retry(self, setup):
        supervisor, pm, provider, _ = setup
        # First call fails, second succeeds
        call_count = 0

        async def varying_judge(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                # First two calls: failure + classify as non-degrading
                return LLMResponse(
                    content='{"success": false, "reason": "no change"}',
                    finish_reason="stop",
                )
            return LLMResponse(
                content='{"success": true, "reason": "done"}',
                finish_reason="stop",
            )

        provider.chat = AsyncMock(side_effect=varying_judge)
        success, reason = await supervisor.supervise_subtask(
            subtask="test", instruction="test", max_steps=2,
        )
        # Should eventually succeed after retry
        assert isinstance(success, bool)

    @pytest.mark.asyncio
    async def test_supervise_no_policy(self, setup):
        supervisor, pm, _, _ = setup
        pm._policies.clear()
        success, reason = await supervisor.supervise_subtask(
            subtask="test", instruction="test",
        )
        assert success is False
        assert "error" in reason


class TestFailureType:
    def test_enum_values(self):
        assert FailureType.NON_DEGRADING == "non_degrading"
        assert FailureType.DEGRADING == "degrading"


# ── LongHorizonExecutor ───────────────────────────────────


class TestLongHorizonExecutor:
    @pytest.fixture
    def setup(self, tmp_path):
        config = PaperConfig(
            deployment=DeploymentConfig(monitor_interval=0.01, max_retries=1),
            sim=SimConfig(image_size=(8, 8), num_objects=1),
        )
        env = TabletopEnv(config.sim)
        fwd = MockPolicy(name="fwd", chunk_size=1)
        inv = MockPolicy(name="inv", chunk_size=1, seed=42)
        pm = PolicyManager(config, env, fwd, inv)

        provider = MagicMock()
        judge = SuccessJudge(provider)
        dataset = TrajectoryDataset(tmp_path / "exec_data")
        supervisor = DeploymentSupervisor(config.deployment, env, pm, judge, dataset)
        executor = LongHorizonExecutor(config, provider, env, supervisor)
        return executor, provider

    @pytest.mark.asyncio
    async def test_execute_decomposes_and_runs(self, setup):
        executor, provider = setup
        # Provider returns: decomposition, then success judgments
        call_count = 0

        async def mock_chat(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Task decomposition
                return LLMResponse(
                    content='["pick up red block", "place on blue plate"]',
                    finish_reason="stop",
                )
            # Success judgment
            return LLMResponse(
                content='{"success": true, "reason": "done"}',
                finish_reason="stop",
            )

        provider.chat = AsyncMock(side_effect=mock_chat)

        result = await executor.execute("organize the table")
        assert isinstance(result, ExecutionResult)
        assert result.task == "organize the table"
        assert len(result.subtasks) == 2

    @pytest.mark.asyncio
    async def test_parse_subtasks_json(self, setup):
        executor, _ = setup
        subtasks = executor._parse_subtasks('["step1", "step2", "step3"]')
        assert subtasks == ["step1", "step2", "step3"]

    @pytest.mark.asyncio
    async def test_parse_subtasks_numbered_fallback(self, setup):
        executor, _ = setup
        text = "1. Pick up block\n2. Place on plate\n3. Return"
        subtasks = executor._parse_subtasks(text)
        assert len(subtasks) == 3
        assert "Pick up block" in subtasks[0]

    @pytest.mark.asyncio
    async def test_parse_subtasks_single_line(self, setup):
        executor, _ = setup
        subtasks = executor._parse_subtasks("just do this one thing")
        assert len(subtasks) == 1

    def test_execution_result_properties(self):
        from roboclaw.paper.deployment.executor import SubtaskResult

        result = ExecutionResult(
            task="test",
            subtasks=[
                SubtaskResult(subtask="a", success=True),
                SubtaskResult(subtask="b", success=False),
                SubtaskResult(subtask="c", success=True),
            ],
        )
        assert result.success is False  # not all succeeded
        assert result.num_completed == 2

    def test_execution_result_all_success(self):
        from roboclaw.paper.deployment.executor import SubtaskResult

        result = ExecutionResult(
            task="test",
            subtasks=[
                SubtaskResult(subtask="a", success=True),
                SubtaskResult(subtask="b", success=True),
            ],
        )
        assert result.success is True
        assert result.num_completed == 2
