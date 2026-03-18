"""Tests for roboclaw.paper.flywheel.flywheel (DataFlywheel)."""

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from roboclaw.paper.config import EAPConfig, PaperConfig, SimConfig, TrainerConfig
from roboclaw.paper.eap.engine import EAPEngine, EAPBatchResult
from roboclaw.paper.eap.judge import SuccessJudge
from roboclaw.paper.eap.trajectory import TrajectoryDataset
from roboclaw.paper.flywheel.flywheel import DataFlywheel, FlywheelIterationResult
from roboclaw.paper.policy.manager import PolicyManager
from roboclaw.paper.policy.mock_policy import MockPolicy
from roboclaw.paper.policy.trainer import PolicyTrainer
from roboclaw.paper.sim.tabletop_env import TabletopEnv
from roboclaw.providers.base import LLMResponse


@pytest.fixture
def flywheel_setup(tmp_path):
    config = PaperConfig(
        eap=EAPConfig(episodes_per_batch=2, max_steps_per_episode=3),
        sim=SimConfig(image_size=(8, 8), num_objects=1),
        trainer=TrainerConfig(output_dir=str(tmp_path / "checkpoints")),
    )
    config.task_name = "test_flywheel"
    config.task_instruction = "pick up"
    config.flywheel.data_dir = str(tmp_path / "data")

    env = TabletopEnv(config.sim)
    fwd = MockPolicy(name="fwd", chunk_size=1)
    inv = MockPolicy(name="inv", chunk_size=1, seed=42)

    provider = MagicMock()
    provider.chat = AsyncMock(return_value=LLMResponse(
        content='{"success": true, "reason": "ok"}', finish_reason="stop"
    ))
    judge = SuccessJudge(provider)

    dataset = TrajectoryDataset(tmp_path / "dataset")
    policy_manager = PolicyManager(config, env, fwd, inv)
    trainer = PolicyTrainer(config.trainer)

    engine = EAPEngine(config, env, fwd, inv, judge, dataset)

    flywheel = DataFlywheel(config, engine, dataset, trainer, policy_manager)
    return flywheel, dataset, config


class TestDataFlywheel:
    @pytest.mark.asyncio
    async def test_run_single_iteration(self, flywheel_setup):
        flywheel, dataset, _ = flywheel_setup
        results = await flywheel.run(num_iterations=1, episodes_per_iter=2, mock_training=True)

        assert len(results) == 1
        r = results[0]
        assert isinstance(r, FlywheelIterationResult)
        assert r.iteration == 0
        assert r.batch_result.num_episodes == 2
        assert r.dataset_size == 4  # 2 episodes × 2 (fwd+inv)
        assert r.checkpoint_path is not None

    @pytest.mark.asyncio
    async def test_run_multiple_iterations(self, flywheel_setup):
        flywheel, dataset, _ = flywheel_setup
        results = await flywheel.run(num_iterations=3, episodes_per_iter=2, mock_training=True)

        assert len(results) == 3
        # Dataset accumulates across iterations
        assert results[0].dataset_size == 4
        assert results[1].dataset_size == 8
        assert results[2].dataset_size == 12

        # Iteration indices
        assert [r.iteration for r in results] == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_metrics_recorded(self, flywheel_setup):
        flywheel, _, _ = flywheel_setup
        results = await flywheel.run(num_iterations=1, episodes_per_iter=2, mock_training=True)

        metrics = results[0].metrics
        assert "success_rate_forward" in metrics
        assert "success_rate_inverse" in metrics
        assert "total_steps" in metrics
        assert "total" in metrics
        assert metrics["success_rate_forward"] == 1.0  # mock judge always succeeds

    @pytest.mark.asyncio
    async def test_results_property(self, flywheel_setup):
        flywheel, _, _ = flywheel_setup
        assert flywheel.results == []

        await flywheel.run(num_iterations=2, episodes_per_iter=1, mock_training=True)
        assert len(flywheel.results) == 2

    @pytest.mark.asyncio
    async def test_checkpoint_created(self, flywheel_setup, tmp_path):
        flywheel, _, config = flywheel_setup
        results = await flywheel.run(num_iterations=1, episodes_per_iter=1, mock_training=True)

        ckpt_path = results[0].checkpoint_path
        assert ckpt_path is not None
        from pathlib import Path
        assert (Path(ckpt_path) / "checkpoint.pt").exists()

    @pytest.mark.asyncio
    async def test_training_failure_continues(self, flywheel_setup):
        flywheel, _, _ = flywheel_setup
        # Replace trainer with one that fails
        flywheel.trainer.train = AsyncMock(side_effect=RuntimeError("train crash"))

        # Should not raise, but checkpoint_path should be None
        results = await flywheel.run(num_iterations=1, episodes_per_iter=1, mock_training=False)
        assert results[0].checkpoint_path is None

    @pytest.mark.asyncio
    async def test_export_dirs_created(self, flywheel_setup, tmp_path):
        flywheel, _, config = flywheel_setup
        await flywheel.run(num_iterations=2, episodes_per_iter=1, mock_training=True)

        from pathlib import Path
        data_dir = Path(config.flywheel.data_dir)
        assert (data_dir / "train_iter_00").exists()
        assert (data_dir / "train_iter_01").exists()
