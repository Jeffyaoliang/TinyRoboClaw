"""Tests for roboclaw.paper.eap (Trajectory, TrajectoryDataset, SuccessJudge, EAPEngine)."""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from roboclaw.paper.config import EAPConfig, PaperConfig, SimConfig
from roboclaw.paper.eap.engine import EAPEngine, EAPBatchResult
from roboclaw.paper.eap.judge import SuccessJudge
from roboclaw.paper.eap.trajectory import TimeStep, Trajectory, TrajectoryDataset
from roboclaw.paper.policy.mock_policy import MockPolicy
from roboclaw.paper.sim.tabletop_env import TabletopEnv
from roboclaw.providers.base import LLMResponse


# ── Trajectory ──────────────────────────────────────────────


class TestTimeStep:
    def test_to_dict(self):
        ts = TimeStep(
            image=np.zeros((32, 32, 3), dtype=np.uint8),
            joint_positions=np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0], dtype=np.float32),
            gripper_open=0.8,
            action_joints=np.array([0.2, 0.3, 0.4, 0.0, 0.0, 0.0], dtype=np.float32),
            action_gripper=0.5,
            timestamp=1.0,
        )
        d = ts.to_dict()
        assert d["gripper_open"] == 0.8
        assert d["timestamp"] == 1.0
        assert len(d["joint_positions"]) == 6
        assert "action_joints" in d
        assert d["action_gripper"] == 0.5

    def test_to_dict_no_action(self):
        ts = TimeStep(
            image=np.zeros((32, 32, 3), dtype=np.uint8),
            joint_positions=np.zeros(6, dtype=np.float32),
            gripper_open=1.0,
        )
        d = ts.to_dict()
        assert "action_joints" not in d
        assert "action_gripper" not in d


class TestTrajectory:
    @pytest.fixture
    def traj(self):
        t = Trajectory(task="pick", direction="forward")
        for i in range(5):
            t.add_step(TimeStep(
                image=np.full((8, 8, 3), i, dtype=np.uint8),
                joint_positions=np.full(6, i * 0.1, dtype=np.float32),
                gripper_open=float(i) / 5,
                action_joints=np.full(6, (i + 1) * 0.1, dtype=np.float32),
                action_gripper=0.5,
                timestamp=i * 0.05,
            ))
        return t

    def test_length(self, traj):
        assert traj.length == 5

    def test_get_images(self, traj):
        imgs = traj.get_images()
        assert imgs.shape == (5, 8, 8, 3)

    def test_get_joint_trajectory(self, traj):
        joints = traj.get_joint_trajectory()
        assert joints.shape == (5, 6)
        np.testing.assert_almost_equal(joints[0, 0], 0.0)
        np.testing.assert_almost_equal(joints[4, 0], 0.4)

    def test_get_action_trajectory(self, traj):
        actions = traj.get_action_trajectory()
        assert actions is not None
        assert actions.shape == (5, 6)

    def test_to_meta_dict(self, traj):
        traj.success = True
        meta = traj.to_meta_dict()
        assert meta["task"] == "pick"
        assert meta["direction"] == "forward"
        assert meta["success"] is True
        assert meta["length"] == 5
        assert len(meta["steps"]) == 5


# ── TrajectoryDataset ──────────────────────────────────────


class TestTrajectoryDataset:
    @pytest.fixture
    def dataset(self, tmp_path):
        return TrajectoryDataset(tmp_path / "dataset")

    def _make_traj(self, task="pick", direction="forward", success=True, n_steps=3):
        t = Trajectory(task=task, direction=direction, success=success)
        for i in range(n_steps):
            t.add_step(TimeStep(
                image=np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8),
                joint_positions=np.random.randn(6).astype(np.float32),
                gripper_open=0.5,
                action_joints=np.random.randn(6).astype(np.float32),
                action_gripper=0.5,
            ))
        return t

    def test_add_and_size(self, dataset):
        assert dataset.size == 0
        idx = dataset.add(self._make_traj())
        assert idx == 0
        assert dataset.size == 1
        dataset.add(self._make_traj())
        assert dataset.size == 2

    def test_get_meta(self, dataset):
        dataset.add(self._make_traj(task="test_task", direction="forward", success=True))
        meta = dataset.get_meta(0)
        assert meta["task"] == "test_task"
        assert meta["direction"] == "forward"
        assert meta["success"] is True
        assert meta["traj_id"] == "traj_0000"

    def test_get_meta_out_of_range(self, dataset):
        with pytest.raises(IndexError):
            dataset.get_meta(0)

    def test_load_images(self, dataset):
        traj = self._make_traj(n_steps=4)
        dataset.add(traj)
        images = dataset.load_images(0)
        assert images.shape == (4, 8, 8, 3)

    def test_filter_by_task(self, dataset):
        dataset.add(self._make_traj(task="A"))
        dataset.add(self._make_traj(task="B"))
        dataset.add(self._make_traj(task="A"))
        results = dataset.filter(task="A")
        assert len(results) == 2

    def test_filter_by_direction(self, dataset):
        dataset.add(self._make_traj(direction="forward"))
        dataset.add(self._make_traj(direction="inverse"))
        dataset.add(self._make_traj(direction="forward"))
        results = dataset.filter(direction="inverse")
        assert len(results) == 1

    def test_filter_by_success(self, dataset):
        dataset.add(self._make_traj(success=True))
        dataset.add(self._make_traj(success=False))
        dataset.add(self._make_traj(success=True))
        results = dataset.filter(success=True)
        assert len(results) == 2

    def test_filter_combined(self, dataset):
        dataset.add(self._make_traj(task="A", direction="forward", success=True))
        dataset.add(self._make_traj(task="A", direction="forward", success=False))
        dataset.add(self._make_traj(task="A", direction="inverse", success=True))
        results = dataset.filter(task="A", direction="forward", success=True)
        assert len(results) == 1

    def test_stats(self, dataset):
        dataset.add(self._make_traj(direction="forward", success=True))
        dataset.add(self._make_traj(direction="forward", success=False))
        dataset.add(self._make_traj(direction="inverse", success=True))
        stats = dataset.stats()
        assert stats["total"] == 3
        assert stats["forward"] == 2
        assert stats["inverse"] == 1
        assert stats["success_rate_forward"] == 0.5
        assert stats["success_rate_inverse"] == 1.0

    def test_stats_empty(self, dataset):
        stats = dataset.stats()
        assert stats["total"] == 0
        assert stats["success_rate_forward"] == 0.0

    def test_iter_all(self, dataset):
        dataset.add(self._make_traj())
        dataset.add(self._make_traj())
        items = list(dataset.iter_all())
        assert len(items) == 2

    def test_export_lerobot(self, dataset, tmp_path):
        dataset.add(self._make_traj(direction="forward", success=True, n_steps=5))
        dataset.add(self._make_traj(direction="forward", success=False, n_steps=3))  # excluded
        dataset.add(self._make_traj(direction="inverse", success=True, n_steps=4))  # excluded

        export_dir = tmp_path / "export"
        result = dataset.export_for_training(export_dir, format="lerobot")
        assert result == export_dir
        assert (export_dir / "dataset.npz").exists()
        assert (export_dir / "meta.json").exists()

        with open(export_dir / "meta.json") as f:
            meta = json.load(f)
        # Only 1 successful forward trajectory
        assert len(meta["episodes"]) == 1
        assert meta["total_steps"] == 5

    def test_export_raw(self, dataset, tmp_path):
        dataset.add(self._make_traj(n_steps=3))
        export_dir = tmp_path / "export_raw"
        result = dataset.export_for_training(export_dir, format="raw")
        assert result == export_dir
        assert (export_dir / "traj_0000_images.npz").exists()

    def test_persistence(self, tmp_path):
        """Dataset survives re-instantiation."""
        ds1 = TrajectoryDataset(tmp_path / "persist")
        ds1.add(self._make_traj())
        ds1.add(self._make_traj())
        assert ds1.size == 2

        ds2 = TrajectoryDataset(tmp_path / "persist")
        assert ds2.size == 2
        meta = ds2.get_meta(0)
        assert meta["traj_id"] == "traj_0000"


# ── SuccessJudge ───────────────────────────────────────────


class TestSuccessJudge:
    @pytest.fixture
    def mock_provider(self):
        provider = MagicMock()
        provider.chat = AsyncMock()
        return provider

    @pytest.mark.asyncio
    async def test_judge_success(self, mock_provider):
        mock_provider.chat.return_value = LLMResponse(
            content='{"success": true, "reason": "Object is in the target area"}',
            finish_reason="stop",
        )
        judge = SuccessJudge(mock_provider, model="test")
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        success, reason = await judge.judge(image, "pick up block", "forward")
        assert success is True
        assert "target area" in reason

    @pytest.mark.asyncio
    async def test_judge_failure(self, mock_provider):
        mock_provider.chat.return_value = LLMResponse(
            content='{"success": false, "reason": "Block is still on the table"}',
            finish_reason="stop",
        )
        judge = SuccessJudge(mock_provider)
        success, reason = await judge.judge(np.zeros((32, 32, 3), dtype=np.uint8), "pick up", "forward")
        assert success is False
        assert "table" in reason

    @pytest.mark.asyncio
    async def test_judge_malformed_response(self, mock_provider):
        mock_provider.chat.return_value = LLMResponse(
            content="I'm not sure what happened",
            finish_reason="stop",
        )
        judge = SuccessJudge(mock_provider)
        success, reason = await judge.judge(np.zeros((32, 32, 3), dtype=np.uint8), "test", "forward")
        assert success is False  # Fallback: no "success" keyword without "not"

    @pytest.mark.asyncio
    async def test_judge_provider_error(self, mock_provider):
        mock_provider.chat = AsyncMock(side_effect=RuntimeError("API error"))
        judge = SuccessJudge(mock_provider)
        success, reason = await judge.judge(np.zeros((32, 32, 3), dtype=np.uint8), "test", "forward")
        assert success is False
        assert "judge_error" in reason


# ── EAPEngine ──────────────────────────────────────────────


class TestEAPEngine:
    @pytest.fixture
    def setup(self, tmp_path):
        config = PaperConfig(
            eap=EAPConfig(episodes_per_batch=2, max_steps_per_episode=3),
            sim=SimConfig(image_size=(8, 8), num_objects=1),
        )
        config.task_name = "test"
        config.task_instruction = "pick up"

        env = TabletopEnv(config.sim)
        fwd = MockPolicy(name="fwd", chunk_size=1)
        inv = MockPolicy(name="inv", chunk_size=1, seed=42)

        provider = MagicMock()
        provider.chat = AsyncMock(return_value=LLMResponse(
            content='{"success": true, "reason": "done"}', finish_reason="stop"
        ))
        judge = SuccessJudge(provider)
        dataset = TrajectoryDataset(tmp_path / "eap_data")

        engine = EAPEngine(config, env, fwd, inv, judge, dataset)
        return engine, dataset

    @pytest.mark.asyncio
    async def test_run_batch(self, setup):
        engine, dataset = setup
        result = await engine.run_batch(2)

        assert isinstance(result, EAPBatchResult)
        assert result.num_episodes == 2
        # Each episode produces 2 trajectories (forward + inverse)
        assert dataset.size == 4
        assert result.total_steps > 0

    @pytest.mark.asyncio
    async def test_batch_result_metrics(self, setup):
        engine, _ = setup
        result = await engine.run_batch(3)
        assert result.num_episodes == 3
        # Mock judge always returns success
        assert result.success_rate_forward == 1.0
        assert result.success_rate_inverse == 1.0

    @pytest.mark.asyncio
    async def test_trajectories_stored(self, setup):
        engine, dataset = setup
        await engine.run_batch(1)

        # Should have 1 forward + 1 inverse
        fwd = dataset.filter(direction="forward")
        inv = dataset.filter(direction="inverse")
        assert len(fwd) == 1
        assert len(inv) == 1
        assert fwd[0]["task"] == "test"
        assert inv[0]["task"] == "test"

    @pytest.mark.asyncio
    async def test_trajectory_length(self, setup):
        engine, dataset = setup
        await engine.run_batch(1)

        meta = dataset.get_meta(0)
        # max_steps_per_episode = 3
        assert meta["length"] == 3

    @pytest.mark.asyncio
    async def test_judge_reason_stored(self, setup):
        engine, dataset = setup
        await engine.run_batch(1)
        meta = dataset.get_meta(0)
        assert meta["metadata"]["judge_reason"] == "done"
