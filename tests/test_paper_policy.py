"""Tests for roboclaw.paper.policy (MockPolicy + PolicyManager)."""

import asyncio

import numpy as np
import pytest

from roboclaw.paper.config import PaperConfig, SimConfig
from roboclaw.paper.policy.mock_policy import MockPolicy
from roboclaw.paper.policy.manager import PolicyManager
from roboclaw.paper.sim.base_env import ActionChunk, Observation
from roboclaw.paper.sim.tabletop_env import TabletopEnv


class TestMockPolicy:
    @pytest.fixture
    def obs(self):
        return Observation(
            image=np.zeros((64, 64, 3), dtype=np.uint8),
            joint_positions=np.array([0.1, -0.2, 0.3, 0.0, 0.1, -0.1], dtype=np.float32),
            gripper_open=0.5,
        )

    @pytest.mark.asyncio
    async def test_random_mode(self, obs):
        policy = MockPolicy(name="test_random", mode="random", chunk_size=10)
        action = await policy.infer(obs, "pick up object")
        assert isinstance(action, ActionChunk)
        assert action.joint_targets.shape == (10, 6)
        assert action.gripper_targets.shape == (10,)

    @pytest.mark.asyncio
    async def test_reach_mode(self, obs):
        policy = MockPolicy(name="test_reach", mode="reach", chunk_size=5)
        action = await policy.infer(obs, "go home")
        assert action.joint_targets.shape == (5, 6)
        # Reach mode moves toward zero → targets should be closer to zero than obs
        mean_action = np.mean(np.abs(action.joint_targets), axis=0)
        mean_obs = np.abs(obs.joint_positions)
        assert np.all(mean_action < mean_obs + 0.05)  # allow for noise

    def test_name_property(self):
        policy = MockPolicy(name="my_policy")
        assert policy.name == "my_policy"

    def test_is_ready(self):
        policy = MockPolicy()
        assert policy.is_ready is True

    def test_reset(self):
        policy = MockPolicy()
        policy._step = 10
        policy.reset()
        assert policy._step == 0

    @pytest.mark.asyncio
    async def test_deterministic_with_seed(self, obs):
        p1 = MockPolicy(seed=42, chunk_size=5)
        p2 = MockPolicy(seed=42, chunk_size=5)
        a1 = await p1.infer(obs, "test")
        a2 = await p2.infer(obs, "test")
        np.testing.assert_array_equal(a1.joint_targets, a2.joint_targets)

    @pytest.mark.asyncio
    async def test_different_seeds_differ(self, obs):
        p1 = MockPolicy(seed=0, chunk_size=5)
        p2 = MockPolicy(seed=99, chunk_size=5)
        a1 = await p1.infer(obs, "test")
        a2 = await p2.infer(obs, "test")
        assert not np.array_equal(a1.joint_targets, a2.joint_targets)


class TestPolicyManager:
    @pytest.fixture
    def setup(self):
        config = PaperConfig(sim=SimConfig(image_size=(32, 32)))
        env = TabletopEnv(config.sim)
        fwd = MockPolicy(name="fwd", mode="random", chunk_size=5)
        inv = MockPolicy(name="inv", mode="random", chunk_size=5, seed=42)
        manager = PolicyManager(config, env, fwd, inv)
        return manager, env, fwd, inv

    def test_active_policy_default(self, setup):
        manager, _, fwd, _ = setup
        assert manager.active_direction == "forward"
        assert manager.active_policy is fwd

    def test_set_policy(self, setup):
        manager, _, _, _ = setup
        new_policy = MockPolicy(name="new_fwd")
        manager.set_policy("forward", new_policy)
        assert manager.active_policy is new_policy

    @pytest.mark.asyncio
    async def test_start_and_stop_policy(self, setup):
        manager, _, _, _ = setup
        result = await manager.start_policy("pick up", direction="forward", max_steps=5)
        assert "started" in result
        assert manager.is_running

        # Wait for execution to finish (max_steps=5, very quick)
        await asyncio.sleep(0.1)
        # It may have finished by itself; if not, stop it
        if manager.is_running:
            await manager.stop_policy()
        assert not manager.is_running

    @pytest.mark.asyncio
    async def test_start_policy_already_running(self, setup):
        manager, _, _, _ = setup
        await manager.start_policy("pick up", max_steps=1000)
        result = await manager.start_policy("another task")
        assert "error" in result
        await manager.stop_policy()

    @pytest.mark.asyncio
    async def test_start_policy_no_policy(self, setup):
        manager, _, _, _ = setup
        manager._policies.clear()
        result = await manager.start_policy("pick up", direction="forward")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_switch_direction(self, setup):
        manager, _, _, inv = setup
        await manager.switch_direction("inverse")
        assert manager.active_direction == "inverse"
        assert manager.active_policy is inv

    @pytest.mark.asyncio
    async def test_reload_policy(self, setup):
        manager, _, _, _ = setup
        new_policy = MockPolicy(name="reloaded")
        await manager.reload_policy("forward", new_policy)
        assert manager.active_policy is new_policy
        assert manager.active_policy.name == "reloaded"
