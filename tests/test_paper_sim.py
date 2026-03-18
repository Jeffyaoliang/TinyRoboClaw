"""Tests for roboclaw.paper.sim (BaseEnvironment + TabletopEnv)."""

import numpy as np
import pytest

from roboclaw.paper.config import SimConfig
from roboclaw.paper.sim.base_env import ActionChunk, Observation
from roboclaw.paper.sim.tabletop_env import TabletopEnv


class TestObservation:
    def test_fields(self):
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        joints = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)
        obs = Observation(image=img, joint_positions=joints, gripper_open=0.5, timestamp=1.0)
        assert obs.image.shape == (224, 224, 3)
        assert obs.joint_positions.shape == (6,)
        assert obs.gripper_open == 0.5
        assert obs.timestamp == 1.0


class TestActionChunk:
    def test_chunk_size(self):
        joints = np.zeros((10, 6), dtype=np.float32)
        gripper = np.zeros(10, dtype=np.float32)
        chunk = ActionChunk(joint_targets=joints, gripper_targets=gripper)
        assert chunk.chunk_size == 10


class TestTabletopEnv:
    @pytest.fixture
    def env(self):
        return TabletopEnv(SimConfig(num_objects=3, image_size=(64, 64)))

    def test_reset_returns_observation(self, env):
        obs = env.reset()
        assert isinstance(obs, Observation)
        assert obs.image.shape == (64, 64, 3)
        assert obs.joint_positions.shape == (6,)
        assert obs.gripper_open == 1.0
        assert obs.timestamp == 0.0

    def test_reset_resets_joints(self, env):
        # Move joints
        action = ActionChunk(
            joint_targets=np.ones((1, 6), dtype=np.float32),
            gripper_targets=np.array([0.0], dtype=np.float32),
        )
        env.step(action)
        assert not np.allclose(env.get_joint_positions(), 0.0)

        # Reset
        obs = env.reset()
        assert np.allclose(obs.joint_positions, 0.0)
        assert obs.gripper_open == 1.0

    def test_step_updates_joints(self, env):
        env.reset()
        target = np.full((1, 6), 0.5, dtype=np.float32)
        action = ActionChunk(joint_targets=target, gripper_targets=np.array([0.5], dtype=np.float32))
        obs, info = env.step(action)

        assert info["steps_executed"] == 1
        # Joints should have moved toward 0.5 (proportional control: 0.5 * target)
        assert np.all(obs.joint_positions > 0.0)
        assert np.all(obs.joint_positions < 0.5)

    def test_multi_step_action_chunk(self, env):
        env.reset()
        chunk_size = 5
        target = np.full((chunk_size, 6), 1.0, dtype=np.float32)
        gripper = np.full(chunk_size, 0.0, dtype=np.float32)
        action = ActionChunk(joint_targets=target, gripper_targets=gripper)

        obs, info = env.step(action)
        assert info["steps_executed"] == chunk_size
        # After 5 steps of moving toward 1.0, joints should be closer to 1.0
        assert np.all(obs.joint_positions > 0.3)

    def test_capture_image(self, env):
        env.reset()
        img = env.capture_image()
        assert img.shape == (64, 64, 3)
        assert img.dtype == np.uint8
        # Should not be all white (objects are drawn)
        assert not np.all(img == 255)

    def test_get_object_positions(self, env):
        env.reset()
        positions = env.get_object_positions()
        assert len(positions) == 3
        for name, pos in positions.items():
            assert name.startswith("object_")
            assert pos.shape == (3,)
            # Objects should be within workspace bounds
            assert np.all(pos >= 0.0)
            assert np.all(pos <= 0.5)

    def test_get_gripper_state(self, env):
        env.reset()
        assert env.get_gripper_state() == 1.0

        # Close gripper
        action = ActionChunk(
            joint_targets=np.zeros((1, 6), dtype=np.float32),
            gripper_targets=np.array([0.0], dtype=np.float32),
        )
        env.step(action)
        assert env.get_gripper_state() < 1.0

    def test_get_observation_no_step(self, env):
        env.reset()
        obs1 = env.get_observation()
        obs2 = env.get_observation()
        np.testing.assert_array_equal(obs1.joint_positions, obs2.joint_positions)
        assert obs1.gripper_open == obs2.gripper_open

    def test_deterministic_reset(self):
        """Same seed → same initial object positions."""
        env1 = TabletopEnv(SimConfig(num_objects=3))
        env2 = TabletopEnv(SimConfig(num_objects=3))
        obs1 = env1.reset()
        obs2 = env2.reset()
        pos1 = env1.get_object_positions()
        pos2 = env2.get_object_positions()
        for name in pos1:
            np.testing.assert_array_almost_equal(pos1[name], pos2[name])
