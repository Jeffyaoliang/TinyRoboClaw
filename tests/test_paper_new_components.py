"""Tests for new components: Skills, MCP Server, PolicyPool, Recycler, DemoImport, Integration."""

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from roboclaw.paper.config import PaperConfig, SimConfig, DeploymentConfig, EAPConfig, TrainerConfig
from roboclaw.paper.sim.tabletop_env import TabletopEnv
from roboclaw.paper.policy.mock_policy import MockPolicy
from roboclaw.paper.policy.manager import PolicyManager
from roboclaw.paper.policy.pool import PolicyPool, PolicyPair
from roboclaw.paper.eap.trajectory import TrajectoryDataset, Trajectory, TimeStep
from roboclaw.paper.flywheel.recycler import DeploymentRecycler
from roboclaw.paper.eap.demo_import import DemoImporter
from roboclaw.paper.skills.base import RoboticSkill, SkillResult, SkillStatus
from roboclaw.paper.skills.data_collection import DataCollectionSkill
from roboclaw.paper.skills.long_horizon import LongHorizonSkill
from roboclaw.paper.mcp.server import PaperMCPServer
from roboclaw.providers.base import LLMProvider, LLMResponse


class MockProvider(LLMProvider):
    async def chat(self, messages, **kwargs):
        return LLMResponse(content='{"success": true, "reason": "mock"}', finish_reason="stop")
    def get_default_model(self):
        return "mock"


# ── PolicyPool ─────────────────────────────────────────

class TestPolicyPool:
    def test_register_and_get(self):
        pool = PolicyPool()
        fwd = MockPolicy(name="fwd_pick")
        inv = MockPolicy(name="inv_pick")
        pool.register("pick lipstick", fwd, inv)

        pair = pool.get("pick lipstick")
        assert pair is not None
        assert pair.forward.name == "fwd_pick"
        assert pair.inverse.name == "inv_pick"

    def test_get_nonexistent(self):
        pool = PolicyPool()
        assert pool.get("nonexistent") is None

    def test_default_pair(self):
        pool = PolicyPool()
        fwd = MockPolicy(name="fwd")
        inv = MockPolicy(name="inv")
        pool.register("task_a", fwd, inv)
        pool.register("task_b", MockPolicy(name="fwd_b"), MockPolicy(name="inv_b"))

        # First registered is default
        pair = pool.get_or_default("unknown_task")
        assert pair.subtask == "task_a"

    def test_get_forward_inverse(self):
        pool = PolicyPool()
        pool.register("task", MockPolicy(name="f"), MockPolicy(name="i"))
        assert pool.get_forward("task").name == "f"
        assert pool.get_inverse("task").name == "i"

    def test_update_policy(self):
        pool = PolicyPool()
        pool.register("task", MockPolicy(name="old"), MockPolicy(name="inv"))
        pool.update_policy("task", "forward", MockPolicy(name="new"), checkpoint="/path")
        assert pool.get("task").forward.name == "new"
        assert pool.get("task").checkpoint_forward == "/path"

    def test_subtasks_and_size(self):
        pool = PolicyPool()
        pool.register("a", MockPolicy(), MockPolicy())
        pool.register("b", MockPolicy(), MockPolicy())
        assert pool.size == 2
        assert set(pool.subtasks) == {"a", "b"}

    def test_summary(self):
        pool = PolicyPool()
        pool.register("task", MockPolicy(name="f"), MockPolicy(name="i"))
        pool.update_metrics("task", {"success_rate": 0.8})
        summary = pool.summary()
        assert len(summary) == 1
        assert summary[0]["forward"] == "f"
        assert summary[0]["metrics"]["success_rate"] == 0.8


# ── DeploymentRecycler ─────────────────────────────────

class TestDeploymentRecycler:
    def _make_traj(self, task="t", direction="forward", success=True, n=3):
        t = Trajectory(task=task, direction=direction, success=success)
        for i in range(n):
            t.add_step(TimeStep(
                image=np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8),
                joint_positions=np.random.randn(6).astype(np.float32),
                gripper_open=0.5,
                action_joints=np.random.randn(6).astype(np.float32),
                action_gripper=0.5,
            ))
        return t

    def test_recycle_successful_forward(self, tmp_path):
        train_ds = TrajectoryDataset(tmp_path / "train")
        deploy_ds = TrajectoryDataset(tmp_path / "deploy")

        deploy_ds.add(self._make_traj(success=True, direction="forward"))
        deploy_ds.add(self._make_traj(success=False, direction="forward"))  # Filtered
        deploy_ds.add(self._make_traj(success=True, direction="inverse"))  # Filtered

        recycler = DeploymentRecycler(train_ds, deploy_ds, success_only=True, forward_only=True)
        count = recycler.recycle()

        assert count == 1
        assert train_ds.size == 1

    def test_recycle_all(self, tmp_path):
        train_ds = TrajectoryDataset(tmp_path / "train")
        deploy_ds = TrajectoryDataset(tmp_path / "deploy")

        deploy_ds.add(self._make_traj(success=True))
        deploy_ds.add(self._make_traj(success=False))
        deploy_ds.add(self._make_traj(success=True, direction="inverse"))

        recycler = DeploymentRecycler(train_ds, deploy_ds, success_only=False, forward_only=False)
        count = recycler.recycle()
        assert count == 3
        assert train_ds.size == 3

    def test_incremental_recycle(self, tmp_path):
        train_ds = TrajectoryDataset(tmp_path / "train")
        deploy_ds = TrajectoryDataset(tmp_path / "deploy")
        recycler = DeploymentRecycler(train_ds, deploy_ds, success_only=False, forward_only=False)

        deploy_ds.add(self._make_traj())
        assert recycler.recycle() == 1

        deploy_ds.add(self._make_traj())
        deploy_ds.add(self._make_traj())
        assert recycler.recycle() == 2
        assert train_ds.size == 3

    def test_pending_count(self, tmp_path):
        train_ds = TrajectoryDataset(tmp_path / "train")
        deploy_ds = TrajectoryDataset(tmp_path / "deploy")
        recycler = DeploymentRecycler(train_ds, deploy_ds)

        deploy_ds.add(self._make_traj())
        deploy_ds.add(self._make_traj())
        assert recycler.pending_count == 2

        recycler.recycle()
        assert recycler.pending_count == 0


# ── DemoImporter ───────────────────────────────────────

class TestDemoImporter:
    def test_import_npz(self, tmp_path):
        ds = TrajectoryDataset(tmp_path / "ds")
        importer = DemoImporter(ds)

        # Create demo npz
        T, J = 10, 6
        np.savez(
            tmp_path / "demo.npz",
            joint_positions=np.random.randn(T, J).astype(np.float32),
            images=np.random.randint(0, 255, (T, 32, 32, 3), dtype=np.uint8),
            gripper=np.full(T, 0.8, dtype=np.float32),
        )

        idx = importer.import_from_npz(tmp_path / "demo.npz", task="pick")
        assert idx == 0
        assert ds.size == 1

        meta = ds.get_meta(0)
        assert meta["task"] == "pick"
        assert meta["success"] is True
        assert meta["length"] == 10
        assert meta["metadata"]["source"] == "human_demo"

    def test_import_joint_trajectory(self, tmp_path):
        ds = TrajectoryDataset(tmp_path / "ds")
        importer = DemoImporter(ds)

        joints = np.random.randn(20, 6).astype(np.float32)
        idx = importer.import_from_joint_trajectory(joints, task="reach")
        assert idx == 0
        assert ds.size == 1
        assert ds.get_meta(0)["length"] == 20

    def test_import_directory(self, tmp_path):
        ds = TrajectoryDataset(tmp_path / "ds")
        importer = DemoImporter(ds)

        demo_dir = tmp_path / "demos"
        demo_dir.mkdir()
        for i in range(3):
            np.savez(
                demo_dir / f"demo_{i}.npz",
                joint_positions=np.random.randn(5, 6).astype(np.float32),
            )

        indices = importer.import_directory(demo_dir, task="test")
        assert len(indices) == 3
        assert ds.size == 3


# ── Skills ─────────────────────────────────────────────

class TestDataCollectionSkill:
    @pytest.fixture
    def tools(self):
        pm = MagicMock()
        pm.start_policy = AsyncMock(return_value="started:forward:mock")
        pm.stop_policy = AsyncMock()
        pm.switch_direction = AsyncMock()

        env = TabletopEnv(SimConfig(image_size=(8, 8), num_objects=1))

        from roboclaw.paper.tools.start_policy import StartPolicyTool
        from roboclaw.paper.tools.terminate_policy import TerminatePolicyTool
        from roboclaw.paper.tools.switch_policy import SwitchPolicyTool
        from roboclaw.paper.tools.fetch_robot_stats import FetchRobotStatsTool
        from roboclaw.paper.tools.call_human import CallHumanTool

        return {
            "start_policy": StartPolicyTool(pm),
            "terminate_policy": TerminatePolicyTool(pm),
            "switch_policy": SwitchPolicyTool(pm),
            "fetch_robot_stats": FetchRobotStatsTool(env),
            "call_human": CallHumanTool(callback=AsyncMock(return_value="ok")),
        }

    @pytest.mark.asyncio
    async def test_execute(self, tools):
        skill = DataCollectionSkill()
        assert skill.name == "data-collection"

        result = await skill.execute("pick up block", tools, num_episodes=2, max_steps=5)
        assert result.success
        assert len(result.subtask_results) == 2
        assert result.data["total_episodes"] == 2

    @pytest.mark.asyncio
    async def test_missing_tool(self):
        skill = DataCollectionSkill()
        result = await skill.execute("test", {}, num_episodes=1)
        # Should handle gracefully
        assert isinstance(result, SkillResult)


class TestLongHorizonSkill:
    @pytest.fixture
    def tools(self):
        pm = MagicMock()
        pm.start_policy = AsyncMock(return_value="started:forward:mock")
        pm.stop_policy = AsyncMock()
        pm.switch_direction = AsyncMock()

        env = TabletopEnv(SimConfig(image_size=(8, 8), num_objects=1))
        provider = MockProvider()

        from roboclaw.paper.tools.start_policy import StartPolicyTool
        from roboclaw.paper.tools.terminate_policy import TerminatePolicyTool
        from roboclaw.paper.tools.switch_policy import SwitchPolicyTool
        from roboclaw.paper.tools.env_summary import EnvSummaryTool
        from roboclaw.paper.tools.fetch_robot_stats import FetchRobotStatsTool
        from roboclaw.paper.tools.call_human import CallHumanTool

        return {
            "start_policy": StartPolicyTool(pm),
            "terminate_policy": TerminatePolicyTool(pm),
            "switch_policy": SwitchPolicyTool(pm),
            "env_summary": EnvSummaryTool(env, provider),
            "fetch_robot_stats": FetchRobotStatsTool(env),
            "call_human": CallHumanTool(callback=AsyncMock(return_value="ok")),
        }

    @pytest.mark.asyncio
    async def test_execute_with_subtasks(self, tools):
        skill = LongHorizonSkill()
        assert skill.name == "long-horizon-execution"

        result = await skill.execute(
            "organize table", tools,
            subtasks=["pick up block", "place on plate"],
        )
        assert result.success
        assert len(result.subtask_results) == 2
        assert result.data["total_subtasks"] == 2


# ── MCP Server ─────────────────────────────────────────

class TestPaperMCPServer:
    @pytest.fixture
    def server(self):
        env = TabletopEnv(SimConfig(image_size=(8, 8), num_objects=1))
        config = PaperConfig()
        fwd = MockPolicy(name="fwd")
        inv = MockPolicy(name="inv")
        pm = PolicyManager(config, env, fwd, inv)
        provider = MockProvider()
        return PaperMCPServer(env, pm, provider)

    def test_get_tools(self, server):
        tools = server.get_tools()
        assert "start_policy" in tools
        assert "terminate_policy" in tools
        assert "switch_policy" in tools
        assert "env_summary" in tools
        assert "fetch_robot_stats" in tools
        assert "call_human" in tools

    def test_get_tool_definitions(self, server):
        defs = server.get_tool_definitions()
        assert len(defs) == 6
        names = {d["name"] for d in defs}
        assert "start_policy" in names
        assert all("inputSchema" in d for d in defs)

    @pytest.mark.asyncio
    async def test_call_tool(self, server):
        result = await server.call_tool("fetch_robot_stats", {})
        data = json.loads(result)
        assert "joint_positions" in data

    @pytest.mark.asyncio
    async def test_call_unknown_tool(self, server):
        result = await server.call_tool("nonexistent", {})
        assert "error" in result.lower()

    def test_register_with_registry(self, server):
        from roboclaw.agent.tools.registry import ToolRegistry
        registry = ToolRegistry()
        server.register_with_tool_registry(registry)
        assert registry.has("start_policy")
        assert registry.has("fetch_robot_stats")


# ── Integration ────────────────────────────────────────

class TestIntegration:
    @pytest.mark.asyncio
    async def test_create_paper_agent(self):
        from roboclaw.paper.integration import create_paper_agent

        config = PaperConfig(sim=SimConfig(image_size=(8, 8)))
        env = TabletopEnv(config.sim)
        provider = MockProvider()
        fwd = MockPolicy(name="fwd")
        inv = MockPolicy(name="inv")
        pm = PolicyManager(config, env, fwd, inv)

        agent = create_paper_agent(config, provider, env, pm, mode="data_collection")
        assert agent is not None
        assert "start_policy" in agent.tools
        assert "skill_data_collection" in agent.tools
        assert "skill_long_horizon_execution" in agent.tools

    @pytest.mark.asyncio
    async def test_create_full_pipeline(self, tmp_path):
        from roboclaw.paper.integration import create_full_pipeline

        config = PaperConfig(
            sim=SimConfig(image_size=(8, 8)),
            eap=EAPConfig(episodes_per_batch=1, max_steps_per_episode=3),
        )
        config.task_name = "test"
        config.task_instruction = "pick"

        env = TabletopEnv(config.sim)
        provider = MockProvider()
        fwd = MockPolicy(name="fwd", chunk_size=1)
        inv = MockPolicy(name="inv", chunk_size=1)

        components = create_full_pipeline(
            config, env, provider, fwd, inv,
            data_dir=str(tmp_path / "pipeline_data"),
        )

        assert "agent" in components
        assert "flywheel" in components
        assert "recycler" in components
        assert "policy_pool" in components
        assert components["policy_pool"].size == 1

        # Test EAP through the pipeline
        result = await components["eap_engine"].run_batch(1)
        assert result.num_episodes == 1
        assert components["training_dataset"].size == 2
