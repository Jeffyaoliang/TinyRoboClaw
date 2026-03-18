"""End-to-end tests: CLI → Agent → EAP → Flywheel → Deployment 全链路."""

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from roboclaw.paper.config import (
    DeploymentConfig,
    EAPConfig,
    FlywheelConfig,
    PaperConfig,
    SimConfig,
    TrainerConfig,
)
from roboclaw.paper.sim.tabletop_env import TabletopEnv
from roboclaw.paper.policy.mock_policy import MockPolicy
from roboclaw.paper.policy.manager import PolicyManager
from roboclaw.paper.policy.trainer import PolicyTrainer
from roboclaw.paper.eap.engine import EAPEngine
from roboclaw.paper.eap.judge import SuccessJudge
from roboclaw.paper.eap.trajectory import TrajectoryDataset
from roboclaw.paper.flywheel.flywheel import DataFlywheel
from roboclaw.paper.deployment.supervisor import DeploymentSupervisor
from roboclaw.paper.deployment.executor import LongHorizonExecutor
from roboclaw.paper.agent.loop import VLMAgentLoop
from roboclaw.paper.tools.start_policy import StartPolicyTool
from roboclaw.paper.tools.terminate_policy import TerminatePolicyTool
from roboclaw.paper.tools.switch_policy import SwitchPolicyTool
from roboclaw.paper.tools.env_summary import EnvSummaryTool
from roboclaw.paper.tools.fetch_robot_stats import FetchRobotStatsTool
from roboclaw.paper.tools.call_human import CallHumanTool
from roboclaw.providers.base import LLMProvider, LLMResponse


# ── Shared fixtures ──────────────────────────────────────

class MockLLMProvider(LLMProvider):
    """Mock provider that returns configurable responses."""

    def __init__(self):
        super().__init__()
        self._responses: list[LLMResponse] = []
        self._default = LLMResponse(
            content='{"success": true, "reason": "mock"}',
            finish_reason="stop",
        )
        self._call_count = 0

    def set_responses(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)

    async def chat(self, messages, **kwargs) -> LLMResponse:
        self._call_count += 1
        if self._responses:
            return self._responses.pop(0)
        return self._default

    def get_default_model(self) -> str:
        return "mock"


@pytest.fixture
def tmp_data_dir(tmp_path):
    return tmp_path


@pytest.fixture
def mock_provider():
    return MockLLMProvider()


@pytest.fixture
def config(tmp_data_dir):
    return PaperConfig(
        eap=EAPConfig(episodes_per_batch=2, max_steps_per_episode=5),
        sim=SimConfig(image_size=(8, 8), num_objects=2),
        flywheel=FlywheelConfig(
            num_iterations=2,
            episodes_per_iteration=2,
            data_dir=str(tmp_data_dir / "flywheel_data"),
        ),
        trainer=TrainerConfig(output_dir=str(tmp_data_dir / "checkpoints")),
        deployment=DeploymentConfig(monitor_interval=0.01, max_retries=1),
        task_name="e2e_test",
        task_instruction="pick up the red block",
        inverse_instruction="put back the red block",
    )


@pytest.fixture
def env(config):
    return TabletopEnv(config.sim)


@pytest.fixture
def policies():
    fwd = MockPolicy(name="fwd", mode="reach", chunk_size=1)
    inv = MockPolicy(name="inv", mode="random", chunk_size=1, seed=99)
    return fwd, inv


# ── E2E Test 1: EAP 单轮采集 → 数据验证 ──────────────

class TestE2EEAP:
    """EAP engine collects trajectories and stores them correctly."""

    @pytest.mark.asyncio
    async def test_eap_collect_and_verify_data(self, config, env, policies, mock_provider, tmp_data_dir):
        fwd, inv = policies
        dataset = TrajectoryDataset(tmp_data_dir / "eap_data")
        judge = SuccessJudge(mock_provider)

        engine = EAPEngine(config, env, fwd, inv, judge, dataset)
        result = await engine.run_batch(3)

        # Verify batch result
        assert result.num_episodes == 3
        assert result.total_steps > 0
        assert result.success_rate_forward == 1.0  # mock judge

        # Verify dataset
        assert dataset.size == 6  # 3 forward + 3 inverse
        stats = dataset.stats()
        assert stats["forward"] == 3
        assert stats["inverse"] == 3

        # Verify trajectory content
        meta = dataset.get_meta(0)
        assert meta["task"] == "e2e_test"
        assert meta["direction"] == "forward"
        assert meta["length"] == 5  # max_steps_per_episode
        assert len(meta["steps"]) == 5

        # Verify images stored
        images = dataset.load_images(0)
        assert images.shape == (5, 8, 8, 3)

        # Verify export works
        export_dir = tmp_data_dir / "export"
        dataset.export_for_training(export_dir, format="lerobot")
        assert (export_dir / "dataset.npz").exists()
        assert (export_dir / "meta.json").exists()

        with open(export_dir / "meta.json") as f:
            export_meta = json.load(f)
        assert export_meta["total_steps"] > 0


# ── E2E Test 2: 完整飞轮 → 多轮迭代 → 数据累积 ──────

class TestE2EFlywheel:
    """Full flywheel: EAP → accumulate → export → mock train → repeat."""

    @pytest.mark.asyncio
    async def test_flywheel_full_loop(self, config, env, policies, mock_provider, tmp_data_dir):
        fwd, inv = policies
        dataset = TrajectoryDataset(tmp_data_dir / "flywheel_ds")
        judge = SuccessJudge(mock_provider)
        pm = PolicyManager(config, env, fwd, inv)
        trainer = PolicyTrainer(config.trainer)

        engine = EAPEngine(config, env, fwd, inv, judge, dataset)
        flywheel = DataFlywheel(config, engine, dataset, trainer, pm)

        results = await flywheel.run(
            num_iterations=2,
            episodes_per_iter=2,
            mock_training=True,
        )

        # Two iterations completed
        assert len(results) == 2

        # Data accumulated across iterations
        assert results[0].dataset_size == 4   # 2 eps × 2 (fwd+inv)
        assert results[1].dataset_size == 8   # accumulated

        # Checkpoints created
        assert results[0].checkpoint_path is not None
        assert results[1].checkpoint_path is not None
        assert Path(results[0].checkpoint_path).exists()
        assert Path(results[1].checkpoint_path).exists()

        # Metrics tracked
        assert results[0].metrics["success_rate_forward"] == 1.0
        assert results[1].metrics["total"] == 8

        # Export dirs created
        data_dir = Path(config.flywheel.data_dir)
        assert (data_dir / "train_iter_00").exists()
        assert (data_dir / "train_iter_01").exists()

        # Verify flywheel results property
        assert flywheel.results == results


# ── E2E Test 3: Agent 循环 → 工具调用 → 策略执行 ─────

class TestE2EAgentLoop:
    """VLM Agent loop with tools controlling policy execution."""

    @pytest.mark.asyncio
    async def test_agent_tool_chain(self, config, env, policies, mock_provider):
        fwd, inv = policies
        pm = PolicyManager(config, env, fwd, inv)

        # EnvSummaryTool calls provider.chat internally, so use a separate provider
        env_summary_provider = MockLLMProvider()

        # Create tools
        tools = {
            "start_policy": StartPolicyTool(pm),
            "terminate_policy": TerminatePolicyTool(pm),
            "switch_policy": SwitchPolicyTool(pm),
            "env_summary": EnvSummaryTool(env, env_summary_provider),
            "fetch_robot_stats": FetchRobotStatsTool(env),
        }

        # Agent responses: observe → start policy → observe → terminate
        mock_provider.set_responses([
            LLMResponse(
                content='{"reasoning": "Let me check the scene", "tool": "env_summary", "args": {}}',
                finish_reason="stop",
            ),
            LLMResponse(
                content='{"reasoning": "Scene looks good, start forward policy", "tool": "start_policy", "args": {"instruction": "pick up block", "direction": "forward", "max_steps": 3}}',
                finish_reason="stop",
            ),
            LLMResponse(
                content='{"reasoning": "Check robot state", "tool": "fetch_robot_stats", "args": {}}',
                finish_reason="stop",
            ),
            LLMResponse(
                content='{"reasoning": "Done, terminate", "tool": "terminate", "args": {}}',
                finish_reason="stop",
            ),
        ])

        loop = VLMAgentLoop(config, mock_provider, env, tools=tools, mode="data_collection")
        records = await loop.run_episode("pick up the red block")

        # Verify execution
        assert len(records) == 4

        # Step 1: env_summary → got description
        assert records[0]["tool"] == "env_summary"
        assert records[0]["tool_result"] != ""

        # Step 2: start_policy → policy started
        assert records[1]["tool"] == "start_policy"
        assert "started" in records[1]["tool_result"].lower() or "Policy started" in records[1]["tool_result"]

        # Step 3: fetch_robot_stats → got JSON stats
        assert records[2]["tool"] == "fetch_robot_stats"
        stats = json.loads(records[2]["tool_result"])
        assert "joint_positions" in stats
        assert "gripper_open" in stats

        # Step 4: terminate → done
        assert records[3]["tool"] == "terminate"
        assert records[3]["done"] is True

        # Memory was updated
        assert loop.memory.task.task_instruction == "pick up the red block"
        assert len(loop.memory.task.observation_log) == 4

        # Cleanup: stop any running policy
        if pm.is_running:
            await pm.stop_policy()


# ── E2E Test 4: 部署 → 任务分解 → 监督执行 ──────────

class TestE2EDeployment:
    """Long-horizon deployment: decompose → supervise subtasks → handle results."""

    @pytest.mark.asyncio
    async def test_deployment_full_flow(self, config, env, policies, mock_provider, tmp_data_dir):
        fwd, inv = policies
        pm = PolicyManager(config, env, fwd, inv)
        judge = SuccessJudge(mock_provider)
        dataset = TrajectoryDataset(tmp_data_dir / "deploy_data")

        supervisor = DeploymentSupervisor(config.deployment, env, pm, judge, dataset)
        executor = LongHorizonExecutor(config, mock_provider, env, supervisor)

        # Mock provider: first call = decomposition, rest = success judgments
        call_count = 0

        async def staged_responses(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return LLMResponse(
                    content='["pick up the red block", "place on the plate"]',
                    finish_reason="stop",
                )
            return LLMResponse(
                content='{"success": true, "reason": "completed"}',
                finish_reason="stop",
            )

        mock_provider.chat = AsyncMock(side_effect=staged_responses)

        result = await executor.execute("organize the table")

        # Task decomposed and executed
        assert result.task == "organize the table"
        assert len(result.subtasks) == 2
        assert result.subtasks[0].subtask == "pick up the red block"
        assert result.subtasks[1].subtask == "place on the plate"

        # Deployment data collected
        assert dataset.size >= 1


# ── E2E Test 5: 全链路 CLI → Flywheel → 数据 ────────

class TestE2ECLIFlow:
    """Simulate CLI collect command flow end-to-end."""

    @pytest.mark.asyncio
    async def test_cli_collect_flow(self, config, tmp_data_dir):
        """Simulate what `python -m roboclaw.paper.cli collect --mock` does."""

        env = TabletopEnv(config.sim)
        provider = MockLLMProvider()

        fwd = MockPolicy(name="fwd", mode="random", chunk_size=1)
        inv = MockPolicy(name="inv", mode="random", chunk_size=1, seed=42)

        dataset = TrajectoryDataset(tmp_data_dir / "cli_data")
        judge = SuccessJudge(provider)
        pm = PolicyManager(config, env, fwd, inv)
        trainer = PolicyTrainer(config.trainer)

        engine = EAPEngine(config, env, fwd, inv, judge, dataset)
        flywheel = DataFlywheel(config, engine, dataset, trainer, pm)

        # Run flywheel (same as CLI collect --mock)
        results = await flywheel.run(
            num_iterations=2,
            episodes_per_iter=3,
            mock_training=True,
        )

        # Verify end state
        assert len(results) == 2
        assert dataset.size == 12  # 2 iters × 3 eps × 2 (fwd+inv)

        # Verify data integrity: every trajectory has correct structure
        for meta in dataset.iter_all():
            assert "task" in meta
            assert "direction" in meta
            assert "success" in meta
            assert "length" in meta
            assert "steps" in meta
            assert meta["task"] == "e2e_test"
            assert meta["direction"] in ("forward", "inverse")
            assert meta["length"] == 5

        # Verify export
        export_dir = tmp_data_dir / "final_export"
        dataset.export_for_training(export_dir, format="lerobot")
        data = np.load(export_dir / "dataset.npz")
        assert "observations" in data
        assert "actions" in data
        assert "episode_ids" in data
        assert len(data["observations"]) > 0


# ── E2E Test 6: 策略切换 + 监督恢复 ─────────────────

class TestE2EPolicySwitching:
    """Test forward/inverse policy switching during supervised execution."""

    @pytest.mark.asyncio
    async def test_policy_switch_and_recovery(self, config, env, policies, mock_provider, tmp_data_dir):
        fwd, inv = policies
        pm = PolicyManager(config, env, fwd, inv)

        # Verify initial state
        assert pm.active_direction == "forward"
        assert pm.active_policy.name == "fwd"

        # Start forward policy
        result = await pm.start_policy("pick up", direction="forward", max_steps=3)
        assert "started" in result
        # Wait for completion
        import asyncio
        await asyncio.sleep(0.2)
        if pm.is_running:
            await pm.stop_policy()

        # Switch to inverse
        await pm.switch_direction("inverse")
        assert pm.active_direction == "inverse"
        assert pm.active_policy.name == "inv"

        # Start inverse policy
        result = await pm.start_policy("put back", direction="inverse", max_steps=3)
        assert "started" in result
        await asyncio.sleep(0.2)
        if pm.is_running:
            await pm.stop_policy()

        # Switch back
        await pm.switch_direction("forward")
        assert pm.active_direction == "forward"

        # Reload policy
        new_fwd = MockPolicy(name="fwd_v2", mode="reach", chunk_size=1)
        await pm.reload_policy("forward", new_fwd)
        assert pm.active_policy.name == "fwd_v2"


# ── E2E Test 7: 数据持久化 + 重新加载 ───────────────

class TestE2EDataPersistence:
    """Verify data survives across dataset instances and is exportable."""

    @pytest.mark.asyncio
    async def test_data_persistence_across_sessions(self, config, env, policies, mock_provider, tmp_data_dir):
        fwd, inv = policies
        data_path = tmp_data_dir / "persist_data"

        # Session 1: collect data
        ds1 = TrajectoryDataset(data_path)
        judge = SuccessJudge(mock_provider)
        engine = EAPEngine(config, env, fwd, inv, judge, ds1)
        await engine.run_batch(2)
        assert ds1.size == 4

        # Session 2: re-open and verify
        ds2 = TrajectoryDataset(data_path)
        assert ds2.size == 4

        # Session 2: add more data
        engine2 = EAPEngine(config, env, fwd, inv, judge, ds2)
        await engine2.run_batch(1)
        assert ds2.size == 6

        # Session 3: verify accumulated data
        ds3 = TrajectoryDataset(data_path)
        assert ds3.size == 6

        stats = ds3.stats()
        assert stats["total"] == 6
        assert stats["forward"] == 3
        assert stats["inverse"] == 3

        # Verify all images loadable
        for i in range(6):
            images = ds3.load_images(i)
            assert images.ndim == 4  # (T, H, W, 3)
            assert images.shape[0] == 5  # max_steps_per_episode
