"""Command-line interface for the paper implementation.

Usage:
    python -m roboclaw.paper.cli collect --task lipstick --iterations 5
    python -m roboclaw.paper.cli deploy --task "organize vanity table"
    python -m roboclaw.paper.cli train --task lipstick --data-dir data/paper
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from loguru import logger


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RoboClaw Paper Implementation CLI",
        prog="python -m roboclaw.paper.cli",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- collect ---
    collect_parser = subparsers.add_parser("collect", help="Run EAP data collection flywheel")
    collect_parser.add_argument("--task", required=True, help="Task name or config file")
    collect_parser.add_argument("--iterations", type=int, default=5, help="Number of flywheel iterations")
    collect_parser.add_argument("--episodes", type=int, default=50, help="Episodes per iteration")
    collect_parser.add_argument("--data-dir", default="data/paper", help="Data directory")
    collect_parser.add_argument("--mock", action="store_true", help="Use mock policies (no real VLA)")
    collect_parser.add_argument("--config", help="Path to YAML config file")

    # --- deploy ---
    deploy_parser = subparsers.add_parser("deploy", help="Deploy for long-horizon task execution")
    deploy_parser.add_argument("--task", required=True, help="Task instruction")
    deploy_parser.add_argument("--config", help="Path to YAML config file")
    deploy_parser.add_argument("--mock", action="store_true", help="Use mock policies")

    # --- train ---
    train_parser = subparsers.add_parser("train", help="Train/retrain policy with collected data")
    train_parser.add_argument("--task", required=True, help="Task name")
    train_parser.add_argument("--data-dir", required=True, help="Training data directory")
    train_parser.add_argument("--config", help="Path to YAML config file")
    train_parser.add_argument("--mock", action="store_true", help="Mock training (no actual training)")

    args = parser.parse_args()

    if args.command == "collect":
        asyncio.run(_cmd_collect(args))
    elif args.command == "deploy":
        asyncio.run(_cmd_deploy(args))
    elif args.command == "train":
        asyncio.run(_cmd_train(args))


async def _cmd_collect(args: argparse.Namespace) -> None:
    """Run EAP data collection with the data flywheel."""
    config = _load_config(args)
    config.task_name = args.task

    from roboclaw.paper.eap.engine import EAPEngine
    from roboclaw.paper.eap.judge import SuccessJudge
    from roboclaw.paper.eap.trajectory import TrajectoryDataset
    from roboclaw.paper.flywheel.flywheel import DataFlywheel
    from roboclaw.paper.policy.manager import PolicyManager
    from roboclaw.paper.policy.trainer import PolicyTrainer

    env, provider = _create_env_and_provider(config, mock=args.mock)

    # Create policies
    if args.mock:
        from roboclaw.paper.policy.mock_policy import MockPolicy
        fwd_policy = MockPolicy(name="mock_forward", mode="random")
        inv_policy = MockPolicy(name="mock_inverse", mode="random", seed=42)
    else:
        from roboclaw.paper.policy.openpi_client import OpenPIClient
        fwd_policy = OpenPIClient(config.policy_server, policy_name="openpi_forward")
        inv_policy = OpenPIClient(config.policy_server, policy_name="openpi_inverse")

    dataset = TrajectoryDataset(args.data_dir)
    judge = SuccessJudge(provider, model=config.eap.judge_model or config.vlm.model)
    policy_manager = PolicyManager(config, env, fwd_policy, inv_policy)
    trainer = PolicyTrainer(config.trainer)

    eap_engine = EAPEngine(
        config=config,
        env=env,
        forward_policy=fwd_policy,
        inverse_policy=inv_policy,
        judge=judge,
        dataset=dataset,
    )

    flywheel = DataFlywheel(
        config=config,
        eap_engine=eap_engine,
        dataset=dataset,
        trainer=trainer,
        policy_manager=policy_manager,
    )

    results = await flywheel.run(
        num_iterations=args.iterations,
        episodes_per_iter=args.episodes,
        mock_training=args.mock,
    )

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"Data Flywheel Complete: {len(results)} iterations")
    print(f"{'=' * 60}")
    for r in results:
        print(
            f"  Iter {r.iteration}: "
            f"fwd={r.batch_result.success_rate_forward:.1%} "
            f"inv={r.batch_result.success_rate_inverse:.1%} "
            f"dataset={r.dataset_size}"
        )
    print(f"\nFinal dataset: {dataset.size} trajectories")
    print(f"Data dir: {args.data_dir}")


async def _cmd_deploy(args: argparse.Namespace) -> None:
    """Execute a long-horizon task."""
    config = _load_config(args)

    from roboclaw.paper.deployment.executor import LongHorizonExecutor
    from roboclaw.paper.deployment.supervisor import DeploymentSupervisor
    from roboclaw.paper.eap.judge import SuccessJudge
    from roboclaw.paper.eap.trajectory import TrajectoryDataset
    from roboclaw.paper.policy.manager import PolicyManager

    env, provider = _create_env_and_provider(config, mock=args.mock)

    if args.mock:
        from roboclaw.paper.policy.mock_policy import MockPolicy
        fwd_policy = MockPolicy(name="mock_forward", mode="reach")
        inv_policy = MockPolicy(name="mock_inverse", mode="random", seed=42)
    else:
        from roboclaw.paper.policy.openpi_client import OpenPIClient
        fwd_policy = OpenPIClient(config.policy_server, policy_name="openpi_forward")
        inv_policy = OpenPIClient(config.policy_server, policy_name="openpi_inverse")

    policy_manager = PolicyManager(config, env, fwd_policy, inv_policy)
    judge = SuccessJudge(provider, model=config.vlm.model)
    dataset = TrajectoryDataset(config.flywheel.data_dir)

    supervisor = DeploymentSupervisor(
        config=config.deployment,
        env=env,
        policy_manager=policy_manager,
        judge=judge,
        dataset=dataset,
    )

    executor = LongHorizonExecutor(
        config=config,
        provider=provider,
        env=env,
        supervisor=supervisor,
    )

    result = await executor.execute(args.task)

    print(f"\n{'=' * 60}")
    print(f"Task: {result.task}")
    print(f"Success: {result.success}")
    print(f"Completed: {result.num_completed}/{len(result.subtasks)}")
    print(f"{'=' * 60}")
    for s in result.subtasks:
        status = "OK" if s.success else "FAIL"
        print(f"  [{status}] {s.subtask}: {s.reason}")


async def _cmd_train(args: argparse.Namespace) -> None:
    """Train policy with collected data."""
    config = _load_config(args)

    from roboclaw.paper.policy.trainer import PolicyTrainer

    trainer = PolicyTrainer(config.trainer)

    if args.mock:
        ckpt = await trainer.train_mock(args.data_dir, args.task)
    else:
        ckpt = await trainer.train(args.data_dir, args.task)

    print(f"Training complete. Checkpoint: {ckpt}")


def _load_config(args: argparse.Namespace) -> "PaperConfig":
    """Load config from YAML or create default."""
    from roboclaw.paper.config import PaperConfig

    config_path = getattr(args, "config", None)
    if config_path:
        return PaperConfig.from_yaml(config_path)

    # Try to load task-specific config
    task = getattr(args, "task", "default")
    task_config = Path(f"configs/paper/tasks/{task}.yaml")
    if task_config.exists():
        return PaperConfig.from_yaml(task_config)

    default_config = Path("configs/paper/default.yaml")
    if default_config.exists():
        return PaperConfig.from_yaml(default_config)

    return PaperConfig()


def _create_env_and_provider(
    config: "PaperConfig",
    mock: bool = False,
) -> tuple:
    """Create environment and LLM provider."""
    from roboclaw.paper.sim.tabletop_env import TabletopEnv

    env = TabletopEnv(config.sim)

    if mock:
        provider = _create_mock_provider()
    else:
        from roboclaw.providers.litellm_provider import LiteLLMProvider
        provider = LiteLLMProvider(
            api_key=config.vlm.api_key,
            api_base=config.vlm.api_base,
        )

    return env, provider


def _create_mock_provider() -> "LLMProvider":
    """Create a mock LLM provider for testing."""
    from roboclaw.providers.base import LLMProvider, LLMResponse

    class MockLLMProvider(LLMProvider):
        async def chat(self, messages, **kwargs) -> LLMResponse:
            return LLMResponse(
                content='{"success": true, "reason": "mock judgment"}',
                finish_reason="stop",
            )

        def get_default_model(self) -> str:
            return "mock"

    return MockLLMProvider()


if __name__ == "__main__":
    main()
