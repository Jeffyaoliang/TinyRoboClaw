"""Data flywheel: iterative collect → accumulate → retrain → loop (paper Sec 4.2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from roboclaw.paper.config import PaperConfig
from roboclaw.paper.eap.engine import EAPEngine, EAPBatchResult
from roboclaw.paper.eap.trajectory import TrajectoryDataset
from roboclaw.paper.policy.base import PolicyInterface
from roboclaw.paper.policy.manager import PolicyManager
from roboclaw.paper.policy.trainer import PolicyTrainer


@dataclass
class FlywheelIterationResult:
    """Result of a single flywheel iteration."""

    iteration: int
    batch_result: EAPBatchResult
    dataset_size: int
    checkpoint_path: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


class DataFlywheel:
    """Iterative data flywheel for autonomous policy improvement.

    Each iteration:
    1. EAP autonomous data collection
    2. Accumulate to dataset D
    3. Export training data
    4. Retrain forward policy (LoRA)
    5. Load new policy
    6. Log metrics and repeat
    """

    def __init__(
        self,
        config: PaperConfig,
        eap_engine: EAPEngine,
        dataset: TrajectoryDataset,
        trainer: PolicyTrainer,
        policy_manager: PolicyManager,
    ):
        self.config = config
        self.eap_engine = eap_engine
        self.dataset = dataset
        self.trainer = trainer
        self.policy_manager = policy_manager
        self._results: list[FlywheelIterationResult] = []

    async def run(
        self,
        num_iterations: int | None = None,
        episodes_per_iter: int | None = None,
        mock_training: bool = False,
    ) -> list[FlywheelIterationResult]:
        """Run the full data flywheel.

        Args:
            num_iterations: Number of flywheel iterations (default from config).
            episodes_per_iter: Episodes per iteration (default from config).
            mock_training: If True, skip actual training (for testing).

        Returns:
            List of results for each iteration.
        """
        n_iter = num_iterations or self.config.flywheel.num_iterations
        n_episodes = episodes_per_iter or self.config.flywheel.episodes_per_iteration
        data_dir = Path(self.config.flywheel.data_dir)

        logger.info(f"Starting data flywheel: {n_iter} iterations, {n_episodes} episodes each")

        for i in range(n_iter):
            logger.info(f"=== Flywheel iteration {i + 1}/{n_iter} ===")

            # 1. EAP autonomous data collection
            batch = await self.eap_engine.run_batch(n_episodes)

            # 2. Dataset already accumulated by EAP engine
            ds_stats = self.dataset.stats()
            logger.info(f"  Dataset size: {ds_stats['total']} trajectories")

            # 3. Export training data
            export_dir = data_dir / f"train_iter_{i:02d}"
            self.dataset.export_for_training(
                export_dir,
                format=self.config.flywheel.export_format,
            )
            logger.info(f"  Exported to: {export_dir}")

            # 4. Retrain forward policy
            checkpoint_path: str | None = None
            if mock_training:
                ckpt = await self.trainer.train_mock(
                    dataset_path=export_dir,
                    task_name=f"{self.config.task_name}_iter{i}",
                )
                checkpoint_path = str(ckpt)
            else:
                try:
                    ckpt = await self.trainer.train(
                        dataset_path=export_dir,
                        task_name=f"{self.config.task_name}_iter{i}",
                    )
                    checkpoint_path = str(ckpt)
                except Exception as e:
                    logger.warning(f"  Training failed (continuing with current policy): {e}")

            # 5. Load new policy if training succeeded
            if checkpoint_path:
                logger.info(f"  New checkpoint: {checkpoint_path}")
                # In production, this would load the new LoRA weights
                # For now, we just log it
                # await self.policy_manager.reload_policy("forward", new_policy)

            # 6. Record metrics
            result = FlywheelIterationResult(
                iteration=i,
                batch_result=batch,
                dataset_size=ds_stats["total"],
                checkpoint_path=checkpoint_path,
                metrics={
                    "success_rate_forward": batch.success_rate_forward,
                    "success_rate_inverse": batch.success_rate_inverse,
                    "total_steps": batch.total_steps,
                    **ds_stats,
                },
            )
            self._results.append(result)

            logger.info(
                f"  Iteration {i + 1} done: "
                f"fwd={batch.success_rate_forward:.1%}, "
                f"inv={batch.success_rate_inverse:.1%}, "
                f"dataset={ds_stats['total']}"
            )

        logger.info("Data flywheel complete.")
        return self._results

    @property
    def results(self) -> list[FlywheelIterationResult]:
        return self._results
