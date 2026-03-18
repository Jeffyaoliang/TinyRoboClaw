"""LoRA fine-tuning pipeline for OpenPI policies (paper Sec 3.2).

OpenPI training workflow (from official README):
1. 数据转换为 LeRobot 格式
2. uv run scripts/compute_norm_stats.py --config-name=<config>
3. XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py <config> --exp-name=<name>
4. uv run scripts/serve_policy.py policy:checkpoint --policy.config=<config> --policy.dir=<ckpt>
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from loguru import logger

from roboclaw.paper.config import TrainerConfig


class PolicyTrainer:
    """Wraps OpenPI training commands as subprocess calls.

    Training config (from paper):
    - LoRA rank=16, alpha=16
    - 10k steps, batch_size=16
    - Input: LeRobot-format dataset
    - Output: LoRA checkpoint
    """

    def __init__(self, config: TrainerConfig):
        self.config = config
        self._training = False

    @property
    def is_training(self) -> bool:
        return self._training

    async def train(
        self,
        dataset_path: str | Path,
        task_name: str = "default",
        checkpoint_name: str | None = None,
        model_config: str = "pi0_fast_base",
    ) -> Path:
        """Launch LoRA fine-tuning via OpenPI's uv-based workflow.

        Args:
            dataset_path: Path to exported LeRobot-format training data.
            task_name: Name for the training run.
            checkpoint_name: Experiment name (auto-generated if None).
            model_config: OpenPI model config name.

        Returns:
            Path to the output checkpoint directory.
        """
        dataset_path = Path(dataset_path).resolve()
        openpi_dir = Path(self.config.openpi_dir).expanduser().resolve()

        if not openpi_dir.exists():
            raise FileNotFoundError(
                f"OpenPI 目录不存在: {openpi_dir}\n"
                f"请先运行: bash scripts/setup_openpi.sh install"
            )

        exp_name = checkpoint_name or f"roboclaw_{task_name}"

        # Step 1: 计算归一化统计量
        logger.info("Step 1/2: 计算归一化统计量...")
        await self._run_in_openpi(
            openpi_dir,
            ["uv", "run", "scripts/compute_norm_stats.py", f"--config-name={model_config}"],
        )

        # Step 2: 启动训练
        logger.info(f"Step 2/2: 启动 LoRA 训练 (config={model_config}, exp={exp_name})...")

        env = {
            **os.environ,
            "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.9",
        }

        await self._run_in_openpi(
            openpi_dir,
            ["uv", "run", "scripts/train.py", model_config, f"--exp-name={exp_name}", "--overwrite"],
            env=env,
        )

        # 找到最新的 checkpoint
        checkpoint_dir = openpi_dir / "checkpoints" / model_config / exp_name
        latest_ckpt = self._find_latest_checkpoint(checkpoint_dir)

        logger.info(f"训练完成. Checkpoint: {latest_ckpt}")
        return latest_ckpt

    async def _run_in_openpi(
        self,
        openpi_dir: Path,
        cmd: list[str],
        env: dict[str, str] | None = None,
    ) -> None:
        """Run a command inside the OpenPI directory."""
        logger.info(f"运行: {' '.join(cmd)}")
        self._training = True

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(openpi_dir),
                env=env,
            )

            stdout, stderr = await process.communicate()

            # Log output
            if stdout:
                for line in stdout.decode().strip().split("\n")[-10:]:
                    logger.info(f"  {line}")
            if stderr:
                for line in stderr.decode().strip().split("\n")[-5:]:
                    logger.warning(f"  {line}")

            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "unknown error"
                raise RuntimeError(f"命令失败 (code={process.returncode}): {error_msg[-500:]}")

        finally:
            self._training = False

    def _find_latest_checkpoint(self, checkpoint_dir: Path) -> Path:
        """Find the latest numbered checkpoint subdirectory."""
        if not checkpoint_dir.exists():
            return checkpoint_dir

        # OpenPI saves checkpoints as numbered directories: 2000/, 4000/, ...
        numbered = []
        for d in checkpoint_dir.iterdir():
            if d.is_dir() and d.name.isdigit():
                numbered.append((int(d.name), d))

        if numbered:
            numbered.sort(key=lambda x: x[0], reverse=True)
            return numbered[0][1]

        return checkpoint_dir

    async def serve_checkpoint(
        self,
        checkpoint_path: str | Path,
        model_config: str = "pi0_fast_base",
        port: int = 8000,
    ) -> asyncio.subprocess.Process:
        """Start OpenPI policy server with a trained checkpoint.

        Returns the subprocess (runs in background).
        """
        openpi_dir = Path(self.config.openpi_dir).expanduser().resolve()
        checkpoint_path = Path(checkpoint_path).resolve()

        cmd = [
            "uv", "run", "scripts/serve_policy.py",
            "policy:checkpoint",
            f"--policy.config={model_config}",
            f"--policy.dir={checkpoint_path}",
            f"--port={port}",
        ]

        logger.info(f"启动推理服务: {' '.join(cmd)}")

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(openpi_dir),
        )

        # Wait a bit for server to start
        await asyncio.sleep(5)

        if process.returncode is not None:
            stderr = await process.stderr.read()
            raise RuntimeError(f"推理服务启动失败: {stderr.decode()}")

        logger.info(f"推理服务已启动 (port={port}, PID={process.pid})")
        return process

    async def train_mock(
        self,
        dataset_path: str | Path,
        task_name: str = "default",
    ) -> Path:
        """Mock training for testing (no actual training, just creates checkpoint dir)."""
        output_dir = Path(self.config.output_dir) / f"lora_{task_name}"
        output_dir.mkdir(parents=True, exist_ok=True)

        (output_dir / "checkpoint.pt").write_text("mock_checkpoint")
        (output_dir / "config.json").write_text(
            f'{{"task": "{task_name}", "steps": {self.config.train_steps}}}'
        )

        logger.info(f"Mock training complete. Checkpoint: {output_dir}")
        return output_dir
