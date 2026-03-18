"""Tests for roboclaw.paper.config."""

import tempfile
from pathlib import Path

import pytest
import yaml

from roboclaw.paper.config import (
    DeploymentConfig,
    EAPConfig,
    FlywheelConfig,
    PaperConfig,
    PolicyServerConfig,
    SimConfig,
    TrainerConfig,
    VLMConfig,
)


class TestPaperConfig:
    def test_default_values(self):
        cfg = PaperConfig()
        assert cfg.vlm.model == "gpt-4o"
        assert cfg.vlm.temperature == 0.3
        assert cfg.policy_server.port == 8000
        assert cfg.trainer.lora_rank == 16
        assert cfg.trainer.lora_alpha == 16
        assert cfg.trainer.train_steps == 10_000
        assert cfg.eap.episodes_per_batch == 50
        assert cfg.flywheel.num_iterations == 5
        assert cfg.deployment.max_retries == 3
        assert cfg.task_name == "default"

    def test_custom_values(self):
        cfg = PaperConfig(
            vlm=VLMConfig(model="qwen-vl", temperature=0.5),
            task_name="lipstick",
            task_instruction="insert lipstick",
        )
        assert cfg.vlm.model == "qwen-vl"
        assert cfg.vlm.temperature == 0.5
        assert cfg.task_name == "lipstick"
        assert cfg.task_instruction == "insert lipstick"

    def test_from_yaml(self):
        data = {
            "task_name": "test_task",
            "task_instruction": "pick up the block",
            "vlm": {"model": "claude-sonnet", "temperature": 0.2},
            "eap": {"episodes_per_batch": 10, "max_steps_per_episode": 50},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            f.flush()
            cfg = PaperConfig.from_yaml(f.name)

        assert cfg.task_name == "test_task"
        assert cfg.vlm.model == "claude-sonnet"
        assert cfg.vlm.temperature == 0.2
        assert cfg.eap.episodes_per_batch == 10
        # Defaults preserved for unspecified fields
        assert cfg.trainer.lora_rank == 16
        assert cfg.deployment.max_retries == 3

    def test_sim_config_defaults(self):
        cfg = SimConfig()
        assert len(cfg.workspace_bounds) == 6
        assert cfg.num_objects == 3
        assert cfg.image_size == (224, 224)
        assert cfg.dt == 0.05

    def test_trainer_config(self):
        cfg = TrainerConfig(train_steps=5000, batch_size=8)
        assert cfg.train_steps == 5000
        assert cfg.batch_size == 8
        assert cfg.lora_rank == 16  # default

    def test_policy_server_config(self):
        cfg = PolicyServerConfig(host="10.0.0.1", port=9000)
        assert cfg.host == "10.0.0.1"
        assert cfg.port == 9000
        assert cfg.ws_url == "ws://localhost:8000"  # independent field
