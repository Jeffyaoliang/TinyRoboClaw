"""Paper-specific configuration (PaperConfig)."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class VLMConfig(BaseModel):
    """VLM agent configuration."""

    provider: str = "openai"
    model: str = "gpt-4o"
    api_key: str | None = None
    api_base: str | None = None
    temperature: float = 0.3
    max_tokens: int = 4096
    max_iterations: int = 40


class PolicyServerConfig(BaseModel):
    """OpenPI policy server configuration."""

    host: str = "localhost"
    port: int = 8000
    ws_url: str = "ws://localhost:8000"
    model_name: str = "pi0_fast_base"
    action_chunk_size: int = 50


class TrainerConfig(BaseModel):
    """LoRA fine-tuning configuration."""

    lora_rank: int = 16
    lora_alpha: int = 16
    train_steps: int = 10_000
    batch_size: int = 16
    learning_rate: float = 1e-4
    save_interval: int = 2000
    openpi_dir: str = "~/openpi"
    output_dir: str = "checkpoints"


class EAPConfig(BaseModel):
    """Entangled Action Pair configuration."""

    episodes_per_batch: int = 50
    max_steps_per_episode: int = 200
    success_threshold: float = 0.5
    judge_model: str | None = None  # None → use VLM config


class FlywheelConfig(BaseModel):
    """Data flywheel configuration."""

    num_iterations: int = 5
    episodes_per_iteration: int = 50
    data_dir: str = "data/paper"
    export_format: str = "lerobot"


class DeploymentConfig(BaseModel):
    """Deployment supervision configuration."""

    monitor_interval: float = 2.0
    max_retries: int = 3
    failure_threshold: float = 0.3
    collect_deployment_data: bool = True


class SimConfig(BaseModel):
    """Simulation environment configuration."""

    workspace_bounds: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0, 0.5, 0.5, 0.3])
    num_objects: int = 3
    image_size: tuple[int, int] = (224, 224)
    dt: float = 0.05


class PaperConfig(BaseModel):
    """Root configuration for the paper implementation."""

    vlm: VLMConfig = Field(default_factory=VLMConfig)
    policy_server: PolicyServerConfig = Field(default_factory=PolicyServerConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    eap: EAPConfig = Field(default_factory=EAPConfig)
    flywheel: FlywheelConfig = Field(default_factory=FlywheelConfig)
    deployment: DeploymentConfig = Field(default_factory=DeploymentConfig)
    sim: SimConfig = Field(default_factory=SimConfig)

    task_name: str = "default"
    task_instruction: str = ""
    inverse_instruction: str = ""

    @classmethod
    def from_yaml(cls, path: str | Path) -> PaperConfig:
        """Load configuration from a YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
