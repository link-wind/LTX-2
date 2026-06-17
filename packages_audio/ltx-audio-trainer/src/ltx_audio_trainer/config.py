from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ltx_audio_trainer.quantization import QuantizationOptions
from ltx_audio_trainer.training_strategies.text_to_audio import TextToAudioConfig


class ConfigBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelConfig(ConfigBaseModel):
    model_path: str | Path = Field(
        ...,
        description="Local path to the audio checkpoint safetensors file.",
    )
    text_encoder_path: str | Path | None = Field(
        default=None,
        description="Local path to the Gemma model directory.",
    )
    training_mode: Literal["lora", "full"] = Field(
        default="lora",
        description="Whether to train LoRA adapters or full weights.",
    )
    load_checkpoint: str | Path | None = Field(
        default=None,
        description="Optional path to a trainer checkpoint for resuming.",
    )

    @model_validator(mode="after")
    def validate_paths(self) -> "ModelConfig":
        if not Path(self.model_path).exists():
            raise ValueError(f"Model path does not exist: {self.model_path}")
        if self.text_encoder_path is not None and not Path(self.text_encoder_path).exists():
            raise ValueError(f"Text encoder path does not exist: {self.text_encoder_path}")
        return self


class LoraConfig(ConfigBaseModel):
    rank: int = Field(default=64, ge=2)
    alpha: int = Field(default=64, ge=1)
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)
    target_modules: list[str] = Field(
        default=["to_k", "to_q", "to_v", "to_out.0"],
        description="Attention modules to target with LoRA.",
    )


TrainingStrategyConfig = TextToAudioConfig


class OptimizationConfig(ConfigBaseModel):
    learning_rate: float = Field(default=5e-4)
    steps: int = Field(default=3000, gt=0)
    batch_size: int = Field(default=2, gt=0)
    gradient_accumulation_steps: int = Field(default=1, gt=0)
    max_grad_norm: float = Field(default=1.0, gt=0)
    optimizer_type: Literal["adamw", "adamw8bit"] = Field(default="adamw")
    scheduler_type: Literal["constant", "linear", "cosine", "cosine_with_restarts", "polynomial", "step"] = Field(
        default="linear"
    )
    scheduler_params: dict = Field(default_factory=dict)
    enable_gradient_checkpointing: bool = Field(default=False)


class AccelerationConfig(ConfigBaseModel):
    mixed_precision_mode: Literal["no", "fp16", "bf16"] | None = Field(default="bf16")
    quantization: QuantizationOptions | None = Field(default=None)
    load_text_encoder_in_8bit: bool = Field(default=False)
    offload_optimizer_during_validation: bool = Field(default=False)


class DataConfig(ConfigBaseModel):
    preprocessed_data_root: str = Field(
        ...,
        description="Path to the precomputed audio latent + text feature dataset root.",
    )
    num_dataloader_workers: int = Field(default=2, ge=0)


class ValidationConfig(ConfigBaseModel):
    prompts: list[str] = Field(default_factory=list)
    negative_prompt: str = Field(default="")
    audio_duration_seconds: float = Field(
        default=10.0,
        gt=0,
        description="Validation audio duration in seconds.",
    )
    seed: int = Field(default=42)
    inference_steps: int = Field(default=50, gt=0)
    interval: int | None = Field(default=100, gt=0)
    guidance_scale: float = Field(default=4.0, ge=1.0)
    skip_initial_validation: bool = Field(default=False)


class CheckpointsConfig(ConfigBaseModel):
    interval: int | None = Field(default=None, gt=0)
    keep_last_n: int = Field(default=1, ge=-1)
    precision: Literal["bfloat16", "float32"] = Field(default="bfloat16")
    no_resume: bool = Field(default=False)
    save_training_state: Literal["full", "minimal", "off"] = Field(default="minimal")


class HubConfig(ConfigBaseModel):
    push_to_hub: bool = Field(default=False)
    hub_model_id: str | None = Field(default=None)

    @model_validator(mode="after")
    def validate_hub_config(self) -> "HubConfig":
        if self.push_to_hub and not self.hub_model_id:
            raise ValueError("hub_model_id must be specified when push_to_hub is True")
        return self


class WandbConfig(ConfigBaseModel):
    enabled: bool = Field(default=False)
    project: str = Field(default="ltx-audio-trainer")
    entity: str | None = Field(default=None)
    tags: list[str] = Field(default_factory=list)
    log_validation_audio: bool = Field(default=True)


class FlowMatchingConfig(ConfigBaseModel):
    timestep_sampling_mode: Literal["uniform", "shifted_logit_normal"] = Field(default="shifted_logit_normal")
    timestep_sampling_params: dict = Field(default_factory=dict)


class AudioTrainerConfig(ConfigBaseModel):
    model: ModelConfig
    lora: LoraConfig | None = Field(default=None)
    training_strategy: TrainingStrategyConfig = Field(default_factory=TextToAudioConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    acceleration: AccelerationConfig = Field(default_factory=AccelerationConfig)
    data: DataConfig
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    checkpoints: CheckpointsConfig = Field(default_factory=CheckpointsConfig)
    hub: HubConfig = Field(default_factory=HubConfig)
    flow_matching: FlowMatchingConfig = Field(default_factory=FlowMatchingConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)
    seed: int = Field(default=42)
    output_dir: str = Field(default="outputs")

    @model_validator(mode="after")
    def validate_lora_compatibility(self) -> "AudioTrainerConfig":
        if self.model.training_mode == "lora" and self.lora is None:
            self.lora = LoraConfig()
        if self.model.training_mode == "full" and self.lora is not None:
            raise ValueError("lora config must be omitted when training_mode='full'")
        return self
