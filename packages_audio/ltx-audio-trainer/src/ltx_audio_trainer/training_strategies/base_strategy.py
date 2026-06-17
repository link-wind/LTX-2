from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor

from ltx_audio_core.components.patchifiers import AudioPatchifier
from ltx_audio_core.model.transformer.modality import Modality
from ltx_audio_core.types import AudioLatentShape
from ltx_audio_trainer.timestep_samplers import TimestepSampler


class TrainingStrategyConfigBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Literal["text_to_audio"] = Field(description="Training strategy name")


@dataclass
class ModelInputs:
    """Container for audio transformer inputs and loss targets."""

    audio: Modality
    audio_targets: Tensor
    audio_loss_mask: Tensor


class TrainingStrategy(ABC):
    """Base class for audio-only training strategies."""

    name: Literal["text_to_audio"]

    def __init__(self, config: TrainingStrategyConfigBase) -> None:
        self.config = config
        self._audio_patchifier = AudioPatchifier(patch_size=1)

    @abstractmethod
    def get_data_sources(self) -> list[str] | dict[str, str]:
        """Return the precomputed directories required by this strategy."""

    @abstractmethod
    def prepare_training_inputs(
        self,
        batch: dict[str, Any],
        timestep_sampler: TimestepSampler,
    ) -> ModelInputs:
        """Convert a dataset batch into transformer-ready inputs and targets."""

    @abstractmethod
    def compute_loss(
        self,
        audio_pred: Tensor,
        inputs: ModelInputs,
    ) -> Tensor:
        """Return per-sample loss values shaped ``[batch]``."""

    def _get_audio_positions(
        self,
        num_time_steps: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        positions = self._audio_patchifier.get_patch_grid_bounds(
            output_shape=AudioLatentShape(
                frames=num_time_steps,
                mel_bins=16,
                batch=batch_size,
                channels=8,
            ),
            device=device,
        )
        return positions.to(dtype=dtype)
