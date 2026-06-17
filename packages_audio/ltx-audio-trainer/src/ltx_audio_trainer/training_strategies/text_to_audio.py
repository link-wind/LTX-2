from __future__ import annotations

from typing import Any, Literal

import torch
from pydantic import Field
from torch import Tensor

from ltx_audio_core.model.transformer.modality import Modality
from ltx_audio_trainer.timestep_samplers import TimestepSampler
from ltx_audio_trainer.training_strategies.base_strategy import (
    ModelInputs,
    TrainingStrategy,
    TrainingStrategyConfigBase,
)


class TextToAudioConfig(TrainingStrategyConfigBase):
    name: Literal["text_to_audio"] = "text_to_audio"
    with_text_conditioning_dropout: bool = Field(
        default=True,
        description="Whether to randomly drop text conditioning during training.",
    )
    text_conditioning_dropout_p: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Probability of dropping text conditioning on a sample.",
    )


class TextToAudioStrategy(TrainingStrategy):
    name: Literal["text_to_audio"] = "text_to_audio"
    config: TextToAudioConfig

    def __init__(self, config: TextToAudioConfig) -> None:
        super().__init__(config)

    def get_data_sources(self) -> dict[str, str]:
        return {
            "audio_latents": "latent_conditions",
            "conditions": "text_conditions",
        }

    def prepare_training_inputs(
        self,
        batch: dict[str, Any],
        timestep_sampler: TimestepSampler,
    ) -> ModelInputs:
        latent_conditions = batch["latent_conditions"]
        text_conditions = batch["text_conditions"]

        audio_latents = latent_conditions["latents"]
        audio_latents = self._audio_patchifier.patchify(audio_latents)

        batch_size, num_time_steps, _ = audio_latents.shape
        device = audio_latents.device
        dtype = audio_latents.dtype

        audio_context = text_conditions.get("audio_prompt_embeds")
        if audio_context is None:
            audio_context = text_conditions["video_prompt_embeds"]

        context_mask = text_conditions["prompt_attention_mask"]
        audio_context, context_mask = self._apply_text_conditioning_dropout(audio_context, context_mask)

        sigma = timestep_sampler.sample_for(audio_latents)
        sigma_expanded = sigma.view(-1, 1, 1)
        noise = torch.randn_like(audio_latents)
        noisy_audio = (1 - sigma_expanded) * audio_latents + sigma_expanded * noise
        targets = noise - audio_latents
        timesteps = sigma.view(-1, 1).expand(-1, num_time_steps)
        positions = self._get_audio_positions(
            num_time_steps=num_time_steps,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

        audio = Modality(
            latent=noisy_audio,
            sigma=sigma,
            timesteps=timesteps,
            positions=positions,
            context=audio_context,
            context_mask=context_mask,
        )
        loss_mask = torch.ones(batch_size, num_time_steps, dtype=torch.bool, device=device)

        return ModelInputs(
            audio=audio,
            audio_targets=targets,
            audio_loss_mask=loss_mask,
        )

    def compute_loss(
        self,
        audio_pred: Tensor,
        inputs: ModelInputs,
    ) -> Tensor:
        audio_loss = (audio_pred - inputs.audio_targets).pow(2)
        loss_mask = inputs.audio_loss_mask.unsqueeze(-1).float()
        masked = audio_loss * loss_mask
        return masked.mean(dim=[-2, -1]) / loss_mask.mean(dim=[-2, -1]).clamp(min=1e-8)

    def _apply_text_conditioning_dropout(
        self,
        audio_context: Tensor,
        context_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if not self.config.with_text_conditioning_dropout or self.config.text_conditioning_dropout_p <= 0:
            return audio_context, context_mask

        batch_size = audio_context.shape[0]
        keep_mask = torch.rand(batch_size, device=audio_context.device) >= self.config.text_conditioning_dropout_p
        context_keep = keep_mask.view(batch_size, 1, 1)
        mask_keep = keep_mask.view(batch_size, 1)

        dropped_context = torch.where(context_keep, audio_context, torch.zeros_like(audio_context))
        dropped_mask = torch.where(mask_keep, context_mask, torch.zeros_like(context_mask))
        return dropped_context, dropped_mask
