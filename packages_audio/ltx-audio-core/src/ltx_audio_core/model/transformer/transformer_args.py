from dataclasses import dataclass, replace

import torch

from ltx_audio_core.guidance.perturbations import BatchedPerturbationConfig, PerturbationType
from ltx_audio_core.model.transformer.adaln import AdaLayerNormSingle
from ltx_audio_core.model.transformer.modality import Modality
from ltx_audio_core.model.transformer.rope import (
    LTXRopeType,
    generate_freq_grid_np,
    generate_freq_grid_pytorch,
    precompute_freqs_cis,
)


@dataclass(frozen=True)
class TransformerArgs:
    x: torch.Tensor
    context: torch.Tensor
    context_mask: torch.Tensor | None
    timesteps: torch.Tensor
    embedded_timestep: torch.Tensor
    positional_embeddings: tuple[torch.Tensor, torch.Tensor]
    enabled: bool
    prompt_timestep: torch.Tensor | None = None
    self_attention_mask: torch.Tensor | None = None
    self_attn_perturbation_mask: torch.Tensor | None = None
    self_attn_all_perturbed: bool = False


class BlockPerturbationsProcessor:
    """Audio-only per-block preparation for self-attention perturbations."""

    def __call__(
        self,
        args: TransformerArgs,
        perturbations: BatchedPerturbationConfig,
        block_idx: int,
        self_attn_type: PerturbationType,
    ) -> TransformerArgs:
        device, dtype = args.x.device, args.x.dtype

        all_self = perturbations.all_in_batch(self_attn_type, block_idx)
        any_self = perturbations.any_in_batch(self_attn_type, block_idx)

        self_mask: torch.Tensor | None = None
        if any_self and not all_self:
            self_mask = perturbations.mask(self_attn_type, block_idx, device, dtype).view(-1, 1, 1)

        return replace(
            args,
            self_attn_perturbation_mask=self_mask,
            self_attn_all_perturbed=all_self,
        )


class TransformerArgsPreprocessor:
    def __init__(  # noqa: PLR0913
        self,
        patchify_proj: torch.nn.Linear,
        adaln: AdaLayerNormSingle,
        inner_dim: int,
        max_pos: list[int],
        num_attention_heads: int,
        use_middle_indices_grid: bool,
        timestep_scale_multiplier: int,
        double_precision_rope: bool,
        positional_embedding_theta: float,
        rope_type: LTXRopeType,
        caption_projection: torch.nn.Module | None = None,
        prompt_adaln: AdaLayerNormSingle | None = None,
    ) -> None:
        self.patchify_proj = patchify_proj
        self.adaln = adaln
        self.inner_dim = inner_dim
        self.max_pos = max_pos
        self.num_attention_heads = num_attention_heads
        self.use_middle_indices_grid = use_middle_indices_grid
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.double_precision_rope = double_precision_rope
        self.positional_embedding_theta = positional_embedding_theta
        self.rope_type = rope_type
        self.caption_projection = caption_projection
        self.prompt_adaln = prompt_adaln

    def _prepare_timestep(
        self,
        timestep: torch.Tensor,
        adaln: AdaLayerNormSingle,
        batch_size: int,
        hidden_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        timestep_scaled = timestep * self.timestep_scale_multiplier
        timestep, embedded_timestep = adaln(
            timestep_scaled.flatten(),
            hidden_dtype=hidden_dtype,
        )
        timestep = timestep.view(batch_size, -1, timestep.shape[-1])
        embedded_timestep = embedded_timestep.view(batch_size, -1, embedded_timestep.shape[-1])
        return timestep, embedded_timestep

    def _prepare_context(
        self,
        context: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if self.caption_projection is not None:
            context = self.caption_projection(context)
        batch_size = x.shape[0]
        return context.view(batch_size, -1, x.shape[-1])

    def _prepare_attention_mask(
        self,
        attention_mask: torch.Tensor | None,
        x_dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if attention_mask is None or torch.is_floating_point(attention_mask):
            return attention_mask

        return (attention_mask - 1).to(x_dtype).reshape(
            (attention_mask.shape[0], 1, -1, attention_mask.shape[-1])
        ) * torch.finfo(x_dtype).max

    def _prepare_self_attention_mask(
        self,
        attention_mask: torch.Tensor | None,
        x_dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if attention_mask is None:
            return None

        finfo = torch.finfo(x_dtype)
        eps = finfo.tiny

        bias = torch.full_like(attention_mask, finfo.min, dtype=x_dtype)
        positive = attention_mask > 0
        if positive.any():
            bias[positive] = torch.log(attention_mask[positive].clamp(min=eps)).to(x_dtype)

        return bias.unsqueeze(1)

    def _prepare_positional_embeddings(
        self,
        positions: torch.Tensor,
        inner_dim: int,
        max_pos: list[int],
        use_middle_indices_grid: bool,
        num_attention_heads: int,
        x_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        freq_grid_generator = generate_freq_grid_np if self.double_precision_rope else generate_freq_grid_pytorch
        return precompute_freqs_cis(
            positions,
            dim=inner_dim,
            out_dtype=x_dtype,
            theta=self.positional_embedding_theta,
            max_pos=max_pos,
            use_middle_indices_grid=use_middle_indices_grid,
            num_attention_heads=num_attention_heads,
            rope_type=self.rope_type,
            freq_grid_generator=freq_grid_generator,
        )

    def prepare(self, modality: Modality) -> TransformerArgs:
        x = self.patchify_proj(modality.latent)
        batch_size = x.shape[0]

        timestep, embedded_timestep = self._prepare_timestep(
            modality.timesteps,
            self.adaln,
            batch_size,
            modality.latent.dtype,
        )

        prompt_timestep = None
        if self.prompt_adaln is not None:
            prompt_timestep, _ = self._prepare_timestep(
                modality.sigma,
                self.prompt_adaln,
                batch_size,
                modality.latent.dtype,
            )

        context = self._prepare_context(modality.context, x)
        context_mask = self._prepare_attention_mask(modality.context_mask, modality.latent.dtype)
        positional_embeddings = self._prepare_positional_embeddings(
            positions=modality.positions,
            inner_dim=self.inner_dim,
            max_pos=self.max_pos,
            use_middle_indices_grid=self.use_middle_indices_grid,
            num_attention_heads=self.num_attention_heads,
            x_dtype=modality.latent.dtype,
        )
        self_attention_mask = self._prepare_self_attention_mask(
            modality.attention_mask,
            modality.latent.dtype,
        )

        return TransformerArgs(
            x=x,
            context=context,
            context_mask=context_mask,
            timesteps=timestep,
            embedded_timestep=embedded_timestep,
            positional_embeddings=positional_embeddings,
            enabled=modality.enabled,
            prompt_timestep=prompt_timestep,
            self_attention_mask=self_attention_mask,
        )
