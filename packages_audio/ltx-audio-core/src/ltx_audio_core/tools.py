from dataclasses import replace
from typing import Protocol

import torch
from torch._prims_common import DeviceLikeType

from ltx_audio_core.components.patchifiers import AudioPatchifier
from ltx_audio_core.components.protocols import Patchifier
from ltx_audio_core.types import AudioLatentShape, LatentState


class LatentTools(Protocol):
    """
    Audio latent utility protocol used by conditioning and diffusion helpers.
    """

    patchifier: Patchifier
    target_shape: AudioLatentShape

    def create_initial_state(
        self,
        device: DeviceLikeType,
        dtype: torch.dtype,
        initial_latent: torch.Tensor | None = None,
    ) -> LatentState:
        ...

    def patchify(self, latent_state: LatentState) -> LatentState:
        if latent_state.latent.shape != self.target_shape.to_torch_shape():
            raise ValueError(
                f"Latent state has shape {latent_state.latent.shape}, "
                f"expected shape is {self.target_shape.to_torch_shape()}"
            )

        latent_state = latent_state.clone()
        latent = self.patchifier.patchify(latent_state.latent)
        clean_latent = self.patchifier.patchify(latent_state.clean_latent)
        denoise_mask = self.patchifier.patchify(latent_state.denoise_mask)
        return replace(
            latent_state,
            latent=latent,
            denoise_mask=denoise_mask,
            clean_latent=clean_latent,
        )

    def unpatchify(self, latent_state: LatentState) -> LatentState:
        latent_state = latent_state.clone()
        latent = self.patchifier.unpatchify(latent_state.latent, output_shape=self.target_shape)
        clean_latent = self.patchifier.unpatchify(latent_state.clean_latent, output_shape=self.target_shape)
        denoise_mask = self.patchifier.unpatchify(
            latent_state.denoise_mask,
            output_shape=self.target_shape.mask_shape(),
        )
        return replace(
            latent_state,
            latent=latent,
            denoise_mask=denoise_mask,
            clean_latent=clean_latent,
        )

    def clear_conditioning(self, latent_state: LatentState) -> LatentState:
        latent_state = latent_state.clone()

        num_tokens = self.patchifier.get_token_count(self.target_shape)
        latent = latent_state.latent[:, :num_tokens]
        clean_latent = latent_state.clean_latent[:, :num_tokens]
        denoise_mask = torch.ones_like(latent_state.denoise_mask)[:, :num_tokens]
        positions = latent_state.positions[:, :, :num_tokens]

        return LatentState(
            latent=latent,
            denoise_mask=denoise_mask,
            positions=positions,
            clean_latent=clean_latent,
            attention_mask=None,
        )


class AudioLatentTools:
    """
    Audio-only latent tools for building the initial patchified diffusion state.
    """

    patchifier: AudioPatchifier
    target_shape: AudioLatentShape

    def __init__(self, patchifier: AudioPatchifier, target_shape: AudioLatentShape):
        self.patchifier = patchifier
        self.target_shape = target_shape

    def create_initial_state(
        self,
        device: DeviceLikeType,
        dtype: torch.dtype,
        initial_latent: torch.Tensor | None = None,
    ) -> LatentState:
        if initial_latent is not None:
            assert initial_latent.shape == self.target_shape.to_torch_shape(), (
                f"Latent shape {initial_latent.shape} "
                f"does not match target shape {self.target_shape.to_torch_shape()}"
            )
        else:
            initial_latent = torch.zeros(
                *self.target_shape.to_torch_shape(),
                device=device,
                dtype=dtype,
            )

        clean_latent = initial_latent.clone()
        denoise_mask = torch.ones(
            *self.target_shape.mask_shape().to_torch_shape(),
            device=device,
            dtype=torch.float32,
        )
        positions = self.patchifier.get_patch_grid_bounds(
            output_shape=self.target_shape,
            device=device,
        )

        return self.patchify(
            LatentState(
                latent=initial_latent,
                denoise_mask=denoise_mask,
                positions=positions,
                clean_latent=clean_latent,
            )
        )
