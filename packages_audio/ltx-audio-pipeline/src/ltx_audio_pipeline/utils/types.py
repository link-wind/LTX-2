from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

import torch

from ltx_audio_core.conditioning import ConditioningItem
from ltx_audio_core.guidance.perturbations import BatchedPerturbationConfig
from ltx_audio_core.model.transformer import Modality
from ltx_audio_core.types import LatentState


@dataclass(frozen=True)
class DenoisedLatentResult:
    """Output of one denoiser call for a single modality."""

    denoised: torch.Tensor
    uncond: torch.Tensor | None = None
    cond: torch.Tensor | None = None
    ptb: torch.Tensor | None = None
    mod: torch.Tensor | None = None

    @classmethod
    def result_or_none(
        cls,
        denoised: torch.Tensor | None,
        uncond: torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
        ptb: torch.Tensor | None = None,
        mod: torch.Tensor | None = None,
    ) -> DenoisedLatentResult | None:
        if denoised is None:
            return None
        return cls(denoised=denoised, uncond=uncond, cond=cond, ptb=ptb, mod=mod)


class TransformerLike(Protocol):
    """AV-shaped transformer protocol accepted by denoisers and samplers.

    This keeps the pipeline surface aligned with the main AV pipeline while
    still allowing an audio-only adapter to satisfy the contract.
    """

    def __call__(
        self,
        video: Modality | None = None,
        audio: Modality | None = None,
        perturbations: BatchedPerturbationConfig | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]: ...


class Denoiser(Protocol):
    """Protocol for a denoiser that receives the transformer at call time."""

    def __call__(
        self,
        transformer: TransformerLike,
        video_state: LatentState | None,
        audio_state: LatentState | None,
        sigmas: torch.Tensor,
        step_index: int,
    ) -> tuple[DenoisedLatentResult | None, DenoisedLatentResult | None]: ...


@dataclass(frozen=True)
class ModalitySpec:
    """Specification for one modality passed to a diffusion stage."""

    context: torch.Tensor
    conditionings: list[ConditioningItem] = field(default_factory=list)
    noise_scale: float = 1.0
    frozen: bool = False
    initial_latent: torch.Tensor | None = None


class OffloadMode(Enum):
    """Weight offloading strategy."""

    NONE = "none"
    CPU = "cpu"
    DISK = "disk"


__all__ = [
    "DenoisedLatentResult",
    "Denoiser",
    "LatentState",
    "ModalitySpec",
    "OffloadMode",
    "TransformerLike",
]
