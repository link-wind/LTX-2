from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace

from safetensors import safe_open

from ltx_audio_core.components.guiders import MultiModalGuiderParams


@dataclass(frozen=True)
class PipelineParams:
    seed: int = 10
    height: int = 512
    width: int = 768
    num_frames: int = 121
    frame_rate: float = 24.0
    num_inference_steps: int = 40
    audio_guider_params: MultiModalGuiderParams = field(
        default_factory=lambda: MultiModalGuiderParams(
            cfg_scale=7.0,
            stg_scale=1.0,
            rescale_scale=0.7,
            modality_scale=3.0,
            skip_step=0,
            stg_blocks=[29],
        )
    )


LTX_2_PARAMS = PipelineParams()
LTX_2_3_PARAMS = replace(
    LTX_2_PARAMS,
    num_inference_steps=30,
    audio_guider_params=replace(LTX_2_PARAMS.audio_guider_params, stg_blocks=[28]),
)

_LTX_2_3_MODEL_VERSION_PREFIX = "2.3"


def detect_params(checkpoint_path: str) -> PipelineParams:
    logger = logging.getLogger(__name__)

    try:
        with safe_open(checkpoint_path, framework="pt") as handle:
            metadata = handle.metadata() or {}
        version = metadata.get("model_version", "")
    except Exception:
        logger.warning("Could not read checkpoint metadata from %s, using LTX-2 defaults", checkpoint_path)
        return LTX_2_PARAMS

    if version.startswith(_LTX_2_3_MODEL_VERSION_PREFIX):
        return LTX_2_3_PARAMS

    logger.info("Using LTX_2_PARAMS for checkpoint (version=%s)", version or "unknown")
    return LTX_2_PARAMS


__all__ = [
    "LTX_2_3_PARAMS",
    "LTX_2_PARAMS",
    "PipelineParams",
    "detect_params",
]
