from __future__ import annotations

from ltx_audio_trainer import logger
from ltx_audio_trainer.training_strategies.base_strategy import (
    ModelInputs,
    TrainingStrategy,
    TrainingStrategyConfigBase,
)
from ltx_audio_trainer.training_strategies.text_to_audio import TextToAudioConfig, TextToAudioStrategy

TrainingStrategyConfig = TextToAudioConfig

__all__ = [
    "ModelInputs",
    "TextToAudioConfig",
    "TextToAudioStrategy",
    "TrainingStrategy",
    "TrainingStrategyConfig",
    "TrainingStrategyConfigBase",
    "get_training_strategy",
]


def get_training_strategy(config: TrainingStrategyConfig) -> TrainingStrategy:
    match config:
        case TextToAudioConfig():
            strategy = TextToAudioStrategy(config)
        case _:
            raise ValueError(f"Unknown training strategy config type: {type(config).__name__}")

    logger.debug(f"Using {strategy.__class__.__name__} for audio-only training")
    return strategy
