from __future__ import annotations

import ast
import sys
from pathlib import Path


def test_config_module_exists_with_audio_trainer_classes() -> None:
    config_path = Path(__file__).resolve().parents[1] / "src" / "ltx_audio_trainer" / "config.py"

    assert config_path.exists(), f"Missing config module: {config_path}"

    module = ast.parse(config_path.read_text(encoding="utf-8"))
    class_names = {node.name for node in module.body if isinstance(node, ast.ClassDef)}

    assert {
        "ConfigBaseModel",
        "ModelConfig",
        "LoraConfig",
        "OptimizationConfig",
        "AccelerationConfig",
        "DataConfig",
        "ValidationConfig",
        "CheckpointsConfig",
        "FlowMatchingConfig",
        "AudioTrainerConfig",
    }.issubset(class_names)
    assert _has_training_strategy_config_alias(module)


def test_config_module_mentions_text_to_audio_strategy_and_quantization() -> None:
    config_path = Path(__file__).resolve().parents[1] / "src" / "ltx_audio_trainer" / "config.py"
    source = config_path.read_text(encoding="utf-8")

    assert "text_to_audio" in source
    assert "QuantizationOptions" in source
    assert "preprocessed_data_root" in source
    assert "audio_duration_seconds" in source


def test_audio_trainer_config_can_be_imported() -> None:
    package_src = Path(__file__).resolve().parents[1] / "src"
    sys.path.insert(0, str(package_src))
    try:
        from ltx_audio_trainer.config import AudioTrainerConfig  # noqa: PLC0415

        assert AudioTrainerConfig.__name__ == "AudioTrainerConfig"
    finally:
        sys.path.pop(0)


def test_audio_trainer_config_can_be_instantiated_with_minimal_paths() -> None:
    package_src = Path(__file__).resolve().parents[1] / "src"
    sys.path.insert(0, str(package_src))
    try:
        from ltx_audio_trainer.config import AudioTrainerConfig  # noqa: PLC0415

        checkpoint_path = Path(__file__).resolve().parents[2] / "ltx-audio-pipeline" / "README.md"
        gemma_root = Path(__file__).resolve().parents[2] / "ltx-audio-core" / "src"

        config = AudioTrainerConfig(
            model={
                "model_path": str(checkpoint_path),
                "text_encoder_path": str(gemma_root),
            },
            data={
                "preprocessed_data_root": "dummy-precomputed-root",
            },
        )

        assert config.model.training_mode == "lora"
        assert config.validation.audio_duration_seconds > 0
        assert config.training_strategy.name == "text_to_audio"
    finally:
        sys.path.pop(0)


def _has_training_strategy_config_alias(module: ast.Module) -> bool:
    for node in module.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "TrainingStrategyConfig"
            for target in node.targets
        ):
            return True
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "TrainingStrategyConfig"
        ):
            return True
    return False
