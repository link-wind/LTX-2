from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import are_progress_bars_disabled, disable_progress_bars, enable_progress_bars
from rich.progress import Progress, SpinnerColumn, TextColumn

from ltx_audio_trainer import logger
from ltx_audio_trainer.config import AudioTrainerConfig


def push_to_hub(
    weights_path: Path,
    sampled_audio_paths: list[Path] | None,
    config: AudioTrainerConfig,
) -> None:
    """Push the trained weights and validation audio samples to HuggingFace Hub."""
    if not config.hub.hub_model_id:
        logger.warning("⚠️ HuggingFace hub_model_id not specified, skipping push to hub")
        return

    api = HfApi()
    original_progress_state = are_progress_bars_disabled()
    disable_progress_bars()

    try:
        try:
            repo = create_repo(
                repo_id=config.hub.hub_model_id,
                repo_type="model",
                exist_ok=True,
            )
            repo_id = repo.repo_id
            logger.info(f"🤗 Successfully created HuggingFace model repository at: {repo.url}")
        except Exception as exc:
            logger.error(f"❌ Failed to create HuggingFace model repository: {exc}")
            return

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                try:
                    task_copy = progress.add_task("Copying weights...", total=None)
                    weights_dest = temp_path / weights_path.name
                    shutil.copy2(weights_path, weights_dest)
                    progress.update(task_copy, description="✓ Weights copied")

                    task_card = progress.add_task("Creating model card and audio samples...", total=None)
                    _create_model_card(
                        output_dir=temp_path,
                        sampled_audio_paths=sampled_audio_paths,
                        config=config,
                    )
                    progress.update(task_card, description="✓ Model card and audio samples created")

                    task_upload = progress.add_task("Pushing files to HuggingFace Hub...", total=None)
                    api.upload_folder(
                        folder_path=str(temp_path),
                        repo_id=repo_id,
                        repo_type="model",
                    )
                    progress.update(task_upload, description="✓ Files pushed to HuggingFace Hub")
                    logger.info("✅ Successfully pushed files to HuggingFace Hub")
                except Exception as exc:
                    logger.error(f"❌ Failed to process and push files to HuggingFace Hub: {exc}")
                    raise
    finally:
        if not original_progress_state:
            enable_progress_bars()


def _copy_audio_samples(output_dir: str | Path, sampled_audio_paths: list[Path] | None) -> list[Path]:
    """Copy validation audio samples into the temp output directory."""
    if not sampled_audio_paths:
        return []

    output_dir = Path(output_dir)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(exist_ok=True, parents=True)

    copied_paths: list[Path] = []
    for index, audio_path in enumerate(sampled_audio_paths):
        if not audio_path.exists():
            logger.warning(f"Skipping missing validation audio sample: {audio_path}")
            continue

        suffix = audio_path.suffix if audio_path.suffix else ".wav"
        dest_path = samples_dir / f"sample_{index}{suffix}"
        shutil.copy2(audio_path, dest_path)
        copied_paths.append(dest_path)

    return copied_paths


def _create_model_card(
    output_dir: str | Path,
    sampled_audio_paths: list[Path] | None,
    config: AudioTrainerConfig,
) -> Path:
    """Generate and save an audio-focused model card for the trained model."""
    repo_id = config.hub.hub_model_id or "unknown/ltx-audio-model"
    pretrained_model_name_or_path = config.model.model_path
    validation_prompts = config.validation.prompts
    output_dir = Path(output_dir)
    template_path = Path(__file__).parent.parent.parent / "templates" / "model_card.md"
    template = template_path.read_text(encoding="utf-8")

    model_name = repo_id.split("/")[-1]
    base_model_link = str(pretrained_model_name_or_path)
    model_path_str = str(pretrained_model_name_or_path)
    is_url = model_path_str.startswith(("http://", "https://"))
    base_model_name = model_path_str.split("/")[-1] if is_url else Path(pretrained_model_name_or_path).name

    copied_audio_paths = _copy_audio_samples(output_dir, sampled_audio_paths)
    sample_entries: list[str] = []
    prompts_text = ""

    if validation_prompts and copied_audio_paths:
        prompts_text = "Example prompts used during validation:\n\n"
        for index, (prompt, sample_path) in enumerate(zip(validation_prompts, copied_audio_paths, strict=False)):
            prompts_text += f"- `{prompt}`\n"
            sample_entries.append(
                "<details>"
                f"<summary>Sample {index + 1}</summary>\n\n"
                f"<audio controls src=\"./samples/{sample_path.name}\"></audio>\n\n"
                f"Prompt: `{prompt}`\n"
                "</details>"
            )

    model_card_content = template.format(
        base_model=base_model_name,
        base_model_link=base_model_link,
        model_name=model_name,
        training_type="LoRA fine-tuning" if config.model.training_mode == "lora" else "Full model fine-tuning",
        training_steps=config.optimization.steps,
        learning_rate=config.optimization.learning_rate,
        batch_size=config.optimization.batch_size,
        validation_prompts=prompts_text,
        sample_entries="\n\n".join(sample_entries) if sample_entries else "No validation audio samples were uploaded.",
    )

    model_card_path = output_dir / "README.md"
    model_card_path.write_text(model_card_content, encoding="utf-8")
    return model_card_path
