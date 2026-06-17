"""Preprocess an audio dataset into text features and audio latents."""

from pathlib import Path

import typer
from process_audio import compute_audio_latents
from process_captions import compute_captions_embeddings
from rich.console import Console

from ltx_audio_trainer import logger
from ltx_audio_trainer.gpu_utils import free_gpu_memory_context

console = Console()

app = typer.Typer(
    pretty_exceptions_enable=False,
    no_args_is_help=True,
    help="Preprocess an audio dataset by computing caption features and audio latents.",
)


def preprocess_dataset(  # noqa: PLR0913
    dataset_file: str,
    caption_column: str,
    audio_column: str,
    batch_size: int,
    output_dir: str | None,
    lora_trigger: str | None,
    model_path: str,
    text_encoder_path: str,
    device: str,
    remove_llm_prefixes: bool = False,
    load_text_encoder_in_8bit: bool = False,
    overwrite: bool = False,
    decode: bool = False,
) -> None:
    """Run the audio preprocessing pipeline."""
    _validate_dataset_file(dataset_file)

    output_base = Path(output_dir) if output_dir else Path(dataset_file).parent / ".precomputed"
    conditions_dir = output_base / "conditions"
    audio_latents_dir = output_base / "audio_latents"

    if lora_trigger:
        logger.info(f'LoRA trigger word "{lora_trigger}" will be prepended to all captions')

    with free_gpu_memory_context():
        compute_captions_embeddings(
            dataset_file=dataset_file,
            output_dir=str(conditions_dir),
            model_path=model_path,
            text_encoder_path=text_encoder_path,
            caption_column=caption_column,
            media_column=audio_column,
            lora_trigger=lora_trigger,
            remove_llm_prefixes=remove_llm_prefixes,
            batch_size=batch_size,
            device=device,
            load_in_8bit=load_text_encoder_in_8bit,
            overwrite=overwrite,
        )

    with free_gpu_memory_context():
        compute_audio_latents(
            dataset_file=dataset_file,
            audio_column=audio_column,
            output_dir=str(audio_latents_dir),
            model_path=model_path,
            batch_size=batch_size,
            device=device,
            overwrite=overwrite,
            decode=decode,
        )

    logger.info(f"Audio dataset preprocessing complete! Results saved to {output_base}")


def _validate_dataset_file(dataset_path: str) -> None:
    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {dataset_file}")
    if not dataset_file.is_file():
        raise ValueError(f"Dataset path must be a file, not a directory: {dataset_file}")
    if dataset_file.suffix.lower() not in [".csv", ".json", ".jsonl"]:
        raise ValueError(f"Dataset file must be CSV, JSON, or JSONL format: {dataset_file}")


@app.command()
def main(  # noqa: PLR0913
    dataset_path: str = typer.Argument(
        ...,
        help="Path to metadata file (CSV/JSON/JSONL) containing captions and audio paths",
    ),
    model_path: str = typer.Option(
        ...,
        help="Path to LTX audio checkpoint (.safetensors file)",
    ),
    text_encoder_path: str = typer.Option(
        ...,
        help="Path to Gemma text encoder directory",
    ),
    caption_column: str = typer.Option(
        default="caption",
        help="Column name containing captions in the dataset JSON/JSONL/CSV file",
    ),
    audio_column: str = typer.Option(
        default="audio_path",
        help="Column name containing audio paths in the dataset JSON/JSONL/CSV file",
    ),
    batch_size: int = typer.Option(
        default=1,
        help="Batch size for preprocessing",
    ),
    device: str = typer.Option(
        default="cuda",
        help="Device to use for computation",
    ),
    output_dir: str | None = typer.Option(
        default=None,
        help="Output directory (defaults to .precomputed in dataset directory)",
    ),
    lora_trigger: str | None = typer.Option(
        default=None,
        help="Optional trigger word to prepend to each caption",
    ),
    remove_llm_prefixes: bool = typer.Option(
        default=False,
        help="Remove common LLM-generated prefixes from captions",
    ),
    load_text_encoder_in_8bit: bool = typer.Option(
        default=False,
        help="Load the Gemma text encoder in 8-bit precision to save GPU memory",
    ),
    overwrite: bool = typer.Option(
        default=False,
        help="Re-compute every item even if its output exists",
    ),
    decode: bool = typer.Option(
        default=False,
        help="Decode generated latents after encoding to validate the preprocessing output",
    ),
) -> None:
    preprocess_dataset(
        dataset_file=dataset_path,
        caption_column=caption_column,
        audio_column=audio_column,
        batch_size=batch_size,
        output_dir=output_dir,
        lora_trigger=lora_trigger,
        model_path=model_path,
        text_encoder_path=text_encoder_path,
        device=device,
        remove_llm_prefixes=remove_llm_prefixes,
        load_text_encoder_in_8bit=load_text_encoder_in_8bit,
        overwrite=overwrite,
        decode=decode,
    )


if __name__ == "__main__":
    app()
