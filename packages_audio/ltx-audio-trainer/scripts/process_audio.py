"""Compute audio latents for audio generation training."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar
from uuid import uuid4

import pandas as pd
import torch
import torchaudio
import typer
from accelerate import PartialState
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from torch.utils.data import DataLoader, Dataset, Subset

from ltx_audio_core.model.audio_vae.audio_vae import decode_audio, encode_audio
from ltx_audio_core.types import Audio
from ltx_audio_trainer import logger
from ltx_audio_trainer.model_loader import load_audio_vae_decoder, load_preprocess_components, load_vocoder

if TYPE_CHECKING:
    from ltx_audio_core.model.audio_vae.audio_vae import AudioDecoder
    from ltx_audio_core.model.audio_vae.vocoder import Vocoder

T = TypeVar("T")
console = Console()

app = typer.Typer(
    pretty_exceptions_enable=False,
    no_args_is_help=True,
    help="Process audio files and save latent tensors for audio generation training.",
)


class AudioDataset(Dataset):
    def __init__(self, dataset_file: str | Path, audio_column: str = "audio_path") -> None:
        self.dataset_file = Path(dataset_file)
        self.audio_column = audio_column
        self.audio_paths = self._load_audio_paths()

    def __len__(self) -> int:
        return len(self.audio_paths)

    def __getitem__(self, index: int) -> dict[str, str]:
        data_root = self.dataset_file.parent
        audio_path = self.audio_paths[index]
        relative_path = audio_path.relative_to(data_root)
        return {
            "audio_path": str(audio_path),
            "relative_path": str(relative_path),
        }

    def _load_audio_paths(self) -> list[Path]:
        suffix = self.dataset_file.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(self.dataset_file)
            rows = df.to_dict(orient="records")
        elif suffix == ".json":
            with open(self.dataset_file, "r", encoding="utf-8") as file:
                rows = json.load(file)
        elif suffix == ".jsonl":
            rows = []
            with open(self.dataset_file, "r", encoding="utf-8") as file:
                for line in file:
                    rows.append(json.loads(line))
        else:
            raise ValueError("Expected dataset_file to be CSV, JSON, or JSONL")

        data_root = self.dataset_file.parent
        audio_paths: list[Path] = []
        for row in rows:
            if self.audio_column not in row:
                raise ValueError(f"Key '{self.audio_column}' not found in metadata row: {row}")
            audio_path = data_root / Path(str(row[self.audio_column]).strip())
            audio_paths.append(audio_path)
        return audio_paths


def _atomic_save(data: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.name}.tmp.{uuid4().hex}")
    try:
        torch.save(data, tmp_path)
        tmp_path.replace(output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _build_sharded_dataloader(
    dataset: AudioDataset,
    batch_size: int,
    num_workers: int,
    is_done: Callable[[int], bool],
    overwrite: bool,
) -> DataLoader | None:
    state = PartialState()
    shard_indices = list(range(state.process_index, len(dataset), state.num_processes))
    indices = [index for index in shard_indices if overwrite or not is_done(index)]
    if not indices:
        return None
    dataloader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    dataloader.ltx_skipped_count = len(shard_indices) - len(indices)
    return dataloader


def _retry(operation: Callable[[], T], description: str, max_attempts: int = 3) -> T:
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except Exception as error:
            last_error = error
            if attempt == max_attempts:
                break
            logger.warning(f"{description} failed on attempt {attempt}/{max_attempts}: {error}")
    assert last_error is not None
    raise last_error


def _decode_and_validate(latent: torch.Tensor, audio_decoder: "AudioDecoder", vocoder: "Vocoder") -> Audio:
    decode_input = latent.unsqueeze(0) if latent.ndim == 3 else latent
    decoded_audio = decode_audio(decode_input, audio_decoder, vocoder)
    waveform = decoded_audio.waveform
    if waveform.numel() == 0:
        raise ValueError("Decoded audio is empty")
    if not torch.isfinite(waveform).all():
        raise ValueError("Decoded audio contains non-finite values")
    return decoded_audio


def _append_failure_record(manifest_path: Path, record: dict[str, Any]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _save_decode_failure_artifact(
    latent: torch.Tensor,
    output_dir: Path,
    relative_path: Path,
    error_message: str,
) -> Path:
    artifact_path = (output_dir / "_decode_failures" / relative_path).with_suffix(".pt")
    _atomic_save(
        {
            "latents": latent.cpu(),
            "error": error_message,
            "relative_path": str(relative_path),
        },
        artifact_path,
    )
    return artifact_path


def _log_processing_summary(
    modality: str,
    processed: int,
    skipped: int,
    failed: int,
    output_dir: Path,
    failure_manifest: Path | None = None,
) -> None:
    logger.info(f"{modality} summary: processed={processed}, skipped={skipped}, output_dir={output_dir}")
    if failed > 0:
        logger.warning(f"{modality} summary: failed={failed}, failure_manifest={failure_manifest}")


def compute_audio_latents(
    dataset_file: str | Path,
    audio_column: str,
    output_dir: str,
    model_path: str,
    batch_size: int = 1,
    device: str = "cuda",
    overwrite: bool = False,
    decode: bool = False,
) -> None:
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    dataset = AudioDataset(dataset_file=dataset_file, audio_column=audio_column)
    components = load_preprocess_components(
        checkpoint_path=model_path,
        text_encoder_path=None,
        device=device,
        with_text_encoder=False,
        with_embeddings_processor=False,
    )
    if components.audio_vae_encoder is None:
        raise ValueError("Audio VAE encoder is required for audio preprocessing")
    audio_encoder = components.audio_vae_encoder
    audio_decoder = load_audio_vae_decoder(model_path, device=device) if decode else None
    vocoder = load_vocoder(model_path, device=device) if decode else None

    dataloader = _build_sharded_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=0,
        is_done=lambda index: (output_dir_path / Path(dataset[index]["relative_path"])).with_suffix(".pt").exists(),
        overwrite=overwrite,
    )
    if dataloader is None:
        logger.info(f"No pending audio latents for {output_dir_path}")
        return

    failure_manifest = output_dir_path / "_audio_failures.jsonl"
    processed_count = 0
    skipped_count = int(getattr(dataloader, "ltx_skipped_count", 0))
    failed_count = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing audio", total=len(dataloader))
        for batch in dataloader:
            audio_paths = list(batch["audio_path"])
            relative_paths = list(batch["relative_path"])
            for audio_path_str, relative_path_str in zip(audio_paths, relative_paths, strict=True):
                audio_path = Path(audio_path_str)
                relative_path = Path(relative_path_str)
                output_path = (output_dir_path / relative_path).with_suffix(".pt")
                if output_path.exists() and not overwrite:
                    skipped_count += 1
                    continue

                latent_holder: dict[str, torch.Tensor] = {}

                def _encode_audio_item(
                    audio_file: Path = audio_path,
                    latent_store: dict[str, torch.Tensor] = latent_holder,
                ) -> dict[str, Any]:
                    waveform, sample_rate = torchaudio.load(str(audio_file))
                    latent = encode_audio(
                        Audio(
                            waveform=waveform.unsqueeze(0),
                            sampling_rate=sample_rate,
                        ),
                        audio_encoder=audio_encoder,
                    ).squeeze(0)
                    latent_store["latent"] = latent

                    if decode:
                        if audio_decoder is None or vocoder is None:
                            raise ValueError("Audio decoder and vocoder are required when decode=True")
                        _decode_and_validate(latent, audio_decoder, vocoder)

                    return {
                        "latents": latent.cpu(),
                        "num_frames": latent.shape[1],
                        "mel_bins": latent.shape[2],
                        "latent_channels": latent.shape[0],
                    }

                try:
                    payload = _retry(_encode_audio_item, f"Encoding audio for {audio_path}")
                except Exception as error:
                    failed_count += 1
                    artifact_path = None
                    if decode and "latent" in latent_holder:
                        artifact_path = _save_decode_failure_artifact(
                            latent_holder["latent"],
                            output_dir=output_dir_path,
                            relative_path=relative_path,
                            error_message=str(error),
                        )
                    _append_failure_record(
                        failure_manifest,
                        {
                            "input_path": str(audio_path),
                            "output_path": str(output_path),
                            "relative_path": str(relative_path),
                            "error": str(error),
                            "decode_artifact": str(artifact_path) if artifact_path is not None else None,
                        },
                    )
                    logger.warning(f"Failed to encode audio for {audio_path}: {error}")
                    continue

                _atomic_save(payload, output_path)
                processed_count += 1

            progress.advance(task)

    _log_processing_summary(
        modality="audio latents",
        processed=processed_count,
        skipped=skipped_count,
        failed=failed_count,
        output_dir=output_dir_path,
        failure_manifest=failure_manifest if failed_count > 0 else None,
    )


@app.command()
def main(
    dataset_path: str = typer.Argument(...),
    output_dir: str = typer.Option(...),
    model_path: str = typer.Option(...),
    audio_column: str = typer.Option(default="audio_path"),
    batch_size: int = typer.Option(default=1),
    device: str = typer.Option(default="cuda"),
    overwrite: bool = typer.Option(default=False),
    decode: bool = typer.Option(default=False),
) -> None:
    compute_audio_latents(
        dataset_file=dataset_path,
        audio_column=audio_column,
        output_dir=output_dir,
        model_path=model_path,
        batch_size=batch_size,
        device=device,
        overwrite=overwrite,
        decode=decode,
    )


if __name__ == "__main__":
    app()
