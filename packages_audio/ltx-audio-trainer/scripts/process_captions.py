"""Compute caption features for audio generation training."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar
from uuid import uuid4

import pandas as pd
import torch
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
from torch.utils.data import DataLoader, Subset

from ltx_audio_trainer import logger
from ltx_audio_trainer.model_loader import load_text_conditioning_components

T = TypeVar("T")
console = Console()

app = typer.Typer(
    pretty_exceptions_enable=False,
    no_args_is_help=True,
    help="Process text captions and save feature tensors for audio generation training.",
)


class CaptionsDataset:
    def __init__(
        self,
        dataset_file: str | Path,
        caption_column: str,
        media_column: str = "audio_path",
        lora_trigger: str | None = None,
        remove_llm_prefixes: bool = False,
    ) -> None:
        self.dataset_file = Path(dataset_file)
        self.caption_column = caption_column
        self.media_column = media_column
        self.lora_trigger = f"{lora_trigger.strip()} " if lora_trigger else ""
        self.caption_data = self._load_caption_data()
        self.output_paths = list(self.caption_data.keys())
        self.prompts = list(self.caption_data.values())
        if remove_llm_prefixes:
            self.prompts = [prompt.strip() for prompt in self.prompts]

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "prompt": self.lora_trigger + self.prompts[index],
            "output_path": self.output_paths[index],
            "index": index,
        }

    def _load_caption_data(self) -> dict[str, str]:
        suffix = self.dataset_file.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(self.dataset_file)
            return self._rows_to_caption_data(df.to_dict(orient="records"))
        if suffix == ".json":
            with open(self.dataset_file, "r", encoding="utf-8") as file:
                return self._rows_to_caption_data(json.load(file))
        if suffix == ".jsonl":
            rows = []
            with open(self.dataset_file, "r", encoding="utf-8") as file:
                for line in file:
                    rows.append(json.loads(line))
            return self._rows_to_caption_data(rows)
        raise ValueError("Expected dataset_file to be CSV, JSON, or JSONL")

    def _rows_to_caption_data(self, rows: list[dict[str, Any]]) -> dict[str, str]:
        caption_data: dict[str, str] = {}
        for row in rows:
            if self.caption_column not in row:
                raise ValueError(f"Key '{self.caption_column}' not found in metadata row: {row}")
            if self.media_column not in row:
                raise ValueError(f"Key '{self.media_column}' not found in metadata row: {row}")

            media_path = Path(str(row[self.media_column]).strip())
            output_path = str(media_path.with_suffix(".pt"))
            caption_data[output_path] = str(row[self.caption_column])
        return caption_data


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
    dataset: CaptionsDataset,
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


def _append_failure_record(manifest_path: Path, record: dict[str, Any]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


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


def compute_captions_embeddings(  # noqa: PLR0913
    dataset_file: str | Path,
    output_dir: str,
    model_path: str,
    text_encoder_path: str,
    caption_column: str = "caption",
    media_column: str = "audio_path",
    lora_trigger: str | None = None,
    remove_llm_prefixes: bool = False,
    batch_size: int = 8,
    device: str = "cuda",
    load_in_8bit: bool = False,
    overwrite: bool = False,
) -> None:
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    dataset = CaptionsDataset(
        dataset_file=dataset_file,
        caption_column=caption_column,
        media_column=media_column,
        lora_trigger=lora_trigger,
        remove_llm_prefixes=remove_llm_prefixes,
    )

    text_stack = load_text_conditioning_components(
        checkpoint_path=model_path,
        text_encoder_path=text_encoder_path,
        device=device,
        with_text_encoder=True,
        with_embeddings_processor=True,
        load_text_encoder_in_8bit=load_in_8bit,
    )
    if text_stack is None:
        raise ValueError("Failed to load text conditioning components")

    text_encoder = text_stack.require_text_encoder()
    embeddings_processor = text_stack.require_embeddings_processor()

    dataloader = _build_sharded_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=0,
        is_done=lambda index: (output_dir_path / dataset[index]["output_path"]).exists(),
        overwrite=overwrite,
    )
    if dataloader is None:
        logger.info(f"No pending caption features for {output_dir_path}")
        return

    failure_manifest = output_dir_path / "_caption_failures.jsonl"
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
        task = progress.add_task("Processing captions", total=len(dataloader))
        for batch in dataloader:
            prompts = list(batch["prompt"])
            output_paths = list(batch["output_path"])
            for prompt, relative_output_path in zip(prompts, output_paths, strict=True):
                output_path = output_dir_path / relative_output_path
                if output_path.exists() and not overwrite:
                    skipped_count += 1
                    continue

                def _encode_caption(prompt_text: str = prompt) -> dict[str, Any]:
                    hidden_states, attention_mask = text_encoder.encode(prompt_text)
                    processed = embeddings_processor.process_hidden_states(hidden_states, attention_mask)
                    return {
                        "video_prompt_embeds": processed.video_encoding.squeeze(0).cpu(),
                        "audio_prompt_embeds": processed.audio_encoding.squeeze(0).cpu()
                        if processed.audio_encoding is not None
                        else None,
                        "prompt_attention_mask": processed.attention_mask.squeeze(0).cpu(),
                    }

                try:
                    payload = _retry(_encode_caption, f"Encoding caption for {output_path}")
                except Exception as error:
                    failed_count += 1
                    _append_failure_record(
                        failure_manifest,
                        {
                            "input_path": str(relative_output_path),
                            "output_path": str(output_path),
                            "error": str(error),
                        },
                    )
                    logger.warning(f"Failed to encode caption for {output_path}: {error}")
                    continue

                _atomic_save(payload, output_path)
                processed_count += 1

            progress.advance(task)

    _log_processing_summary(
        modality="caption features",
        processed=processed_count,
        skipped=skipped_count,
        failed=failed_count,
        output_dir=output_dir_path,
        failure_manifest=failure_manifest if failed_count > 0 else None,
    )


@app.command()
def main(  # noqa: PLR0913
    dataset_path: str = typer.Argument(...),
    output_dir: str = typer.Option(...),
    model_path: str = typer.Option(...),
    text_encoder_path: str = typer.Option(...),
    caption_column: str = typer.Option(default="caption"),
    media_column: str = typer.Option(default="audio_path"),
    lora_trigger: str | None = typer.Option(default=None),
    remove_llm_prefixes: bool = typer.Option(default=False),
    batch_size: int = typer.Option(default=8),
    device: str = typer.Option(default="cuda"),
    load_in_8bit: bool = typer.Option(default=False),
    overwrite: bool = typer.Option(default=False),
) -> None:
    compute_captions_embeddings(
        dataset_file=dataset_path,
        output_dir=output_dir,
        model_path=model_path,
        text_encoder_path=text_encoder_path,
        caption_column=caption_column,
        media_column=media_column,
        lora_trigger=lora_trigger,
        remove_llm_prefixes=remove_llm_prefixes,
        batch_size=batch_size,
        device=device,
        load_in_8bit=load_in_8bit,
        overwrite=overwrite,
    )


if __name__ == "__main__":
    app()
