from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset

from ltx_audio_trainer import logger

PRECOMPUTED_DIR_NAME = ".precomputed"


class DummyDataset(Dataset):
    """Produce random audio latents and prompt embeddings for smoke tests."""

    def __init__(
        self,
        dataset_length: int = 200,
        latent_channels: int = 8,
        latent_frames: int = 64,
        mel_bins: int = 16,
        prompt_embed_dim: int = 2048,
        prompt_sequence_length: int = 256,
    ) -> None:
        self.dataset_length = dataset_length
        self.latent_channels = latent_channels
        self.latent_frames = latent_frames
        self.mel_bins = mel_bins
        self.prompt_embed_dim = prompt_embed_dim
        self.prompt_sequence_length = prompt_sequence_length

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> dict[str, dict[str, Tensor] | int]:
        return {
            "latent_conditions": {
                "latents": torch.randn(
                    self.latent_channels,
                    self.latent_frames,
                    self.mel_bins,
                ),
                "num_frames": self.latent_frames,
                "mel_bins": self.mel_bins,
            },
            "text_conditions": {
                "video_prompt_embeds": torch.randn(
                    self.prompt_sequence_length,
                    self.prompt_embed_dim,
                ),
                "audio_prompt_embeds": torch.randn(
                    self.prompt_sequence_length,
                    self.prompt_embed_dim,
                ),
                "prompt_attention_mask": torch.ones(
                    self.prompt_sequence_length,
                    dtype=torch.bool,
                ),
            },
            "idx": idx,
        }


class PrecomputedDataset(Dataset):
    def __init__(self, data_root: str, data_sources: dict[str, str] | list[str] | None = None) -> None:
        super().__init__()
        self.data_root = self._setup_data_root(data_root)
        self.data_sources = self._normalize_data_sources(data_sources)
        self.source_paths = self._setup_source_paths()
        self.sample_files = self._discover_samples()
        self._validate_setup()

    @staticmethod
    def _setup_data_root(data_root: str) -> Path:
        root = Path(data_root).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"Data root directory does not exist: {root}")
        if (root / PRECOMPUTED_DIR_NAME).exists():
            root = root / PRECOMPUTED_DIR_NAME
        return root

    @staticmethod
    def _normalize_data_sources(data_sources: dict[str, str] | list[str] | None) -> dict[str, str]:
        if data_sources is None:
            return {"audio_latents": "latent_conditions", "conditions": "text_conditions"}
        if isinstance(data_sources, list):
            return {source: source for source in data_sources}
        if isinstance(data_sources, dict):
            return data_sources.copy()
        raise TypeError(f"data_sources must be dict, list, or None, got {type(data_sources)}")

    def _setup_source_paths(self) -> dict[str, Path]:
        source_paths: dict[str, Path] = {}
        for dir_name in self.data_sources:
            source_path = self.data_root / dir_name
            source_paths[dir_name] = source_path
            if not source_path.exists():
                raise FileNotFoundError(f"Required {dir_name} directory does not exist: {source_path}")
        return source_paths

    def _discover_samples(self) -> dict[str, list[Path]]:
        if not self.data_sources:
            raise ValueError("No data sources configured")

        data_key = "audio_latents" if "audio_latents" in self.data_sources else next(iter(self.data_sources))
        data_path = self.source_paths[data_key]

        def _glob_source(dir_name: str) -> tuple[list[Path], set[str]]:
            source_path = self.source_paths[dir_name]
            paths = list(source_path.glob("**/*.pt"))
            path_set = {str(path) for path in paths}
            return paths, path_set

        with ThreadPoolExecutor(max_workers=len(self.data_sources)) as executor:
            glob_results = dict(
                zip(
                    self.data_sources.keys(),
                    executor.map(_glob_source, self.data_sources.keys()),
                    strict=True,
                )
            )

        data_files, _ = glob_results[data_key]
        if not data_files:
            raise ValueError(f"No data files found in {data_path}")
        data_files.sort()

        other_path_sets = {
            dir_name: path_set for dir_name, (_, path_set) in glob_results.items() if dir_name != data_key
        }

        sample_files: dict[str, list[Path]] = {output_key: [] for output_key in self.data_sources.values()}
        valid_count = 0

        for data_file in data_files:
            rel_path = data_file.relative_to(data_path)
            all_exist = True

            for dir_name, path_set in other_path_sets.items():
                expected = self._get_expected_file_path(dir_name, data_file, rel_path)
                if str(expected) not in path_set:
                    logger.debug(f"Skipping {data_file.name}: no matching {dir_name} file at {expected}")
                    all_exist = False
                    break

            if all_exist:
                self._fill_sample_data_files(data_file, rel_path, sample_files)
                valid_count += 1

        skipped = len(data_files) - valid_count
        if skipped > 0:
            logger.info(f"Fast index: {valid_count} valid samples from {len(data_files)} total ({skipped} skipped)")

        return sample_files

    def _get_expected_file_path(self, dir_name: str, data_file: Path, rel_path: Path) -> Path:
        source_path = self.source_paths[dir_name]
        if dir_name == "conditions" and data_file.name.startswith("latent_"):
            return source_path / f"condition_{data_file.stem[7:]}.pt"
        return source_path / rel_path

    def _fill_sample_data_files(self, data_file: Path, rel_path: Path, sample_files: dict[str, list[Path]]) -> None:
        for dir_name, output_key in self.data_sources.items():
            expected_path = self._get_expected_file_path(dir_name, data_file, rel_path)
            sample_files[output_key].append(expected_path.relative_to(self.source_paths[dir_name]))

    def _validate_setup(self) -> None:
        sample_counts = {key: len(files) for key, files in self.sample_files.items()}
        if not sample_counts or all(count == 0 for count in sample_counts.values()):
            raise ValueError(
                f"No valid samples found in {self.data_root} - all configured data sources "
                f"({list(self.data_sources)}) must have matching files (per-source counts: {sample_counts})"
            )
        if len(set(sample_counts.values())) > 1:
            raise ValueError(f"Mismatched sample counts across sources: {sample_counts}")

    def __len__(self) -> int:
        first_key = next(iter(self.sample_files.keys()))
        return len(self.sample_files[first_key])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | dict | int]:
        result: dict[str, torch.Tensor | dict | int] = {}

        for dir_name, output_key in self.data_sources.items():
            source_path = self.source_paths[dir_name]
            file_rel_path = self.sample_files[output_key][index]
            file_path = source_path / file_rel_path

            try:
                data = torch.load(file_path, map_location="cpu", weights_only=True)
                if "latent" in dir_name.lower():
                    data = self._normalize_audio_latents(data)
                result[output_key] = data
            except Exception as exc:
                raise RuntimeError(f"Failed to load {output_key} from {file_path}: {exc}") from exc

        result["idx"] = index
        return result

    @staticmethod
    def _normalize_audio_latents(data: dict) -> dict:
        latents = data["latents"]
        if latents.dim() != 2:
            return data

        num_frames = data.get("num_frames", latents.shape[0])
        latent_channels = data.get("latent_channels", 8)
        mel_bins = data.get("mel_bins", 16)

        if latents.shape[0] != num_frames:
            raise ValueError(f"Legacy audio latent frame count mismatch: {latents.shape[0]} != {num_frames}")
        if latents.shape[1] != latent_channels * mel_bins:
            raise ValueError(
                "Legacy audio latent feature dimension mismatch: "
                f"{latents.shape[1]} != {latent_channels} * {mel_bins}"
            )

        unpatchified = latents.reshape(num_frames, latent_channels, mel_bins).permute(1, 0, 2).contiguous()
        normalized = data.copy()
        normalized["latents"] = unpatchified
        return normalized
