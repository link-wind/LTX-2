from __future__ import annotations

from pathlib import Path

import typer
import yaml

from ltx_audio_trainer.config import AudioTrainerConfig
from ltx_audio_trainer.trainer import AudioTrainer

app = typer.Typer(
    pretty_exceptions_enable=False,
    no_args_is_help=True,
    help="Train the audio-only LTX transformer using a YAML configuration file.",
)


@app.command()
def main(
    config_path: str = typer.Argument(..., help="Path to the YAML configuration file"),
    disable_progress_bars: bool = typer.Option(
        False,
        "--disable-progress-bars",
        help="Disable Rich progress bars, useful for multi-process runs.",
    ),
) -> None:
    config_file = Path(config_path)
    if not config_file.exists():
        raise typer.BadParameter(f"Configuration file does not exist: {config_file}")

    with config_file.open("r", encoding="utf-8") as file:
        config_data = yaml.safe_load(file)

    trainer_config = AudioTrainerConfig(**config_data)
    trainer = AudioTrainer(trainer_config)
    trainer.train(disable_progress_bars=disable_progress_bars)


if __name__ == "__main__":
    app()
