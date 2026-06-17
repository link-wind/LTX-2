"""Display utilities for audio trainer configuration."""

from rich import box
from rich.console import Console
from rich.table import Table

from ltx_audio_trainer.config import AudioTrainerConfig


def print_config(config: AudioTrainerConfig) -> None:
    """Print configuration as a compact sectioned table."""

    def fmt(value: object, max_len: int = 55) -> str:
        if value is None:
            return "[dim]—[/]"
        if isinstance(value, bool):
            return "[green]✓[/]" if value else "[dim]✗[/]"
        if isinstance(value, (list, tuple)):
            if not value:
                return "[dim]—[/]"
            return ", ".join(str(item) for item in value)
        rendered = str(value)
        return rendered[: max_len - 3] + "..." if len(rendered) > max_len else rendered

    cfg = config
    opt = cfg.optimization
    val = cfg.validation
    accel = cfg.acceleration

    sections: list[tuple[str, list[tuple[str, str]]]] = [
        (
            "🔊 Model",
            [
                ("Base", fmt(cfg.model.model_path)),
                ("Text Encoder", fmt(cfg.model.text_encoder_path)),
                ("Training Mode", f"[bold green]{cfg.model.training_mode.upper()}[/]"),
                ("Load Checkpoint", fmt(cfg.model.load_checkpoint)),
            ],
        ),
    ]

    if cfg.lora:
        sections.append(
            (
                "🔗 LoRA",
                [
                    ("Rank / Alpha", f"{cfg.lora.rank} / {cfg.lora.alpha}"),
                    ("Dropout", str(cfg.lora.dropout)),
                    ("Target Modules", fmt(cfg.lora.target_modules)),
                ],
            )
        )

    strategy_items = [("Name", cfg.training_strategy.name)]
    if hasattr(cfg.training_strategy, "with_text_conditioning_dropout"):
        strategy_items.append(
            ("Text Cond Dropout", fmt(cfg.training_strategy.with_text_conditioning_dropout)),
        )
    if hasattr(cfg.training_strategy, "text_conditioning_dropout_p"):
        strategy_items.append(
            ("Text Dropout P", str(cfg.training_strategy.text_conditioning_dropout_p)),
        )
    sections.append(("🎯 Strategy", strategy_items))

    sections.extend(
        [
            (
                "⚡ Optimization",
                [
                    ("Steps", f"[bold]{opt.steps:,}[/]"),
                    ("Learning Rate", f"{opt.learning_rate:.2e}"),
                    ("Batch Size", str(opt.batch_size)),
                    ("Grad Accumulation", str(opt.gradient_accumulation_steps)),
                    ("Optimizer", opt.optimizer_type),
                    ("Scheduler", opt.scheduler_type),
                    ("Max Grad Norm", str(opt.max_grad_norm)),
                    ("Grad Checkpointing", fmt(opt.enable_gradient_checkpointing)),
                ],
            ),
            (
                "🚀 Acceleration",
                [
                    ("Mixed Precision", accel.mixed_precision_mode or "[dim]—[/]"),
                    ("Quantization", str(accel.quantization) if accel.quantization else "[dim]—[/]"),
                    ("Text Encoder 8bit", fmt(accel.load_text_encoder_in_8bit)),
                    ("Optimizer CPU Offload", fmt(accel.offload_optimizer_during_validation)),
                ],
            ),
            (
                "🎧 Validation",
                [
                    ("Prompts", f"{len(val.prompts)} prompt(s)" if val.prompts else "[dim]—[/]"),
                    ("Interval", f"Every {val.interval} steps" if val.interval else "[dim]Disabled[/]"),
                    ("Audio Duration", f"{val.audio_duration_seconds:.2f}s"),
                    ("Inference Steps", str(val.inference_steps)),
                    ("CFG Scale", str(val.guidance_scale)),
                    ("Negative Prompt", fmt(val.negative_prompt)),
                    ("Seed", str(val.seed)),
                ],
            ),
            (
                "📂 Data & Output",
                [
                    ("Dataset", fmt(cfg.data.preprocessed_data_root)),
                    ("Dataloader Workers", str(cfg.data.num_dataloader_workers)),
                    ("Output Dir", fmt(cfg.output_dir)),
                    ("Seed", str(cfg.seed)),
                ],
            ),
            (
                "🔌 Integrations",
                [
                    (
                        "Checkpoints",
                        f"Every {cfg.checkpoints.interval} steps (keep {cfg.checkpoints.keep_last_n})"
                        if cfg.checkpoints.interval
                        else "[dim]Disabled[/]",
                    ),
                    ("W&B", cfg.wandb.project if cfg.wandb.enabled else "[dim]Disabled[/]"),
                    ("HF Hub", cfg.hub.hub_model_id if cfg.hub.push_to_hub else "[dim]Disabled[/]"),
                ],
            ),
        ]
    )

    table = Table(
        title="[bold]⚙️  Audio Training Configuration[/]",
        show_header=False,
        box=box.ROUNDED,
        border_style="bright_blue",
        padding=(0, 1),
        title_style="bold bright_blue",
    )
    table.add_column("Key", style="white", width=20)
    table.add_column("Value", style="cyan")

    for index, (section_title, items) in enumerate(sections):
        if index > 0:
            table.add_row("", "")
        table.add_row(f"[bold yellow]{section_title}[/]", "")
        for key, value in items:
            table.add_row(f"  {key}", value)

    console = Console()
    console.print()
    console.print(table)
    console.print()
