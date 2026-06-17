from __future__ import annotations

import argparse
import re
from importlib import import_module
from pathlib import Path

import torch
import torchaudio

from ltx_audio_trainer.model_loader import load_model
from ltx_audio_trainer.progress import StandaloneSamplingProgress
from ltx_audio_trainer.validation_sampler import GenerationConfig, ValidationSampler


def extract_lora_target_modules(state_dict: dict[str, torch.Tensor]) -> list[str]:
    target_modules = set()
    pattern = re.compile(r"(.+)\.lora_[AB]\.")

    for key in state_dict:
        match = pattern.match(key)
        if match:
            target_modules.add(match.group(1))

    return sorted(target_modules)


def load_lora_weights(transformer: torch.nn.Module, lora_path: str | Path) -> torch.nn.Module:
    peft_module = import_module("peft")
    lora_config_cls = peft_module.LoraConfig
    get_peft_model = peft_module.get_peft_model
    set_peft_model_state_dict = peft_module.set_peft_model_state_dict

    lora_path = Path(lora_path)
    if lora_path.suffix == ".safetensors":
        load_file = import_module("safetensors.torch").load_file
        state_dict = load_file(str(lora_path))
    else:
        state_dict = torch.load(lora_path, map_location="cpu", weights_only=False)

    state_dict = {k.replace("diffusion_model.", "", 1): v for k, v in state_dict.items()}
    target_modules = extract_lora_target_modules(state_dict)
    if not target_modules:
        raise ValueError(f"Could not extract target modules from LoRA checkpoint: {lora_path}")

    lora_rank = None
    for key, value in state_dict.items():
        if "lora_A" in key and value.ndim == 2:
            lora_rank = value.shape[0]
            break
    if lora_rank is None:
        raise ValueError(f"Could not detect LoRA rank from checkpoint: {lora_path}")

    lora_config = lora_config_cls(
        r=lora_rank,
        lora_alpha=lora_rank,
        target_modules=target_modules,
        lora_dropout=0.0,
        init_lora_weights=True,
    )
    transformer = get_peft_model(transformer, lora_config)
    base_model = transformer.get_base_model() if hasattr(transformer, "get_base_model") else transformer
    set_peft_model_state_dict(base_model, state_dict)
    return transformer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LTX audio generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the base model checkpoint")
    parser.add_argument("--text-encoder-path", type=str, required=True, help="Path to the Gemma text encoder")
    parser.add_argument("--lora-path", type=str, default=None, help="Optional path to LoRA weights")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for generation")
    parser.add_argument("--negative-prompt", type=str, default="", help="Negative prompt")
    parser.add_argument(
        "--audio-duration-seconds",
        type=float,
        default=10.0,
        help="Target audio duration in seconds",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=50,
        help="Number of denoising steps",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=4.0,
        help="Classifier-free guidance scale",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on")
    parser.add_argument(
        "--load-text-encoder-in-8bit",
        action="store_true",
        help="Load Gemma text encoder in 8-bit mode",
    )
    parser.add_argument("--output", type=str, required=True, help="Output wav path")
    args = parser.parse_args()

    components = load_model(
        checkpoint_path=args.checkpoint,
        text_encoder_path=args.text_encoder_path,
        device="cpu",
        dtype=torch.bfloat16,
        with_audio_vae_encoder=False,
        with_audio_vae_decoder=True,
        with_vocoder=True,
        with_text_encoder=True,
        with_embeddings_processor=True,
        load_text_encoder_in_8bit=args.load_text_encoder_in_8bit,
    )

    transformer = components.transformer
    if transformer is None or components.audio_vae_decoder is None or components.vocoder is None:
        raise ValueError("Model loader did not return the required audio generation components")

    if args.lora_path is not None:
        transformer = load_lora_weights(transformer, args.lora_path)

    config = GenerationConfig(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        audio_duration_seconds=args.audio_duration_seconds,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )

    with StandaloneSamplingProgress(num_steps=args.num_inference_steps) as progress:
        sampler = ValidationSampler(
            transformer=transformer,
            audio_decoder=components.audio_vae_decoder,
            vocoder=components.vocoder,
            text_encoder=components.text_encoder,
            embeddings_processor=components.embeddings_processor,
            sampling_context=progress,
            scheduler=getattr(components, "scheduler", None),
        )
        audio = sampler.generate(config, device=args.device)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    waveform = audio.waveform.detach().cpu()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    torchaudio.save(str(output_path), waveform, audio.sampling_rate)


if __name__ == "__main__":
    main()
