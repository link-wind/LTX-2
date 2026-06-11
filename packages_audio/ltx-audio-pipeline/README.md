# LTX-Audio-Pipeline

Audio-only pipeline utilities for LTX-2.

This package mirrors the structure of `ltx-pipelines` where practical so the
audio-only inference path can be merged back into the main pipeline tree with
minimal divergence.

## Current Surface

The current package provides:

- `AudioOneStagePipeline` for prompt-to-audio generation.
- Shared building blocks such as `DiffusionStage`, `PromptEncoder`,
  `AudioConditioner`, `AudioDecoder`, denoisers, samplers, and quantization
  helpers.
- A CLI entrypoint at `python -m ltx_audio_pipeline.audio_one_stage`.

## Quick Start

```bash
python -m ltx_audio_pipeline.audio_one_stage \
    --checkpoint-path path/to/audio_model.safetensors \
    --gemma-root path/to/gemma \
    --prompt "Warm lo-fi piano with soft vinyl crackle and slow brushed drums" \
    --output-path outputs/demo.wav
```

Useful options:

- `--negative-prompt` overrides the default audio artifact suppression prompt.
- `--lora PATH [STRENGTH]` can be passed multiple times.
- `--quantization fp8_scaled` or another supported quantization preset.
- `--offload cpu` or `--offload disk` enables layer streaming for lower-memory inference.
- `--compile` enables `torch.compile` for the diffusion transformer.

Example with extra runtime controls:

```bash
python -m ltx_audio_pipeline.audio_one_stage \
    --checkpoint-path path/to/audio_model.safetensors \
    --gemma-root path/to/gemma \
    --prompt "Minimal techno groove, dry kick, rubbery bassline, tight hats" \
    --negative-prompt "muddy mix, clipping, harsh highs" \
    --output-path outputs/techno.wav \
    --num-inference-steps 30 \
    --audio-cfg-guidance-scale 7.0 \
    --v2a-guidance-scale 3.0 \
    --offload cpu
```

## Notes And Limits

- This package currently targets `audio-only` inference. It keeps the AV-shaped
  internals where possible, but the public generation path here is audio-only.
- `--offload` and `--compile` are mutually exclusive.
- Prompt enhancement currently supports text-only rewriting. Passing an image to
  `generate_enhanced_prompt(...)` is not supported yet.
- Checkpoint defaults are inferred from checkpoint metadata when available.
- The package has surface tests and module-level verification, but it has not
  yet been validated end-to-end against a real production checkpoint in this
  package README flow.
