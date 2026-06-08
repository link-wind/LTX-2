# LTX Audio Single-Stream Notebook Design

> Date: 2026-06-08
> Topic: Audio-only LTX learning notebook
> Status: Draft for review

## Goal

Add a Jupyter notebook inside `packages/ltx-core/` that teaches the audio-only single-stream path of the LTX transformer by directly importing and calling real repository code.

The notebook should help a reader:

- understand the minimal audio-only execution path,
- map each step back to the real source files,
- run the example without needing checkpoints, datasets, or real audio files,
- inspect intermediate tensors produced by the actual `ltx_core` implementation.

## Scope

This work adds one teaching-oriented notebook and no production model changes.

In scope:

- one `.ipynb` notebook under `packages/ltx-core/`,
- direct imports from real `ltx_core` modules,
- a tiny `AudioOnly` model configuration for fast execution,
- synthetic inputs that satisfy the real model interfaces,
- step-by-step execution of `prepare()`, one block forward, and full `LTXModel.forward()`.

Out of scope:

- training code,
- checkpoint loading,
- real audio preprocessing,
- VAE or vocoder demos,
- new helper library modules unless implementation uncovers an unavoidable need.

## Recommended Approach

Use a single self-contained notebook that imports the real `ltx_core` modules directly and keeps all teaching logic in notebook cells.

Why this approach:

- It matches the learning goal best: one file, one flow, no indirection.
- It keeps source mapping explicit by calling the real functions the reader will inspect in the package.
- It avoids extra helper modules that would dilute the “read notebook, jump to source” experience.

Alternatives considered:

1. Notebook plus helper module
   - Cleaner cells, but worse for source-oriented learning because the reader must chase custom wrapper code.

2. Full-model demo only
   - Easier to run, but not granular enough for understanding the architecture step by step.

## Location

Create the notebook at:

`packages/ltx-core/notebooks/ltx_audio_single_stream_walkthrough.ipynb`

Rationale:

- `ltx-core` is where the transformer, preprocessors, and modality definitions live.
- The notebook belongs closest to the source it explains.
- A `notebooks/` directory under the package gives a discoverable home for future learning artifacts.

## Notebook Structure

The notebook will be organized into the following sections.

### 1. Environment Setup

Purpose:

- compute the repository root from the notebook location,
- append `packages/ltx-core/src` to `sys.path`,
- import the real modules used in the walkthrough.

Expected imports:

- `LTXModel`
- `LTXModelType`
- `Modality`
- `BatchedPerturbationConfig`

This setup is designed so the notebook can run from the repository checkout without requiring `pip install -e`.

### 2. Source Map

Purpose:

- show the reader which files the notebook corresponds to,
- establish the reading order for the real code.

Primary source files:

- `packages/ltx-core/src/ltx_core/model/transformer/model.py`
- `packages/ltx-core/src/ltx_core/model/transformer/transformer_args.py`
- `packages/ltx-core/src/ltx_core/model/transformer/transformer.py`
- `packages/ltx-core/src/ltx_core/model/transformer/modality.py`

### 3. Build a Tiny AudioOnly Model

Purpose:

- instantiate a real `LTXModel` with `model_type=AudioOnly`,
- keep layer counts and hidden dimensions intentionally small for teaching and runtime stability.

Expected configuration style:

- small `num_layers`,
- small `audio_num_attention_heads`,
- small `audio_attention_head_dim`,
- modest `audio_in_channels` and `audio_out_channels`,
- modest `audio_cross_attention_dim`.

This preserves the real architecture while making the notebook practical to run.

### 4. Construct a Minimal Fake Audio `Modality`

Purpose:

- create synthetic tensors that satisfy the real `Modality` contract,
- avoid any dependency on audio files, datasets, or upstream pipelines.

The notebook will create:

- `latent`
- `sigma`
- `timesteps`
- `positions`
- `context`
- `context_mask`
- `attention_mask`

The shapes will be simple but valid for the audio-only path.

### 5. Run Real `prepare()`

Purpose:

- call `model.audio_args_preprocessor.prepare(audio_modality, None)`,
- inspect the resulting `TransformerArgs`,
- print or summarize key shapes and fields.

This section is central because it shows how raw modality inputs are transformed before entering a transformer block.

### 6. Run One Real `BasicAVTransformerBlock`

Purpose:

- select `model.transformer_blocks[0]`,
- run it with `video=None` and the prepared audio args,
- inspect input and output shape continuity.

This isolates one audio-only block and makes the single-stream path concrete.

### 7. Run Full `AudioOnly LTXModel.forward()`

Purpose:

- execute the end-to-end audio-only forward pass using the same synthetic modality,
- demonstrate that the decomposed path and the full model path agree structurally.

The notebook will use an empty perturbation configuration from the real guidance system so the forward path remains authentic.

### 8. Reading Guide

Purpose:

- direct the reader back to the exact implementation areas worth reading next,
- reinforce the mapping from notebook cells to source code.

The guide will explicitly point the reader to:

- `prepare()` in `transformer_args.py`,
- `BasicAVTransformerBlock.forward()` in `transformer.py`,
- `LTXModel.forward()` in `model.py`.

## Data Flow

The notebook should teach this exact path:

1. synthetic `Modality` construction,
2. audio preprocessor `prepare()`,
3. one audio-only transformer block,
4. full audio-only `LTXModel.forward()`.

Conceptually, the reader should leave with this model:

`Modality -> prepare() -> TransformerArgs -> transformer block(s) -> output head`

## Runtime and Dependency Strategy

To maximize the chance that the notebook works in-repo:

- imports will rely on repository-relative path setup,
- no checkpoint files will be loaded,
- no optional extras like xformers will be required,
- CPU execution should be sufficient for the tiny configuration.

The notebook will assume that core package dependencies such as `torch` are installed in the active environment.

## Error Handling

The notebook should fail clearly when the environment is incomplete.

Planned guardrails:

- a setup cell that prints the resolved repository root and source path,
- explicit import error messaging if `ltx_core` dependencies are missing,
- simple shape assertions or printed shape summaries before forward execution,
- explanatory markdown near cells that are most likely to fail if the environment is misconfigured.

## Testing and Verification

Implementation should verify the notebook in two layers.

Layer 1: structural validation

- confirm the `.ipynb` is valid JSON,
- confirm required cells exist and contain the expected imports and walkthrough steps.

Layer 2: runtime validation

- execute the notebook programmatically or cell-by-cell in the local environment if feasible,
- at minimum verify imports, `prepare()`, one block forward, and full `AudioOnly` forward all run successfully with synthetic inputs.

If full automated notebook execution is not feasible in the current environment, the implementation should still run the equivalent Python logic separately to confirm the code path is valid and then state the gap clearly.

## Risks

### Import path fragility

If the notebook assumes a fixed working directory, imports may fail. The setup cell must compute paths relative to the notebook file location rather than current shell state.

### Shape mismatches in synthetic inputs

Because the walkthrough uses real code, fake tensors must satisfy the true interface. The implementation should derive shapes carefully from the chosen tiny model configuration.

### Notebook maintenance

Because `.ipynb` files are JSON, manual edits are clumsy. The implementation should keep the notebook compact and cell text intentional to reduce churn.

## Success Criteria

The work is successful if:

- the notebook lives under `packages/ltx-core/notebooks/`,
- it imports real `ltx_core` modules directly,
- it demonstrates the audio-only single-stream path with synthetic data,
- it clearly maps execution steps back to the source files,
- a reader can use it as a guided bridge from concept to real code.

## Implementation Notes

The implementation should prefer minimalism:

- one notebook,
- no new production helpers unless truly necessary,
- no model behavior changes,
- no speculative features beyond the agreed walkthrough.
