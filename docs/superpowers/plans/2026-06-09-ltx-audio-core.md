# LTX Audio Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete audio-only `ltx-audio-core` package under `packages_audio/ltx-audio-core` that contains the model code, shared primitives, and loading utilities needed before pipeline and trainer work begins.

**Architecture:** Reuse the structure and naming patterns from `packages/ltx-core`, but keep this package audio-only. The work is organized from the outside in: first the public package surface and shared helpers, then transformer internals, then audio codec/model components, then loader and quantization support. Video-only modules stay out of scope in this phase.

**Tech Stack:** Python 3.10+, `uv`, PyTorch, `torchaudio`, `einops`, `numpy`, `transformers`, `safetensors`, `accelerate`, `scipy`, `ruff`, `pytest`

---

## File Structure

### Create

- `packages_audio/ltx-audio-core/src/ltx_audio_core/__init__.py`
- `packages_audio/ltx-audio-core/src/ltx_audio_core/types.py`
- `packages_audio/ltx-audio-core/src/ltx_audio_core/utils.py`
- `packages_audio/ltx-audio-core/src/ltx_audio_core/components/__init__.py`
- `packages_audio/ltx-audio-core/src/ltx_audio_core/components/diffusion_steps.py`
- `packages_audio/ltx-audio-core/src/ltx_audio_core/components/guiders.py`
- `packages_audio/ltx-audio-core/src/ltx_audio_core/components/noisers.py`
- `packages_audio/ltx-audio-core/src/ltx_audio_core/components/patchifiers.py`
- `packages_audio/ltx-audio-core/src/ltx_audio_core/components/protocols.py`
- `packages_audio/ltx-audio-core/src/ltx_audio_core/components/schedulers.py`
- `packages_audio/ltx-audio-core/src/ltx_audio_core/guidance/__init__.py`
- `packages_audio/ltx-audio-core/src/ltx_audio_core/guidance/perturbations.py`
- `packages_audio/ltx-audio-core/src/ltx_audio_core/conditioning/__init__.py`
- `packages_audio/ltx-audio-core/src/ltx_audio_core/conditioning/exceptions.py`
- `packages_audio/ltx-audio-core/src/ltx_audio_core/conditioning/item.py`
- `packages_audio/ltx-audio-core/src/ltx_audio_core/conditioning/mask_utils.py`
- `packages_audio/ltx-audio-core/src/ltx_audio_core/conditioning/types/__init__.py`
- `packages_audio/ltx-audio-core/src/ltx_audio_core/model/__init__.py`
- `packages_audio/ltx-audio-core/src/ltx_audio_core/model/common/__init__.py`
- `packages_audio/ltx-audio-core/src/ltx_audio_core/model/common/normalization.py`
- `packages_audio/ltx-audio-core/src/ltx_audio_core/model/transformer/__init__.py`
- `packages_audio/ltx-audio-core/src/ltx_audio_core/model/transformer/adaln.py`
- `packages_audio/ltx-audio-core/src/ltx_audio_core/model/transformer/attention.py`
- `packages_audio/ltx-audio-core/src/ltx_audio_core/model/transformer/feed_forward.py`
- `packages_audio/ltx-audio-core/src/ltx_audio_core/model/transformer/modality.py`
- `packages_audio/ltx-audio-core/src/ltx_audio_core/model/transformer/rope.py`
- `packages_audio/ltx-audio-core/src/ltx_audio_core/model/transformer/text_projection.py`
- `packages_audio/ltx-audio-core/src/ltx_audio_core/model/transformer/timestep_embedding.py`
- `packages_audio/ltx-audio-core/src/ltx_audio_core/model/transformer/transformer_args.py`
- `packages_audio/ltx-audio-core/src/ltx_audio_core/model/transformer/transformer.py`
- `packages_audio/ltx-audio-core/src/ltx_audio_core/model/transformer/model.py`
- `packages_audio/ltx-audio-core/src/ltx_audio_core/model/audio_vae/__init__.py`
- `packages_audio/ltx-audio-core/src/ltx_audio_core/model/audio_vae/attention.py`
- `packages_audio/ltx-audio-core/src/ltx_audio_core/model/audio_vae/audio_vae.py`
- `packages_audio/ltx-audio-core/src/ltx_audio_core/model/audio_vae/causal_conv_2d.py`
- `packages_audio/ltx-audio-core/src/ltx_audio_core/model/audio_vae/causality_axis.py`
- `packages_audio/ltx-audio-core/src/ltx_audio_core/model/audio_vae/downsample.py`
- `packages_audio/ltx-audio-core/src/ltx_audio_core/model/audio_vae/model_configurator.py`
- `packages_audio/ltx-audio-core/src/ltx_audio_core/model/audio_vae/ops.py`
- `packages_audio/ltx-audio-core/src/ltx_audio_core/model/audio_vae/resnet.py`
- `packages_audio/ltx-audio-core/src/ltx_audio_core/model/audio_vae/upsample.py`
- `packages_audio/ltx-audio-core/src/ltx_audio_core/model/audio_vae/vocoder.py`
- `packages_audio/ltx-audio-core/src/ltx_audio_core/loader/__init__.py`
- `packages_audio/ltx-audio-core/src/ltx_audio_core/loader/attention_ops.py`
- `packages_audio/ltx-audio-core/src/ltx_audio_core/loader/fuse_loras.py`
- `packages_audio/ltx-audio-core/src/ltx_audio_core/loader/helpers.py`
- `packages_audio/ltx-audio-core/src/ltx_audio_core/loader/kernels.py`
- `packages_audio/ltx-audio-core/src/ltx_audio_core/loader/module_ops.py`
- `packages_audio/ltx-audio-core/src/ltx_audio_core/loader/primitives.py`
- `packages_audio/ltx-audio-core/src/ltx_audio_core/loader/registry.py`
- `packages_audio/ltx-audio-core/src/ltx_audio_core/loader/sd_ops.py`
- `packages_audio/ltx-audio-core/src/ltx_audio_core/loader/sft_loader.py`
- `packages_audio/ltx-audio-core/src/ltx_audio_core/loader/single_gpu_model_builder.py`
- `packages_audio/ltx-audio-core/src/ltx_audio_core/quantization/__init__.py`
- `packages_audio/ltx-audio-core/src/ltx_audio_core/quantization/fp8_cast.py`
- `packages_audio/ltx-audio-core/src/ltx_audio_core/quantization/fp8_scaled_mm.py`
- `packages_audio/ltx-audio-core/src/ltx_audio_core/quantization/policy.py`
- `packages_audio/ltx-audio-core/src/ltx_audio_core/quantization/trtllm_scaled_usable.py`
- `packages_audio/ltx-audio-core/tests/test_imports.py`
- `packages_audio/ltx-audio-core/tests/test_transformer_audio_only.py`
- `packages_audio/ltx-audio-core/tests/test_audio_vae_shapes.py`
- `packages_audio/ltx-audio-core/tests/test_loader_smoke.py`

### Modify

- `pyproject.toml`
- `packages_audio/ltx-audio-core/pyproject.toml`
- `packages_audio/ltx-audio-core/README.md`

### Reference Sources

- `packages/ltx-core/src/ltx_core/__init__.py`
- `packages/ltx-core/src/ltx_core/types.py`
- `packages/ltx-core/src/ltx_core/utils.py`
- `packages/ltx-core/src/ltx_core/components/*`
- `packages/ltx-core/src/ltx_core/guidance/perturbations.py`
- `packages/ltx-core/src/ltx_core/conditioning/*`
- `packages/ltx-core/src/ltx_core/model/common/normalization.py`
- `packages/ltx-core/src/ltx_core/model/transformer/*`
- `packages/ltx-core/src/ltx_core/model/audio_vae/*`
- `packages/ltx-core/src/ltx_core/loader/*`
- `packages/ltx-core/src/ltx_core/quantization/*`

---

## Task 1: Lock the audio-core package boundary and public exports

**Files:**
- Modify: `pyproject.toml`
- Modify: `packages_audio/ltx-audio-core/README.md`
- Create: `packages_audio/ltx-audio-core/src/ltx_audio_core/__init__.py`
- Create: `packages_audio/ltx-audio-core/tests/test_imports.py`

**Reference sources:**
- `packages/ltx-core/src/ltx_core/__init__.py`
- `packages/ltx-core/src/ltx_core/model/__init__.py`
- `packages/ltx-core/src/ltx_core/types.py`
- `packages/ltx-core/src/ltx_core/utils.py`

- [ ] **Step 1: Write the failing import test**

```python
from ltx_audio_core import __all__  # noqa: F401
```

Expected: fail until the package exports are in place.

- [ ] **Step 2: Implement the package surface**

Export the top-level audio-only names that the rest of the package will depend on, and keep the README focused on audio-only scope.

- [ ] **Step 3: Run the import smoke test**

Run:

```bash
uv run pytest packages_audio/ltx-audio-core/tests/test_imports.py -q
```

Expected: PASS.

- [ ] **Step 4: Commit the package boundary**

```bash
git add pyproject.toml packages_audio/ltx-audio-core
git commit -m "feat: start ltx-audio-core package"
```

### Task 2: Port shared primitives used by audio-only model code

**Files:**
- Create: `packages_audio/ltx-audio-core/src/ltx_audio_core/types.py`
- Create: `packages_audio/ltx-audio-core/src/ltx_audio_core/utils.py`
- Create: `packages_audio/ltx-audio-core/src/ltx_audio_core/components/*`
- Create: `packages_audio/ltx-audio-core/src/ltx_audio_core/guidance/perturbations.py`
- Create: `packages_audio/ltx-audio-core/tests/test_shared_primitives.py`

**Reference sources:**
- `packages/ltx-core/src/ltx_core/types.py`
- `packages/ltx-core/src/ltx_core/utils.py`
- `packages/ltx-core/src/ltx_core/components/protocols.py`
- `packages/ltx-core/src/ltx_core/components/schedulers.py`
- `packages/ltx-core/src/ltx_core/components/noisers.py`
- `packages/ltx-core/src/ltx_core/components/guiders.py`
- `packages/ltx-core/src/ltx_core/components/patchifiers.py`
- `packages/ltx-core/src/ltx_core/components/diffusion_steps.py`
- `packages/ltx-core/src/ltx_core/guidance/perturbations.py`

- [ ] **Step 1: Write a failing test for the shared helpers**

Create a test that imports the new helpers and exercises one representative tensor-shape path.

- [ ] **Step 2: Implement the shared helpers one file at a time**

Keep the code close to the original naming and behavior, but remove any video-only assumptions.

- [ ] **Step 3: Run the shared-primitives test file**

Run:

```bash
uv run pytest packages_audio/ltx-audio-core/tests/test_shared_primitives.py -q
```

Expected: PASS.

- [ ] **Step 4: Commit the primitives layer**

```bash
git add packages_audio/ltx-audio-core/src/ltx_audio_core
git commit -m "feat: add shared primitives to ltx-audio-core"
```

### Task 3: Implement the audio transformer stack end to end

**Files:**
- Create: `packages_audio/ltx-audio-core/src/ltx_audio_core/model/transformer/*`
- Create: `packages_audio/ltx-audio-core/src/ltx_audio_core/model/common/normalization.py`
- Create: `packages_audio/ltx-audio-core/src/ltx_audio_core/model/__init__.py`
- Create: `packages_audio/ltx-audio-core/tests/test_transformer_audio_only.py`

**Reference sources:**
- `packages/ltx-core/src/ltx_core/model/transformer/model.py`
- `packages/ltx-core/src/ltx_core/model/transformer/transformer_args.py`
- `packages/ltx-core/src/ltx_core/model/transformer/transformer.py`
- `packages/ltx-core/src/ltx_core/model/transformer/modality.py`
- `packages/ltx-core/src/ltx_core/model/transformer/adaln.py`
- `packages/ltx-core/src/ltx_core/model/transformer/attention.py`
- `packages/ltx-core/src/ltx_core/model/transformer/feed_forward.py`
- `packages/ltx-core/src/ltx_core/model/transformer/rope.py`
- `packages/ltx-core/src/ltx_core/model/transformer/text_projection.py`
- `packages/ltx-core/src/ltx_core/model/transformer/timestep_embedding.py`

- [ ] **Step 1: Write the failing audio-only shape test**

Cover `Modality`, `TransformerArgsPreprocessor.prepare()`, one transformer block, and the full audio-only `LTXModel.forward()` path.

- [ ] **Step 2: Port the transformer modules in dependency order**

Start with `modality.py`, `transformer_args.py`, `adaln.py`, and `rope.py`, then add attention, feed-forward, and the model wrapper.

- [ ] **Step 3: Run the transformer test file**

Run:

```bash
uv run pytest packages_audio/ltx-audio-core/tests/test_transformer_audio_only.py -q
```

Expected: PASS.

- [ ] **Step 4: Commit the transformer stack**

```bash
git add packages_audio/ltx-audio-core/src/ltx_audio_core packages_audio/ltx-audio-core/tests
git commit -m "feat: add audio transformer stack"
```

### Task 4: Port the audio VAE and vocoder stack

**Files:**
- Create: `packages_audio/ltx-audio-core/src/ltx_audio_core/model/audio_vae/*`
- Create: `packages_audio/ltx-audio-core/tests/test_audio_vae_shapes.py`

**Reference sources:**
- `packages/ltx-core/src/ltx_core/model/audio_vae/audio_vae.py`
- `packages/ltx-core/src/ltx_core/model/audio_vae/vocoder.py`
- `packages/ltx-core/src/ltx_core/model/audio_vae/attention.py`
- `packages/ltx-core/src/ltx_core/model/audio_vae/causal_conv_2d.py`
- `packages/ltx-core/src/ltx_core/model/audio_vae/downsample.py`
- `packages/ltx-core/src/ltx_core/model/audio_vae/upsample.py`
- `packages/ltx-core/src/ltx_core/model/audio_vae/resnet.py`
- `packages/ltx-core/src/ltx_core/model/audio_vae/ops.py`
- `packages/ltx-core/src/ltx_core/model/audio_vae/model_configurator.py`

- [ ] **Step 1: Write the failing audio VAE smoke test**

Test the minimal encode/decode or construct/forward shape path that the audio-only package will expose.

- [ ] **Step 2: Implement the audio codec stack**

Build the modules in the same dependency order as the original package, keeping the API audio-only.

- [ ] **Step 3: Run the audio VAE tests**

Run:

```bash
uv run pytest packages_audio/ltx-audio-core/tests/test_audio_vae_shapes.py -q
```

Expected: PASS.

- [ ] **Step 4: Commit the codec layer**

```bash
git add packages_audio/ltx-audio-core/src/ltx_audio_core packages_audio/ltx-audio-core/tests
git commit -m "feat: add audio vae stack"
```

### Task 5: Add loader and quantization support for audio-only checkpoints

**Files:**
- Create: `packages_audio/ltx-audio-core/src/ltx_audio_core/loader/*`
- Create: `packages_audio/ltx-audio-core/src/ltx_audio_core/quantization/*`
- Create: `packages_audio/ltx-audio-core/tests/test_loader_smoke.py`

**Reference sources:**
- `packages/ltx-core/src/ltx_core/loader/*`
- `packages/ltx-core/src/ltx_core/quantization/*`

- [ ] **Step 1: Write the failing loader smoke test**

Import the loader API and assert a minimal state-dict or builder path exists for audio-only use.

- [ ] **Step 2: Implement the loader and quantization files**

Keep the APIs aligned with the original package so the future pipeline layer can reuse them.

- [ ] **Step 3: Run the loader smoke test**

Run:

```bash
uv run pytest packages_audio/ltx-audio-core/tests/test_loader_smoke.py -q
```

Expected: PASS.

- [ ] **Step 4: Commit the loading layer**

```bash
git add packages_audio/ltx-audio-core/src/ltx_audio_core packages_audio/ltx-audio-core/tests
git commit -m "feat: add loader support to ltx-audio-core"
```

### Task 6: Final pass, docs polish, and workspace verification

**Files:**
- Modify: `packages_audio/ltx-audio-core/README.md`
- Modify: `packages_audio/ltx-audio-core/pyproject.toml`
- Modify: `pyproject.toml`

**Reference sources:**
- `packages/ltx-core/README.md`
- `packages/ltx-core/pyproject.toml`
- `packages/ltx-pipelines/README.md`
- `packages/ltx-trainer/README.md`

- [ ] **Step 1: Remove mismatch or leftover audio-core naming issues**

Make sure the package name, module name, and README wording all say the same thing.

- [ ] **Step 2: Run workspace and import verification**

Run:

```bash
uv workspace list
uv run python - <<'PY'
import ltx_audio_core
print(ltx_audio_core.__file__)
PY
```

Expected: `ltx-audio-core` appears in the workspace list, and the import resolves from `packages_audio/ltx-audio-core/src`.

- [ ] **Step 3: Commit the final cleanup**

```bash
git add pyproject.toml packages_audio/ltx-audio-core
git commit -m "docs: finalize ltx-audio-core package"
```

## Scope Check

This plan stays inside `ltx-audio-core` only. `ltx-audio-pipelines` and `ltx-audio-trainer` are intentionally deferred to the next phase so this package can be completed and validated on its own first.

## Self-Review Notes

- No placeholder requirements remain.
- The task order matches dependency order.
- Every task names the source files it should mirror.
- The test plan is focused on import, shape, and smoke coverage instead of broad integration.
