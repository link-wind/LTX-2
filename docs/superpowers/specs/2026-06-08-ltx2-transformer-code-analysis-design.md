# LTX-2 Transformer Core Code Analysis Design

> Date: 2026-06-08
> Topic: LTX-2 transformer core code learning notes
> Status: Draft for review

## Goal

Create a structured set of learning documents that explains the core transformer code in `LTX-2` for a reader who is moving from entry-level understanding toward confident source-code reading.

The document set should help a reader:

- understand where the transformer sits in the overall `ltx-core` architecture,
- build a clear mental model of the video/audio dual-stream transformer path,
- follow how data moves through preprocessing, RoPE, AdaLN, attention, feed-forward, and block composition,
- read the real source code with less confusion and less jumping between unrelated files.

## Scope

This work defines the design for a learning-note series only. It does not change model behavior or add runtime features.

In scope:

- a dedicated notes directory under the repository root,
- a numbered series of Markdown analysis files named `01-xxxx.md`,
- a consistent document structure across the full series,
- a reading order centered on mechanisms rather than file-by-file dumping,
- coverage limited to `packages/ltx-core/src/ltx_core/model/transformer/`.

Out of scope:

- pipeline analysis outside the transformer package,
- VAE, vocoder, trainer, or loader deep dives,
- code changes to the LTX model implementation,
- diagrams or notebooks unless later requested,
- automatic publishing or site generation for the notes.

## Recommended Approach

Use a mechanism-oriented note series with one opening document for the whole call chain and one closing document for review and navigation.

Why this approach:

- It matches the learning goal better than file-by-file commentary.
- It reduces cognitive load for readers who are new to this codebase.
- It allows each note to answer one focused question, such as “What does AdaLN do here?” instead of forcing the reader through a long mixed-responsibility file.
- It still stays grounded in real source locations and call relationships.

Alternatives considered:

1. Pure file-by-file analysis
   - Closer to repository layout, but harder for new readers because many files are support layers rather than learning layers.

2. Pure mechanism analysis without an entry document
   - Clean topics, but readers may not know where each mechanism is used in the end-to-end path.

3. Minimal notes with only summaries
   - Fast to write, but not enough for the “learn the code” objective.

## Audience

Primary audience:

- readers at an entry-to-intermediate level,
- readers who know the broad idea of Transformers but are not yet comfortable reading a production diffusion-transformer codebase,
- readers who need help understanding module responsibilities, tensor flow, and source-code reading order.

Implications for writing style:

- explain responsibilities before implementation details,
- surface tensor flow and call relationships early,
- avoid assuming that the reader already understands every DiT-specific convention,
- use implementation details to support understanding, not to overwhelm it.

## Location

Create the note series at:

`notes/ltx2-transformer/`

Rationale:

- it keeps the learning artifacts separate from product documentation,
- it leaves room for additional study tracks later,
- it makes the numbered sequence easy to browse at the repository root level.

## Naming Convention

Each note should use the format:

`NN-主题名.md`

Examples:

- `01-LTXModel整体结构与调用主线.md`
- `02-TransformerArgs与预处理管线.md`
- `03-AdaLN与时间步调制.md`

The number provides stable reading order. The title should be explicit enough that a reader can understand the topic from the filename alone.

## Note Series Structure

The initial series should contain the following notes.

### 01. `LTXModel整体结构与调用主线`

Purpose:

- establish the big picture for the transformer package,
- explain the relationship between `model.py`, `transformer.py`, and the mechanism files,
- show how video and audio streams enter the model.

Primary source files:

- `packages/ltx-core/src/ltx_core/model/transformer/model.py`
- `packages/ltx-core/src/ltx_core/model/transformer/transformer.py`

### 02. `TransformerArgs与预处理管线`

Purpose:

- explain why the model does not pass raw latents directly into blocks,
- cover `TransformerArgs`, `TransformerArgsPreprocessor`, and the preparation of timestep, context, masks, and positional embeddings.

Primary source files:

- `packages/ltx-core/src/ltx_core/model/transformer/transformer_args.py`
- `packages/ltx-core/src/ltx_core/model/transformer/modality.py`

### 03. `AdaLN与时间步调制`

Purpose:

- explain how timestep information is injected,
- clarify `AdaLayerNormSingle`, scale/shift modulation, and why this differs from ordinary Transformer normalization flows.

Primary source files:

- `packages/ltx-core/src/ltx_core/model/transformer/adaln.py`
- `packages/ltx-core/src/ltx_core/model/transformer/timestep_embedding.py`

### 04. `RoPE位置编码机制`

Purpose:

- explain how positional encoding is prepared and applied,
- distinguish video, audio, and cross-modal RoPE usage at a conceptual level.

Primary source files:

- `packages/ltx-core/src/ltx_core/model/transformer/rope.py`
- `packages/ltx-core/src/ltx_core/model/transformer/transformer_args.py`

### 05. `Attention模块总览`

Purpose:

- explain the `Attention` class itself,
- walk through self-attention vs cross-attention entry points,
- clarify how query, key, value, and masks are threaded through the implementation.

Primary source files:

- `packages/ltx-core/src/ltx_core/model/transformer/attention.py`

### 06. `Attention后端与算子选择`

Purpose:

- explain why the attention implementation is split into backend callables,
- show how SDPA, xFormers, FlashAttention, and automatic backend selection fit together.

Primary source files:

- `packages/ltx-core/src/ltx_core/model/transformer/attention.py`
- `packages/ltx-core/src/ltx_core/model/transformer/ops.py`

### 07. `FeedForward前馈网络`

Purpose:

- explain the role of the feed-forward sublayer inside the block,
- make the FFN path easy to read in isolation before it is reassembled into the full block.

Primary source files:

- `packages/ltx-core/src/ltx_core/model/transformer/feed_forward.py`
- `packages/ltx-core/src/ltx_core/model/transformer/gelu_approx.py`

### 08. `BasicAVTransformerBlock单层结构`

Purpose:

- explain one full transformer block as the key learning unit,
- show how self-attention, text cross-attention, feed-forward, Ada values, and residual composition fit together.

Primary source files:

- `packages/ltx-core/src/ltx_core/model/transformer/transformer.py`

### 09. `音视频双流交互机制`

Purpose:

- explain how audio and video streams interact inside the block,
- focus on bidirectional cross-modal attention, gating, and separate hidden widths.

Primary source files:

- `packages/ltx-core/src/ltx_core/model/transformer/transformer.py`
- `packages/ltx-core/src/ltx_core/model/transformer/model.py`

### 10. `模型前向传播全链路梳理`

Purpose:

- connect preprocessing, per-block execution, and output projection into one end-to-end path,
- provide the “now I can follow the real forward” reading checkpoint.

Primary source files:

- `packages/ltx-core/src/ltx_core/model/transformer/model.py`
- `packages/ltx-core/src/ltx_core/model/transformer/transformer.py`
- `packages/ltx-core/src/ltx_core/model/transformer/transformer_args.py`

### 11. `关键配置对象与可插拔设计`

Purpose:

- explain `TransformerConfig`, `TransformerOpsConfig`, and the general design idea of pluggable ops,
- help the reader understand where architecture ends and backend customization begins.

Primary source files:

- `packages/ltx-core/src/ltx_core/model/transformer/transformer.py`
- `packages/ltx-core/src/ltx_core/model/transformer/ops.py`
- `packages/ltx-core/src/ltx_core/model/transformer/compiling.py`

### 12. `LTX-2 Transformer学习地图`

Purpose:

- serve as a review and navigation document,
- summarize reading order, dependency relationships, and common confusion points,
- point to next study areas outside this series.

Primary source files:

- all notes in the series,
- selected transformer package files as backlinks.

## Standard Template For Each Note

To keep the series consistent, each note should follow the same top-level structure.

### 1. 这个机制解决什么问题

The opening should answer the practical question first before diving into code.

### 2. 它在整体调用链中的位置

This section should anchor the mechanism inside the end-to-end model path.

### 3. 关键类与函数

List only the relevant classes and functions, not every symbol in the file.

### 4. 数据流或张量流

Show how inputs become outputs, especially where shapes or semantic roles change.

### 5. 关键源码分段解析

Break the source into a few meaningful segments and explain each segment’s job.

### 6. 设计动机与实现取舍

Explain why this design likely exists and what complexity it introduces or avoids.

### 7. 一句话总结与下一篇跳转

Each note should close with a compact takeaway and suggest the next note to read.

## Style Requirements

The notes should be written in Chinese and should borrow the strengths of the provided reference style without copying its exact wording or structure.

Writing priorities:

- explain from whole to part,
- keep the tone educational rather than academic,
- prefer concise code excerpts over giant pasted blocks,
- describe shapes and responsibilities in plain language,
- explicitly call out confusion-prone concepts such as dual-stream width differences, RoPE variants, and AdaLN parameter generation.

The series should avoid:

- line-by-line transcription of entire files,
- unexplained jargon,
- overly long historical speculation,
- mixing unrelated subsystems into a single note.

## Research and Reading Order

The implementation of the note series should read the code in this order:

1. `model.py`
2. `transformer.py`
3. `transformer_args.py`
4. `attention.py`
5. `adaln.py`
6. `rope.py`
7. `feed_forward.py`
8. `ops.py`
9. `compiling.py`
10. supporting files only as needed

This order is chosen to preserve the reader’s mental model:

- first understand the big picture,
- then understand the block contract,
- then understand the mechanisms that make each block work.

## Risks

### Scope drift

The transformer package touches text conditioning, perturbations, backend selection, and multimodal interaction. Without a strict series boundary, the notes can easily expand into the whole repository.

Mitigation:

- keep the series anchored to the transformer directory,
- mention external dependencies only when they are required for understanding the local code.

### Mechanism overlap

Attention, RoPE, AdaLN, and block composition overlap heavily. If note boundaries are loose, the same content will be repeated across multiple files.

Mitigation:

- define a primary question for each note,
- allow brief references across notes instead of re-explaining full mechanisms.

### Reader overload

Even entry-level readers can get lost if a note jumps too quickly from high-level summary to backend-specific optimization details.

Mitigation:

- start with module purpose and call position,
- delay low-level optimization details until the dedicated backend note.

## Success Criteria

The design is successful if:

- the repository has a dedicated `notes/ltx2-transformer/` location for the series,
- the series has a stable numbered outline before drafting begins,
- every note has a clear learning purpose and source boundary,
- the structure supports a reader who wants to learn the real code incrementally,
- future drafting can proceed note by note without redefining scope each time.

## Next Step

After this design is approved, the next planning step should convert the series into an execution plan for drafting order, verification expectations, and incremental delivery of the actual `01-xxxx.md` files.
