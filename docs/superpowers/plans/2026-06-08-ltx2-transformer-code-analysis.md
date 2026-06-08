# LTX-2 Transformer Code Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a complete `notes/ltx2-transformer/` Markdown learning series that explains the LTX-2 transformer core code in a stable `01-xxxx.md` order for entry-to-intermediate readers.

**Architecture:** The work is documentation-first and stays inside the repository as a dedicated study track. Implementation proceeds in four passes: establish the directory and shared note template, draft the mechanism notes in reading order, add the entry and summary notes that tie the series together, then run a consistency pass for naming, headings, code references, and cross-note navigation.

**Tech Stack:** Markdown, shell validation commands, ripgrep, git, source files under `packages/ltx-core/src/ltx_core/model/transformer/`

---

## File Structure

### Create

- `notes/ltx2-transformer/01-LTXModel整体结构与调用主线.md`
- `notes/ltx2-transformer/02-TransformerArgs与预处理管线.md`
- `notes/ltx2-transformer/03-AdaLN与时间步调制.md`
- `notes/ltx2-transformer/04-RoPE位置编码机制.md`
- `notes/ltx2-transformer/05-Attention模块总览.md`
- `notes/ltx2-transformer/06-Attention后端与算子选择.md`
- `notes/ltx2-transformer/07-FeedForward前馈网络.md`
- `notes/ltx2-transformer/08-BasicAVTransformerBlock单层结构.md`
- `notes/ltx2-transformer/09-音视频双流交互机制.md`
- `notes/ltx2-transformer/10-模型前向传播全链路梳理.md`
- `notes/ltx2-transformer/11-关键配置对象与可插拔设计.md`
- `notes/ltx2-transformer/12-LTX-2 Transformer学习地图.md`
- `docs/superpowers/plans/2026-06-08-ltx2-transformer-code-analysis.md`

### Inspect While Drafting

- `packages/ltx-core/src/ltx_core/model/transformer/model.py`
- `packages/ltx-core/src/ltx_core/model/transformer/transformer.py`
- `packages/ltx-core/src/ltx_core/model/transformer/transformer_args.py`
- `packages/ltx-core/src/ltx_core/model/transformer/adaln.py`
- `packages/ltx-core/src/ltx_core/model/transformer/timestep_embedding.py`
- `packages/ltx-core/src/ltx_core/model/transformer/rope.py`
- `packages/ltx-core/src/ltx_core/model/transformer/attention.py`
- `packages/ltx-core/src/ltx_core/model/transformer/feed_forward.py`
- `packages/ltx-core/src/ltx_core/model/transformer/gelu_approx.py`
- `packages/ltx-core/src/ltx_core/model/transformer/ops.py`
- `packages/ltx-core/src/ltx_core/model/transformer/compiling.py`
- `packages/ltx-core/src/ltx_core/model/transformer/modality.py`

### Validation Commands

- `find /Users/linkwind/Code/LTX-2/notes/ltx2-transformer -maxdepth 1 -type f | sort`
- `rg -n "^## " /Users/linkwind/Code/LTX-2/notes/ltx2-transformer`
- `rg -n "TODO|TBD|占位|稍后|待补充" /Users/linkwind/Code/LTX-2/notes/ltx2-transformer`
- `python3 - <<'PY' ... PY` to assert all 12 files exist and all required section headings are present
- `git diff --check`

---

### Task 1: Create the notes directory and lock the shared note template

**Files:**
- Create: `notes/ltx2-transformer/01-LTXModel整体结构与调用主线.md`

- [ ] **Step 1: Write the failing existence check**

Run:

```bash
python3 - <<'PY'
from pathlib import Path

root = Path("/Users/linkwind/Code/LTX-2/notes/ltx2-transformer")
assert root.exists(), f"Missing notes directory: {root}"
PY
```

Expected: FAIL with `Missing notes directory`.

- [ ] **Step 2: Create the notes directory**

Run:

```bash
mkdir -p /Users/linkwind/Code/LTX-2/notes/ltx2-transformer
```

Expected: command succeeds with no output.

- [ ] **Step 3: Draft the shared note skeleton in the first note**

Create `01-LTXModel整体结构与调用主线.md` with this top-level structure, preserving the section names exactly:

```markdown
# 01 LTXModel整体结构与调用主线

## 这个机制解决什么问题

本篇先不钻进某一个算子细节，而是先回答一个更关键的问题：LTX-2 的 Transformer 主干在整个 `ltx-core` 里处在什么位置，输入是怎样进入模型的，video 和 audio 两条流又是怎样在总模型里被组织起来的。

## 它在整体调用链中的位置

可以先把主链路记成一条粗线：

`Modality -> TransformerArgsPreprocessor.prepare() -> BasicAVTransformerBlock x N -> 输出归一化/投影`

## 关键类与函数

- `LTXModel`
- `LTXModelType`
- `_init_video()`
- `_init_audio()`
- `_init_audio_video()`
- `_init_preprocessors()`
- `_init_transformer_blocks()`

## 数据流或张量流

这一节统一用“输入对象是什么、进入哪一层、出来变成什么”来写，不要一开始就铺太多源码细节。

## 关键源码分段解析

这一节按 `model.py` 中的初始化、预处理器注册、block 组装、forward 主线来拆。

## 设计动机与实现取舍

这一节重点解释为什么 LTX-2 不是一个单流 Transformer，以及为什么预处理和 block 逻辑被拆到不同文件。

## 一句话总结与下一篇跳转

如果把 `01` 看成“总地图”，那么下一篇 `02-TransformerArgs与预处理管线.md` 就是正式解释输入在进 block 之前被整理成了什么形状。
```

- [ ] **Step 4: Verify the section headings in the first note**

Run:

```bash
rg -n "^## " /Users/linkwind/Code/LTX-2/notes/ltx2-transformer/01-LTXModel整体结构与调用主线.md
```

Expected output contains exactly these headings:

- `## 这个机制解决什么问题`
- `## 它在整体调用链中的位置`
- `## 关键类与函数`
- `## 数据流或张量流`
- `## 关键源码分段解析`
- `## 设计动机与实现取舍`
- `## 一句话总结与下一篇跳转`

- [ ] **Step 5: Commit the directory and template lock-in**

```bash
git add /Users/linkwind/Code/LTX-2/notes/ltx2-transformer/01-LTXModel整体结构与调用主线.md
git commit -m "docs: start ltx2 transformer notes series"
```

Expected: commit succeeds and tracks the new notes directory.

### Task 2: Draft the entry and preprocessing notes that establish the reading model

**Files:**
- Modify: `notes/ltx2-transformer/01-LTXModel整体结构与调用主线.md`
- Create: `notes/ltx2-transformer/02-TransformerArgs与预处理管线.md`
- Create: `notes/ltx2-transformer/03-AdaLN与时间步调制.md`

- [ ] **Step 1: Write the failing file-count check**

Run:

```bash
python3 - <<'PY'
from pathlib import Path

root = Path("/Users/linkwind/Code/LTX-2/notes/ltx2-transformer")
files = sorted(p.name for p in root.glob("*.md"))
assert len(files) >= 3, f"Expected at least 3 notes, found {len(files)}: {files}"
PY
```

Expected: FAIL because only the first note exists.

- [ ] **Step 2: Finish the first three notes with concrete source anchors**

Write the three notes so each one includes:

```markdown
## 关键类与函数

- `LTXModel`
- `TransformerArgs`
- `TransformerArgsPreprocessor`
- `AdaLayerNormSingle`
```

and at least one explicit code-reference block like:

```markdown
重点阅读文件：

- `packages/ltx-core/src/ltx_core/model/transformer/model.py`
- `packages/ltx-core/src/ltx_core/model/transformer/transformer_args.py`
- `packages/ltx-core/src/ltx_core/model/transformer/adaln.py`
```

For `02`, include a plain-language explanation of:

```markdown
- `patchify_proj` 把原始 latent 映射到 block 需要的隐藏维度；
- `adaln` 先把时间步变成后续调制会用到的向量；
- `context_mask` 和 `self_attention_mask` 会在进入 attention 前被整理成可广播的形态；
- `positional_embeddings` 不是在 block 里临时现算，而是在预处理阶段就准备好。
```

For `03`, include a compact explanation of:

```markdown
AdaLN 在这里不是“普通 LayerNorm 的替代品”这么简单。它承担的是把扩散时间步信息稳定地注入每一层计算图，让同一个 block 在不同 denoising 阶段表现出不同的调制行为。
```

- [ ] **Step 3: Verify the first three filenames and headings**

Run:

```bash
python3 - <<'PY'
from pathlib import Path

root = Path("/Users/linkwind/Code/LTX-2/notes/ltx2-transformer")
required = [
    "01-LTXModel整体结构与调用主线.md",
    "02-TransformerArgs与预处理管线.md",
    "03-AdaLN与时间步调制.md",
]
headings = {
    "## 这个机制解决什么问题",
    "## 它在整体调用链中的位置",
    "## 关键类与函数",
    "## 数据流或张量流",
    "## 关键源码分段解析",
    "## 设计动机与实现取舍",
    "## 一句话总结与下一篇跳转",
}
for name in required:
    path = root / name
    text = path.read_text()
    assert path.exists(), f"Missing note: {name}"
    for heading in headings:
        assert heading in text, f"{name} missing heading: {heading}"
print("validated", len(required), "notes")
PY
```

Expected: PASS and print `validated 3 notes`.

- [ ] **Step 4: Commit the first drafting batch**

```bash
git add /Users/linkwind/Code/LTX-2/notes/ltx2-transformer/01-LTXModel整体结构与调用主线.md \
        /Users/linkwind/Code/LTX-2/notes/ltx2-transformer/02-TransformerArgs与预处理管线.md \
        /Users/linkwind/Code/LTX-2/notes/ltx2-transformer/03-AdaLN与时间步调制.md
git commit -m "docs: draft ltx2 transformer introduction notes"
```

### Task 3: Draft the core mechanism notes for RoPE, attention, backend selection, and FFN

**Files:**
- Create: `notes/ltx2-transformer/04-RoPE位置编码机制.md`
- Create: `notes/ltx2-transformer/05-Attention模块总览.md`
- Create: `notes/ltx2-transformer/06-Attention后端与算子选择.md`
- Create: `notes/ltx2-transformer/07-FeedForward前馈网络.md`

- [ ] **Step 1: Write the failing missing-file check**

Run:

```bash
python3 - <<'PY'
from pathlib import Path

root = Path("/Users/linkwind/Code/LTX-2/notes/ltx2-transformer")
missing = [name for name in [
    "04-RoPE位置编码机制.md",
    "05-Attention模块总览.md",
    "06-Attention后端与算子选择.md",
    "07-FeedForward前馈网络.md",
] if not (root / name).exists()]
assert not missing, f"Missing notes: {missing}"
PY
```

Expected: FAIL listing the four missing notes.

- [ ] **Step 2: Draft the four mechanism notes with topic-specific boundaries**

Each note must contain one short “不要混淆” paragraph that prevents overlap:

```markdown
不要混淆两件事：

- `05-Attention模块总览.md` 讲的是 Attention 类本身怎么接输入、拆 QKV、走 self/cross 两种路径；
- `06-Attention后端与算子选择.md` 讲的是同一套 attention 接口背后，为什么还能切到 SDPA、xFormers 或 FlashAttention。
```

`04-RoPE位置编码机制.md` must include:

```markdown
这里最容易误解的一点是：RoPE 在 LTX-2 里不只是“给序列加位置”，而是要适配视频、音频、跨模态这几种不同的时间/空间组织方式。
```

`07-FeedForward前馈网络.md` must include:

```markdown
虽然 FFN 在阅读体验上常常被 attention 抢走注意力，但在真实 block 里，它承担的是每个 token 独立的通道内变换，是 attention 之外第二个最稳定的容量来源。
```

- [ ] **Step 3: Verify that the four notes mention the intended source files**

Run:

```bash
python3 - <<'PY'
from pathlib import Path

root = Path("/Users/linkwind/Code/LTX-2/notes/ltx2-transformer")
checks = {
    "04-RoPE位置编码机制.md": ["rope.py", "transformer_args.py"],
    "05-Attention模块总览.md": ["attention.py"],
    "06-Attention后端与算子选择.md": ["attention.py", "ops.py"],
    "07-FeedForward前馈网络.md": ["feed_forward.py", "gelu_approx.py"],
}
for name, needles in checks.items():
    text = (root / name).read_text()
    for needle in needles:
        assert needle in text, f"{name} missing source reference: {needle}"
print("validated mechanism note source anchors")
PY
```

Expected: PASS and print `validated mechanism note source anchors`.

- [ ] **Step 4: Commit the mechanism batch**

```bash
git add /Users/linkwind/Code/LTX-2/notes/ltx2-transformer/04-RoPE位置编码机制.md \
        /Users/linkwind/Code/LTX-2/notes/ltx2-transformer/05-Attention模块总览.md \
        /Users/linkwind/Code/LTX-2/notes/ltx2-transformer/06-Attention后端与算子选择.md \
        /Users/linkwind/Code/LTX-2/notes/ltx2-transformer/07-FeedForward前馈网络.md
git commit -m "docs: draft ltx2 transformer mechanism notes"
```

### Task 4: Draft the block-level and end-to-end notes

**Files:**
- Create: `notes/ltx2-transformer/08-BasicAVTransformerBlock单层结构.md`
- Create: `notes/ltx2-transformer/09-音视频双流交互机制.md`
- Create: `notes/ltx2-transformer/10-模型前向传播全链路梳理.md`

- [ ] **Step 1: Write the failing block-batch check**

Run:

```bash
python3 - <<'PY'
from pathlib import Path

root = Path("/Users/linkwind/Code/LTX-2/notes/ltx2-transformer")
required = [
    "08-BasicAVTransformerBlock单层结构.md",
    "09-音视频双流交互机制.md",
    "10-模型前向传播全链路梳理.md",
]
missing = [name for name in required if not (root / name).exists()]
assert not missing, f"Missing block-level notes: {missing}"
PY
```

Expected: FAIL listing the three missing files.

- [ ] **Step 2: Draft the three notes with explicit scope separation**

Use these scope statements verbatim or nearly verbatim in the relevant notes:

```markdown
`08` 的重点是“单层 block 内部怎么拼起来”，不是完整 forward。
```

```markdown
`09` 的重点是“audio 和 video 两条流怎样相互看见对方”，不是文本条件本身怎么编码。
```

```markdown
`10` 的重点是把前面拆开的机制重新接回一条完整主线，让读者可以回到 `LTXModel.forward()` 时不再迷路。
```

For `10`, include one compact chain such as:

```markdown
可以把完整路径记成：

`Modality -> 预处理器 -> TransformerArgs -> 多层 BasicAVTransformerBlock -> norm/proj_out -> 最终输出`
```

- [ ] **Step 3: Verify cross-note navigation between 08, 09, and 10**

Run:

```bash
python3 - <<'PY'
from pathlib import Path

root = Path("/Users/linkwind/Code/LTX-2/notes/ltx2-transformer")
text8 = (root / "08-BasicAVTransformerBlock单层结构.md").read_text()
text9 = (root / "09-音视频双流交互机制.md").read_text()
text10 = (root / "10-模型前向传播全链路梳理.md").read_text()
assert "09-音视频双流交互机制.md" in text8, "08 should point to 09"
assert "10-模型前向传播全链路梳理.md" in text9, "09 should point to 10"
assert "01-LTXModel整体结构与调用主线.md" in text10, "10 should backlink to 01"
print("validated block/end-to-end navigation")
PY
```

Expected: PASS and print `validated block/end-to-end navigation`.

- [ ] **Step 4: Commit the block and full-path batch**

```bash
git add /Users/linkwind/Code/LTX-2/notes/ltx2-transformer/08-BasicAVTransformerBlock单层结构.md \
        /Users/linkwind/Code/LTX-2/notes/ltx2-transformer/09-音视频双流交互机制.md \
        /Users/linkwind/Code/LTX-2/notes/ltx2-transformer/10-模型前向传播全链路梳理.md
git commit -m "docs: draft ltx2 transformer block and forward notes"
```

### Task 5: Draft the configuration note and the final study map

**Files:**
- Create: `notes/ltx2-transformer/11-关键配置对象与可插拔设计.md`
- Create: `notes/ltx2-transformer/12-LTX-2 Transformer学习地图.md`

- [ ] **Step 1: Write the failing final-batch check**

Run:

```bash
python3 - <<'PY'
from pathlib import Path

root = Path("/Users/linkwind/Code/LTX-2/notes/ltx2-transformer")
required = [
    "11-关键配置对象与可插拔设计.md",
    "12-LTX-2 Transformer学习地图.md",
]
missing = [name for name in required if not (root / name).exists()]
assert not missing, f"Missing final notes: {missing}"
PY
```

Expected: FAIL listing the two missing files.

- [ ] **Step 2: Draft the final two notes with reusable navigation content**

`11-关键配置对象与可插拔设计.md` must include this concept summary:

```markdown
这一层最值得理解的不是“某一个配置项的默认值”，而是 LTX-2 把“模型结构”和“底层算子实现”拆开的设计思路。这样同一个 block 结构，才能在不同 attention backend 或 compile 策略之间切换。
```

`12-LTX-2 Transformer学习地图.md` must include a numbered reading order:

```markdown
1. 先读 `01-LTXModel整体结构与调用主线.md`
2. 再读 `02-TransformerArgs与预处理管线.md`
3. 接着读 `03` 到 `07` 这些机制篇
4. 然后读 `08` 到 `10`，把 block 和全链路拼起来
5. 最后用 `11` 和 `12` 做回顾与扩展
```

- [ ] **Step 3: Verify that all 12 note files now exist**

Run:

```bash
python3 - <<'PY'
from pathlib import Path

root = Path("/Users/linkwind/Code/LTX-2/notes/ltx2-transformer")
files = sorted(p.name for p in root.glob("*.md"))
expected = [
    "01-LTXModel整体结构与调用主线.md",
    "02-TransformerArgs与预处理管线.md",
    "03-AdaLN与时间步调制.md",
    "04-RoPE位置编码机制.md",
    "05-Attention模块总览.md",
    "06-Attention后端与算子选择.md",
    "07-FeedForward前馈网络.md",
    "08-BasicAVTransformerBlock单层结构.md",
    "09-音视频双流交互机制.md",
    "10-模型前向传播全链路梳理.md",
    "11-关键配置对象与可插拔设计.md",
    "12-LTX-2 Transformer学习地图.md",
]
assert files == expected, f"Unexpected file set: {files}"
print("validated all 12 note files")
PY
```

Expected: PASS and print `validated all 12 note files`.

- [ ] **Step 4: Commit the final drafting batch**

```bash
git add /Users/linkwind/Code/LTX-2/notes/ltx2-transformer/11-关键配置对象与可插拔设计.md \
        /Users/linkwind/Code/LTX-2/notes/ltx2-transformer/12-LTX-2 Transformer学习地图.md
git commit -m "docs: finish ltx2 transformer study map"
```

### Task 6: Run the full consistency pass and ship the documentation set

**Files:**
- Modify: `notes/ltx2-transformer/*.md`

- [ ] **Step 1: Scan for placeholders and wording gaps**

Run:

```bash
rg -n "TODO|TBD|占位|稍后|待补充" /Users/linkwind/Code/LTX-2/notes/ltx2-transformer
```

Expected: no output.

- [ ] **Step 2: Verify the common heading structure across all notes**

Run:

```bash
python3 - <<'PY'
from pathlib import Path

root = Path("/Users/linkwind/Code/LTX-2/notes/ltx2-transformer")
required = [
    "## 这个机制解决什么问题",
    "## 它在整体调用链中的位置",
    "## 关键类与函数",
    "## 数据流或张量流",
    "## 关键源码分段解析",
    "## 设计动机与实现取舍",
    "## 一句话总结与下一篇跳转",
]
for path in sorted(root.glob("*.md")):
    text = path.read_text()
    for heading in required:
        assert heading in text, f"{path.name} missing heading: {heading}"
print("validated shared headings for all notes")
PY
```

Expected: PASS and print `validated shared headings for all notes`.

- [ ] **Step 3: Check code references, filenames, and whitespace hygiene**

Run:

```bash
find /Users/linkwind/Code/LTX-2/notes/ltx2-transformer -maxdepth 1 -type f | sort
git diff --check
```

Expected:

- `find` lists the 12 files in numeric order
- `git diff --check` reports no whitespace or merge-marker problems

- [ ] **Step 4: Commit the consistency pass**

```bash
git add /Users/linkwind/Code/LTX-2/notes/ltx2-transformer
git commit -m "docs: polish ltx2 transformer notes series"
```

## Spec Coverage Check

- `notes/ltx2-transformer/` location: covered by Task 1.
- stable `01-xxxx.md` outline: covered by Tasks 1 through 5.
- mechanism-first drafting order: covered by Tasks 2 through 5.
- entry-to-intermediate writing structure: enforced by Task 1 template and Task 6 heading validation.
- review/navigation closeout note: covered by Task 5 through `12-LTX-2 Transformer学习地图.md`.

## Placeholder Scan

This plan intentionally avoids `TODO`, `TBD`, and “same as above” shortcuts. Every batch names exact files, exact validation commands, and exact commit points.

## Type and Naming Consistency Check

- All note filenames match the approved spec names.
- All shared section headings use one exact canonical form.
- Cross-note references use concrete filenames rather than ambiguous labels like “previous note” or “next chapter”.

