# 06 Attention后端与算子选择

## 这个机制解决什么问题

同样都是 attention 计算，不同硬件、不同安装环境、不同 mask 需求下，最快的实现并不一样。LTX-2 的做法不是把 backend 写死，而是把 attention 底层调用拆成可切换的后端。

因此，这一篇不再关心“QKV 是什么”，而关心“真正做矩阵计算的是谁”。

## 它在整体调用链中的位置

它位于 `Attention.forward()` 的真正数值核心位置：

`q/k/v -> attention backend -> out`

从代码结构看，路径大致是：

- `AttentionOps` 保存 callable；
- `AttentionFunction` / `MaskedAttentionFunction` 把枚举解析成具体 callable；
- `Attention.forward()` 在运行时根据 `mask` 选择 unmasked 或 masked 路径。

重点阅读文件：

- `packages/ltx-core/src/ltx_core/model/transformer/attention.py`
- `packages/ltx-core/src/ltx_core/model/transformer/ops.py`

## 关键类与函数

- `PytorchAttention`
- `XFormersAttention`
- `FlashAttention3`
- `FlashAttention4`
- `automatic_attention()`
- `automatic_masked_attention()`
- `AttentionFunction`
- `MaskedAttentionFunction`
- `AttentionOps`

## 数据流或张量流

这一层的输入输出形状目标其实没有变，变化的是“谁来算这一步”。

统一输入大致是：

- `q`
- `k`
- `v`
- `heads`
- 可选 `mask`

统一输出大致都是：

- `(B, T, H * D)` 形状的 attention 结果

换句话说，后端抽象的关键不在于“数学定义不同”，而在于“同一接口下，底层 kernel 不同”。

## 关键源码分段解析

### 1. `PytorchAttention` 是最通用、最容易兜底的实现

它内部最终还是调用 PyTorch 的 `scaled_dot_product_attention`，但通过 `sdpa_kernel(..., set_priority=True)` 把 backend 优先级显式交给 torch dispatcher。

这带来两个好处：

- 一份接口能覆盖 cuDNN、Flash、Efficient、Math 等多种 SDPA backend；
- 即使高性能实现不可用，`MATH` 也能兜底。

### 2. `XFormersAttention` 和 `FlashAttention3/4` 是更激进的加速选项

这些类的核心作用不是改变 Attention 语义，而是：

- 更适配某些 GPU 架构；
- 在特定形状下更省显存或更快；
- 把 backend 差异封装在同一个 callable 协议里。

但它们也有明显约束，比如：

- 某些实现不支持 mask；
- 某些实现依赖额外安装包；
- 某些实现只在特定 GPU 架构上值得优先选择。

### 3. `automatic_attention()` 和 `automatic_masked_attention()` 是默认策略的核心

这是整套设计最有工程味的一点：

- 用户不一定显式指定 backend；
- 框架会根据当前环境自动挑一个最合适的实现。

其中 unmasked 路径和 masked 路径还是分开选的，因为“支持 mask”本身就是一个强约束条件。

### 4. `AttentionFunction` / `MaskedAttentionFunction` 把“配置值”变成“可执行对象”

这两个枚举的意义在于把配置层和执行层接起来。

上层配置只需要说：

- 我想用 `AUTOMATIC`
- 或者我明确要 `XFORMERS`
- 或者我想钉死某个 SDPA backend

真正运行时再由 `to_callable()` 解析成对应对象。如果环境不支持，代码会直接抛异常，而不是静默降级到另一个完全不同的 backend。

### 5. masked 路径是单独建模的，不是顺手兼容

`MaskedAttentionFunction` 没有把所有 unmasked backend 都照搬过来。像 `FLASH_ATTENTION_3/4` 这类不支持 mask 的实现，会直接被排除在 masked 枚举之外。

这是一种很稳妥的接口设计：让“某 backend 根本不支持 mask”尽可能提前暴露，而不是拖到运行时深处才炸。

## 设计动机与实现取舍

这套设计最值得学的地方，是它把“模型结构”和“内核选择”解耦了。

- block 只关心自己需要一个 attention 模块；
- `Attention` 只关心自己拿到了哪组 callable；
- backend 选择逻辑集中在 `attention.py` 里。

这样同一个模型结构可以适应：

- CPU 或无额外加速包环境；
- 安装了 xFormers 的环境；
- Hopper / Blackwell 这类更适合 FA3/FA4 的 GPU。

不要混淆两件事：

- [05-Attention模块总览.md](/Users/linkwind/Code/LTX-2/notes/ltx2-transformer/05-Attention模块总览.md) 讲的是 Attention 类本身怎么接输入、拆 QKV、走 self/cross 两种路径；
- 本篇讲的是同一套 attention 接口背后，为什么还能切到 SDPA、xFormers 或 FlashAttention。

## 一句话总结与下一篇跳转

这篇的核心结论是：LTX-2 把 attention 的“结构定义”和“底层 kernel”拆开了，所以同一份模型代码可以跨环境切后端。下一篇 [07-FeedForward前馈网络.md](/Users/linkwind/Code/LTX-2/notes/ltx2-transformer/07-FeedForward前馈网络.md) 会暂时离开注意力，去看 block 里另一个稳定的容量来源。
