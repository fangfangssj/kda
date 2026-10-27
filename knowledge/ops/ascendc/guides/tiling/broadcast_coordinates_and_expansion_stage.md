---
type: Guide
title: 多维广播的坐标映射与展开阶段
description: 以各输入的广播模式和真实 stride 合轴，并比较 GM 搬入时展开与 UB 内展开的资源及接口边界。
tags:
- task.design
- topic.implementation.tiling
- topic.data_movement.layout_transform
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-tiling-design/references/broadcast/patterns.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-tiling-design/references/broadcast/ub-broadcast.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-tiling-design/references/broadcast/nddma-broadcast.md
  role: supporting
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/basic_api/memory_vector_compute/data_move/data_copy_gm_to_ub_nddma.md
  role: constraint
created_at: '2026-09-10T13:12:33Z'
updated_at: '2026-09-10T13:12:33Z'
platforms: []
---

# 多维广播的坐标映射与展开阶段

广播先按算子规则对齐各输入的维度，并检查每个轴的长度是否兼容。对于被扩展的长度为 1 的轴，该输入地址不随输出轴坐标推进，可用零 stride 表示。未广播轴仍使用真实物理 stride；不能把所有输入都重新按连续布局计算。

相邻轴只有在每个输入的广播模式与连续性都允许时才能合并。例如 `[4,3,8]` 与 `[1,3,8]` 可把后两轴合为 24；而 `[2,1,4]` 与 `[2,3,1]` 的广播位置不同，不能将中间与末轴视为同一连续输入段。输出总元素数相同不足以证明合轴保持地址映射。

tile 坐标应从输出空间分解，再分别用各输入的 stride 计算地址。切分轴的最后一块使用实际长度；所有输入的源 tile shape、展开后的目标 tile shape、UB 物理行距和最终 GM 写回范围必须相互一致。

## 广播发生在哪一层

| 路径 | 资源与适用性 |
|---|---|
| 搬入原始数据，再在 UB 展开 | 可能同时保留源区、展开区及 API 临时区，适合已有 UB 中间结果 |
| 支持的多维 GM→UB 搬运中展开 | 可直接写入目标 tile，需满足 DMA 描述符、缓存和布局约束 |
| 计算接口直接复用标量或重复地址 | 可省显式展开，但只适用于接口能够表达的地址模式 |

[NDDMA DataCopy](../../apis/simd_api/basic_api/memory_vector_compute/data_move/data_copy_gm_to_ub_nddma.md)的当前知识范围是 Ascend 950、GM→UB、描述符维度 1–5。更多维度可以通过外层循环分解，但需要分别推进 GM 和 UB 偏移，并验证目标区不交织或重叠。描述符的维度编号、stride 单位和配置类型应核对实际版本，不能把逻辑 shape 顺序直接复制为底层数组顺序。

源 stride 为零表达重复取数的意图，但具体层级、访问粒度和实现限制仍需验证；不能因为某个广播轴无需推进，就认定所有轴的 DMA 都合法。仅切分轴无需广播，也不足以将整个多维访问降级为线性 Copy。

UB Broadcast 的 shape、地址对齐、允许 dtype 和 scratch 要求随重载而变。搬运补齐了物理空间，不表示原始 shape 自动满足某个静态接口的条件；反过来，“无 shape 对齐限制”也不等于地址没有约束。利用 DMA dummy 数据作为复制源时，更需要该模式明确保证填充值，不能依赖未定义 padding。

两条路径的速度取决于重读量、缓存命中、展开写入、临时空间和指令开销。不能把“尾轴超过一半缓存容量”或“省一次 Broadcast 调用”视为所有 shape 的性能分界。验证应包括广播轴在最内/中间/外层、多输入交错广播、尾块以及跨次调用的数据更新。
