---
type: Guide
title: 常驻参数与跨迭代状态的资源预算
description: 区分循环不变量、递推状态和阶段内中间值，评估 UB/L1 常驻复用的容量、同步与数值代价。
tags:
- task.optimization
- topic.memory
- topic.implementation.tiling
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P81_resident_ub_tbuf_reuse.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P34_weight_load_once_reuse.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P35_tbuf_resident_accumulator.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P37_grad_accumulator_cross_batch.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P38_softmax_state_resident.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P41_gamma_scale_resident.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P44_transpose_buffer_resident.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P53_l1_resident_reuse.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P83_l1_resident_reuse_partition.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P18_l1_7buf_resident_partition.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P69_ub_fused_vector_chain.md
  role: supporting
created_at: '2026-09-10T13:32:30Z'
updated_at: '2026-09-10T13:32:30Z'
platforms: []
---

# 常驻参数与跨迭代状态的资源预算

重复使用的权重、scale、广播结果或 Q 矩阵，可以在循环外完成搬运或变换，并在后续迭代复用。成立条件是这段生命周期内的逻辑数据、布局及转换要求保持不变；切换 batch、head、group 或 tile 后，需要重新判断复用范围。

常驻对象有不同语义。只读参数依赖首次发布后的可见性；归约累加器、online softmax 的 max/sum 等状态需要正确初值并接续上一迭代；同一 tile 的 Vector 计算链则可以直接消费片上中间值。三者不能共用一个“已经初始化”的判断。递推状态通常只能跨该核实际拥有的工作项累积，跨核的部分结果仍需明确合并。

## 容量与生命周期

常驻字节数应计入整个有效生命周期内的峰值资源预算。Cast 扩宽、广播展开、索引表、对齐、预加载槽位和 API scratch 都可能比原始参数更大。常驻区增加后，主 tile 或双缓冲深度可能需要下降；减少一次搬运不保证整体更快。

来源的 MLA 示例采用四个 QP 槽和三个 KV 槽，每槽 72 KiB，合计 504 KiB。这是特定布局的容量演算，不能作为其他芯片的可用 L1 容量或通用七槽策略。应按实际 Q/K/V 的物理布局和同时存活范围划分常驻区与轮转区，并为阶段边界、对齐及其他使用者留出真实预算。

`TBuf` 的长生命周期不自动提供同步。对常驻状态的连续更新仍须维持生产消费关系；使用 `ReinterpretCast` 也不完成数值转换、不增加容量，重叠 Cast 必须得到对应重载的明确支持。来源中固定 2 KiB 参数区、12 KiB softmax 状态区或“迭代至少三次必有收益”不作为一般结论。

累计结果延迟到循环末尾写回可减少 GM 交通，但会延长状态存活期，改变浮点累加顺序并推迟其他消费者看到结果的时间。验证应包括初值、跨多轮更新、最后一轮写回、数值范围以及中途切换逻辑数据集的情况。

通过相同输入下的 DMA 量、实际 tile 大小、片上峰值资源和完整时延判断收益。可配合[缓冲所有权](buffer_ownership_and_slot_layout.md)、[归约存活预算](../reductions/live_buffer_budget.md)和[转换链约束](../precision/conversion_chain_and_storage.md)审视具体实现。
