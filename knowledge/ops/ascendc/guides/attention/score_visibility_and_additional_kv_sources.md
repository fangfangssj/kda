---
type: Guide
title: Attention 的 score 可见性与额外 KV 来源
description: 区分 causal/显式 mask 与额外 sink KV，保证有效列集合和 online softmax 的统一归一化。
tags:
- task.design
- topic.compute.reduction
- topic.compute.composite
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-translator/references/attention-patterns/references/mask_causal_pattern.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-translator/references/attention-patterns/references/sink_pattern.md
  role: supporting
created_at: '2026-09-10T14:32:33Z'
updated_at: '2026-09-10T14:32:33Z'
platforms: []
---

# Attention 的 score 可见性与额外 KV 来源

Mask 约束 score 的可见列。无效 K/V 的零填充不能替代 softmax 前的 mask，因为零点积仍可能得到非零概率。显式 mask 的极性、dtype、广播规则及加性或布尔含义由接口规定，不能固定为某条 uint8→half→float 转换链。

Causal 判断应使用同一个 batch 内的逻辑 Q/K 位置，不能使用物理 page 顺序或全局拼接后的 T 行号。左上对齐与右下对齐是不同合同；后者常写为 `k<=q+(Lkv-Lq)`，该偏移不是所有 causal 接口的默认值。整 tile 跳过也必须对所有有效 query 行成立，并保持流水参与者与通知配对。

当接口额外提供独立 sink_k/sink_v 时，它们是另一组 KV 来源。若数学合同规定它们与 local KV 一起参加 attention，sink 与 local 的 score 必须在同一归一化域中合并；不能将两次独立 softmax 的输出直接相加。可沿一套 online 状态逐块处理，或按等价的分区状态归并。

本卡的 sink 专指额外 KV 张量；其他接口可能把同名字段定义为只进入分母的 logit，需要另读接口合同。各来源有各自 layout、有效长度和 tail；独立连续 sink 不必复用 local 的 page table。K 的来源选择还必须延续到对应 V 的加载，不能只在 QK 阶段接入新来源。

验证包括全掩码行、零有效 key、长度不等的 causal、sink/local 不同尾块以及改变 KV 分块后的结果。全掩码状态不能直接套有限 score 更新而产生 `-inf-(-inf)`。状态与归并公式见 [online softmax](online_softmax_state_and_partition_merge.md)；物理地址见[逻辑 token 与缓存地址](logical_tokens_heads_and_cache_addresses.md)。
