---
type: Guide
title: 稀疏索引寻址与连续暂存的语义边界
description: 分离逻辑索引、字节偏移、分块迭代和稀疏 KV 预合并，保留边界与写入归属。
tags:
- task.design
- topic.compute.indexing
- topic.data_movement.layout_transform
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P45_topk_index_cache_iterator.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P32_scatter_block_stride.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P33_gather_offset_table.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P76_v_template_kv_premerge.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P77_l1_chunk_iterator_sparse_aggregatio.md
  role: supporting
created_at: '2026-09-10T13:32:30Z'
updated_at: '2026-09-10T13:32:30Z'
platforms: []
---

# 稀疏索引寻址与连续暂存的语义边界

稀疏访问至少包含逻辑索引到源位置、源位置到物理地址、物理数据到消费布局三层映射。索引缓存或 chunk 迭代器可以减少重复标量计算，但不能改变原算子的索引语义。

例如二维 scatter block 的元素偏移可以写为 `i0*stride0 + i1*stride1`，前提是 stride 使用元素单位且更新块后续维连续。Gather 的偏移表则需遵守所选重载的单位；不能把元素偏移直接当成字节偏移。预计算表只有在 dtype、布局和基地址关系不变时才能复用，其容量与整数表示范围也应计入预算。

重复 scatter 索引的覆盖、累加或未定义行为由算子契约决定。并发写同一位置时，普通搬运不自动实现 scatter-add；索引边界、空索引集、重复项和尾部均须单独验证。

## 稀疏块的聚合

将多个不连续块聚合为一个 L1 chunk，目标是改善消费粒度。实现必须保证迭代器持续前进：当前块大于剩余容量时，要明确拆块或结束当前 chunk，不能在下次仍停留于同一超大块。最后一个 chunk 可以未填满，不应把“恰好填满 64 KiB”作为不变量。

PageAttention/稀疏 KV 也可以先由 Vector 路径收集到连续 GM workspace，再供 Cube 消费。这样将不规则寻址和可能的格式变换前移，但增加一次暂存写读、同步和 workspace；是否划算取决于块大小、复用次数和流水重叠，不能固定以 sparseBlockSize≤4 作为跨平台决策。

预合并必须保存原逻辑位置或其等价映射，使 mask、位置编码、输出还原与原有效 token 集合一致。跳过 causal 无效区、遇到页边界或最后一个不完整 sparse block 时，均需检查映射。消费者读取前应确认暂存数据已经发布，槽位复用则应等待消费完成。

可用包含逻辑行列及页号的输入验证寻址，比较直接稀疏路径与暂存路径的逐元素结果，再比较总搬运量及完整执行时间。周期 Gather 表的一个具体推导见[小通道转置与偏移表](small_channel_transpose_and_offsets.md)。
