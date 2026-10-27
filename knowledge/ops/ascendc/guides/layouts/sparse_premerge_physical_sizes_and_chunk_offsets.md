---
type: Guide
title: 稀疏预合并的物理大小与 Chunk 偏移
description: 补齐 NZ 暂存容量、聚合中逐块搬运、超大块和每轮偏移重置。
tags:
- task.design
- topic.compute.indexing
- topic.data_movement.layout_transform
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P76_v_template_kv_premerge.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P77_l1_chunk_iterator_sparse_aggregation.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P76.yaml
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P77.yaml
  role: supporting
created_at: '2026-09-11T02:07:16Z'
updated_at: '2026-09-11T02:07:16Z'
platforms: []
---

# 稀疏预合并的物理大小与 Chunk 偏移

稀疏KV可经Vector收集、必要布局转换和GM workspace暂存，再由Cube按连续消费布局加载。来源示意4*512*576*sizeof(half)=2359296字节，即2.25MiB；这是指定四份、512×576、half的payload算例，K/V分别存储、NZ padding、实际并发槽及metadata都会改变总量。不能固定分配2.25MB适配所有输入。

每个稀疏逻辑块先经索引/页表求实际K/V地址，预合并后仍需保留原token映射供mask/位置与输出使用。ND→NZ应由实际受支持接口与正确矩阵轴实现；TransDataTo5HD名字本身不证明产生所需NZ。源mergeOffset+=blockSize*headDim只有物理输出恰为这个连续元素数时才完整，有padding需使用物理大小。代码将GM写成LocalTensor及示意LoadData重载均未被本批认证可编译。

L1 chunk聚合是将若干不连续源块放入一个可消费区间；源例仍对每个块分别发搬运，不等于一条DMA覆盖任意离散GM。每个新chunk的局部offset必须重置，实际累加单位与L1容量一致，并包含dtype、padding和布局间隔。64KiB是来源KP分区实例，不是整个L1的统一容量，更不要求所有chunk恰好填满。

迭代器Advance遇到当前块大于整个可用区而直接break，会返回0且不前进，外层while可永久循环。等于容量可作为一个合法满块；大于容量不能仅称为“单块迭代”，必须拆成合法子块或选择能容纳的另一条路径。零大小块要有明确推进，容量0须处理，固定blocksPerChunk也必须大于0。尾chunk未满合法，不能为填满而纳入无效token。

causal裁剪可跳过全无效区，但部分有效块需保留合法部分，FastAdvance应同时更新索引和物理/逻辑进度。Vector发布暂存后Cube才可读取，槽再次使用前要等待消费完成。sparseBlockSize<=4仅为路径候选，非paged但同样存在离散访问时也可评估聚合，不能把PA标签当数学前提。

验证超大块、零块、容量恰好、跨页、最后半块及每chunk局部地址，比较直接稀疏与预合并结果，并将Vector准备、格式转换、同步、空间与Cube连续读取一并计时。基本映射原则见[稀疏暂存边界](sparse_indices_and_contiguous_staging.md)。
