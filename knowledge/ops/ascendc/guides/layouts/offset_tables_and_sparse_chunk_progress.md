---
type: Guide
title: 偏移表、分块 Scatter 与稀疏迭代进度
description: 补齐索引表生成、块内连续性、分页缓存和 L1 chunk 前进条件。
tags:
- task.design
- topic.compute.indexing
- topic.data_movement.layout_transform
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P32_scatter_block_stride.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P33_gather_offset_table.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P45_topk_index_cache_iterator.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P32.yaml
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P33.yaml
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P45.yaml
  role: supporting
created_at: '2026-09-11T01:49:44Z'
updated_at: '2026-09-11T01:49:44Z'
platforms: []
---

# 偏移表、分块 Scatter 与稀疏迭代进度

二维 block scatter 的目标元素偏移可为 `i0*stride0+i1*stride1`，每个索引关联一段连续更新。前提是 stride 为元素单位，后续 updateDimSize 个元素确实连续，且 index 的最后一维至少包含这两个坐标。源更新在 UB 中的行起点 `i*updateRowElements` 又是独立物理行距，可能含 padding；搬出 blockLen 应为 updateDimSize*sizeof(T) 的有效字节，结构体其余字段仍需完整初始化。

先验证索引语义和地址范围再进行窄化/乘法。int64 结果类型并不保护此前已溢出的表达式，偏移宽度也不能代替上下界校验。GetValue 读取索引之前要有 MTE2→Scalar 的合法完成保证。每行一个 DataCopyPad 可以表达离散目标；只有实际目标块可合并时才可减少写出次数。即使底层 tensor 连续，离散或重复 indices 仍不能直接改为一次顺序 DataCopy。

Gather 的字节偏移表把多个逻辑元素映射到一个可寻址 UB 源区；源逻辑数据不必连续，只要每个偏移对应的元素都在合法物理范围。来源 pre/postOffset 循环在 j 变化时重复写 `curOffset*sizeof(P)`，会将同一源元素重复提取 N 次。若目标是长度 N 的连续行，应根据真实映射加入 j 等列偏移；若目标本来是重复广播，则重复偏移可以是合法意图，不能未经语义判断自动改写。

两份各 V1_BASE_T*N 个 uint32 偏移表的基础空间是 `2*V1_BASE_T*N*4` 字节，另计对齐。运行 lenT*N 不得超已生成条目。表需初始化 offset1/offset2/curOffset，乘 sizeof(P) 前后防溢出；变更 N、dtype、nSquare 或源布局关系后应重建。标量 SetValue 填表后还要满足 Vector Gather 消费的同步。固定相对布局可只更新基地址而复用表，但不能把不同任务基址错当同一逻辑数据。

稀疏 Attention 的块索引缓存与 L1ChunkIterator 分开负责缓存页面、当前块、块内 token 游标、当前 chunk 已占空间和完成标记。每次 Next 要么输出非空有效段并推进游标，要么跳过无贡献块/完成当前 chunk，要么终止。一个块大于剩余容量时需拆出能放入的前缀，未消费部分保留到下一 chunk；仅写 `current+length>=capacity` 然后仍返回整段会越界。

当前块耗尽后推进 blockIndexCursor，完成 chunk 后重置相应容量状态，全部完成时返回 false。来源片段缺少这些更新及最终返回，还包含未定义的 chunkEnd，不能据此证明迭代器可运行。L1 容量的单位应与 segment 长度统一，数据宽度、格式 padding、Q/K/V 其他占用需一起预算。

TOPK_CACHE_SIZE=128 是样例数组容量，不能截掉超过128的有效索引；可以分页或扩容并计成本。合并相邻块仍要保持 token 集合、顺序要求、mask/位置映射和重复索引合同。L1 chunk 不满也要输出，页边界需更新 GM 起点；若选择先 Gather 到 GM 再供 Cube 消费，还需 workspace 和发布/回收协议。

验证空索引、跨页、重复/逆序索引、超过缓存容量、超大单块、最后不满块和 j>0 的元素坐标。索引缓存与逐元素访问哪种更快取决于缓存构建、额外 UB、标量下发和后续复用次数，不能靠类名或一次 Gather 保证收益。
