---
type: Guide
title: MoE Dispatch 与 Combine 的元数据对应关系
description: 将特征、来源三元组、收发计数和量化 scale 按同一有效条目顺序关联，区分最大容量与本轮实际 token 数。
tags:
- task.design
- topic.mixture_of_experts
- topic.data_movement.communication.collective_communication
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/api-rules/dispatch-combine-contracts.md
  role: primary
created_at: '2026-09-10T12:37:22Z'
updated_at: '2026-09-10T12:37:22Z'
platforms: []
---

# MoE Dispatch 与 Combine 的元数据对应关系

在来源描述的 Dispatch/Combine 方案中，每条 routed item 同时包含特征载荷和恢复来源所需的元数据。Dispatch 产生来源 rank、原 token 索引和 topK 槽位；Combine 根据这些信息把专家输出送回并恢复原来的 token 顺序。

`expandX`、来源索引及逐条量化尺度必须维持相同的有效条目顺序。传输窗口内可以把载荷和元数据打包成一条通信记录，正式输出中的特征 tensor 仍不能把附加三元组算进隐藏维。

来源将三元组字段命名为 `expandIdx`，顺序为 `[srcRankId, srcTokenIdx, srcTopKIdx]`。具体公开 API 可能使用 assistInfo 等不同名字和版本布局，集成时应以相应接口为准。

## 数量与容量

最大容量 A 决定输出的预留上界，本轮有效长度由实际收发统计决定。以 A 作为实际接收量可能读到无效记录，作为等待目标还可能等待永远不会产生的 token。

Dispatch 按来源 rank 统计的接收数量，在对应 Combine 方案中成为向该来源 rank 回传的数量。统计顺序、分组和有效条目必须对应，不能只因两个数组形状相同就互换。

expertTokenNums 是聚合后的专家统计，不能代替尚未聚合的原始状态。数量与前缀和模式应分别解释；局部核的部分统计需按算法合并后才能写最终结果。

## 量化和验证

expertScales 表示聚合权重，expandScales 等表示与路由数据关联的尺度；二者数学作用不同。量化模式、输出 dtype 与 scale 的产生和消费须一致。

用可识别的 rank/token/topK 编码验证 round trip，覆盖空接收、部分有效容量、mask、共享专家和量化条目重排。本文只描述 reference 的对应关系；具体 950 实现和公开版本边界见 [MoeDistributeDispatch 950](../../operators/aclnn/transformer/mc2/moe_distribute_dispatch_950.md)。
