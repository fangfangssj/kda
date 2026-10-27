---
type: Guide
title: MoE 通信窗口的数据发布、状态消费与复用
description: 区分数据区与状态区的索引布局，按先完成数据、再发布状态、消费后复用的依赖管理多轮通信窗口。
tags:
- task.design
- topic.memory
- topic.implementation.synchronization
- topic.mixture_of_experts
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/api-rules/sync-and-visibility.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/tiling-scheme/window-memory-layout.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/tiling-scheme/double-buffer-protocol.md
  role: supporting
created_at: '2026-09-10T12:37:22Z'
updated_at: '2026-09-10T12:37:22Z'
platforms: []
---

# MoE 通信窗口的数据发布、状态消费与复用

窗口协议需要同时规定数据地址、状态地址和每轮的生命周期。来源方案的 Dispatch 数据按来源 rank、目标本地专家、token 槽组织，状态则按本地专家、来源 rank 组织；两者的维度顺序不同，不能共用同一个展平公式。

Combine 的回传区按原 token 与 topK/共享专家槽组织。数据 stride 和状态 stride 各自按实际格式及接口对齐要求计算；有效 payload、附加元数据和记录总步长也应区分。具体实现可能使用独立状态表或把到达标志嵌入数据记录，二者 ABI 不可混用。

## 发布与消费

就绪状态表达的是“该轮对应数据可以被读取”。因此先要完成数据写入及必要的可见性保证，再发布状态；接收者确认对应状态后消费数据，消费完成后才清理状态或允许下一轮覆盖。

核内流水顺序、同卡跨核阶段同步和跨 rank 通信完成是不同条件。SyncAll 的具体重载控制同卡参与核，不等价于所有 rank 已完成通信，也不能独立保证任意 DCache/GM 路径的一致性。

## 四区轮转的条件

来源提出 dispatch-0、combine-0、dispatch-1、combine-1 四个区域，Dispatch 与 Combine 分别维护自己的选择标志。这样可隔离相邻阶段和相邻轮次，但安全性还依赖协议对旧轮消费完成的保证。

仅执行 `bufferChosen ^= 1` 不会自动建立远端完成条件。在复用第 N 轮的区域前，需要证明所有相关消费者已经完成该轮，并且选择标志在各参与者之间具有一致的轮次含义。快 rank 不能覆盖慢 rank 尚在消费的数据。

## 验证

构造 rank 间速度不一致、空 token、连续多轮、两次以上区域复用和不同核数的用例。观察数据内容、状态轮次、有效计数和复用时机。单轮成功不能证明双缓冲协议安全；完整 950 算子的实际窗口布局见 [Dispatch 实现卡](../../operators/aclnn/transformer/mc2/moe_distribute_dispatch_950.md)。
