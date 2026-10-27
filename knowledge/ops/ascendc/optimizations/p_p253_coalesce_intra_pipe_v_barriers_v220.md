---
type: Optimization
title: "V220 上合并冗余的核内 V 管线 PipeBarrier"
description: "在已验证的 V220（arch22）路径上合并冗余的核内 PipeBarrier<PIPE_V> 调用，同时保留每一个跨管线同步点。原 legacy 锚 P-P253（冲突 ID P-P92）。"
tags: [task.optimization, topic.implementation.synchronization, topic.programming_model.execution_unit.vector]
status: draft
sources: []
created_at: "2026-09-22T05:30:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: [a2]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# 合并冗余的核内 `PipeBarrier<PIPE_V>`（限已验证 V220 路径）

在同一 Vector 管线内的连续矢量指令之间，冗余的 `PipeBarrier<PIPE_V>` 可以合并以省同步开销。**两个边界缺一不可**：

- 只在**已验证的 V220（arch22）路径**上做——其它架构的管线语义未在此验证；
- **保留每一个跨管线同步点**（V↔MTE、V↔Scalar 等）——合并只限核内纯 V 段。

误删跨管线屏障的后果不是性能回退而是数据竞争，属正确性风险；拿不准的屏障一律保留。
