---
type: Guide
title: Ascend C 缓冲所有权、流水交接与槽位布局
description: 区分 TQue 深度和实际缓冲块数，说明手工内存、队列和多阶段共享槽位的布局及同步约束。
tags:
- task.design
- topic.memory
- topic.implementation.synchronization
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-api-best-practices/references/api-buffer.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-api-best-practices/references/api-pipeline.md
  role: supporting
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/basic_api/sync_control/intra_core_sync/pipe_barrier_isasi.md
  role: supporting
created_at: '2026-09-10T12:49:29Z'
updated_at: '2026-09-10T12:49:29Z'
platforms: []
---

# Ascend C 缓冲所有权、流水交接与槽位布局

`TQue` 的模板 depth 描述连续入队等队列能力；`InitBuffer(que, num, size)` 的 num 决定实际缓冲块数量。二者不能互相替代。双缓冲增加可轮转的存储，但只有数据依赖和资源安排允许不同 tile 的搬运与计算重叠时，才可能获得收益。已有 [DoubleBuffer 指南](../operator_practice/simd_operator_optimization/pipeline_scheduling/enable_double_buffer.md)说明其机制和不收益场景。

`AllocTensor` 取得存储，不意味着该存储中的异步生产已经完成。队列交接或匹配的硬件事件应连接真正的生产流水与消费流水。单流水 `PipeBarrier` 约束该流水内部顺序，不能单独承担跨流水的数据可见性；来源中用单流水栅栏替代 MTE→Vector 交接的示例不作为本卡实现依据。

`TBuf` 提供存储而不提供 TQue 的入队/出队所有权协议。手工 LocalTensor 或裸指针同样需要显式维护布局和交接。不能把逻辑位置或分配方式当作数据已准备好的证明。

## 共享布局与资源

同一 Kernel 混用固定 UB 区域和 TPipe 分配时，必须确保物理区间不重叠。若采用占位分配隔离静态区域，应验证真实分配布局确实落在预期区间；声明一个 TBuf 本身不证明隔离成立。

多阶段共享 L1/L0/UB 槽位时，实际分配的 slot 字节数、元素数、轮转索引和每阶段的 slot stride 必须一致。偏移为零的首轮可能掩盖错误，后续轮转才越界。可将槽位布局定义作为共同来源，同时检查 `offset + access_size` 不超出该槽容量。

总预算包含所有同时存活的缓冲、双缓冲倍数、对齐和 API 临时区；事件资源也需按实际流水对及实现约束合并核算，不能把“典型 8 个”解释成所有核、所有队列的统一配额。

事件复用应有合法初态、对应消费和循环结束时的收尾。配对关系可跨 tile，但不能重复 Set 一个仍未消费的事件，或无条件等待从未发布的槽位。验证应覆盖零/一/多 tile、每个轮转槽及不同 stage 组合。
