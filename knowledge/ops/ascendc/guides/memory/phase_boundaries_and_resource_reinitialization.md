---
type: Guide
title: 串行计算阶段的资源重建与预计算结果交接
description: 说明阶段拆分时片上资源回收、独立 TPipe 生命周期和 GM 预计算结果的可见性。
tags:
- task.design
- topic.memory
- topic.implementation.synchronization
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P88_multiphase_compute_phase_decomposit.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P91_softmax_grad_precompute.md
  role: supporting
created_at: '2026-09-10T13:32:30Z'
updated_at: '2026-09-10T13:32:30Z'
platforms: []
---

# 串行计算阶段的资源重建与预计算结果交接

前处理、主循环和后处理对片上资源的需求差异较大时，可以使用明确的串行阶段，使每阶段选择自己的缓冲布局。前一阶段资源仍被异步搬运或计算使用时，不能直接 Reset 或重新分配。

阶段切换应先完成所有相关生产者和消费者，再结束旧队列、事件及存储的使用，最后建立新阶段资源。新建另一个 `TPipe` 也不意味着物理内存已经隔离或旧操作完成。旧 LocalTensor、切片和事件编号不能在资源重建后继续按原含义使用。跨核边界需遵守实际参与范围，不能认为一次局部栅栏或任意形式的 `SyncAll` 能覆盖全部执行模式。

另一类拆分把主循环反复使用的子结果提前算入 GM workspace。例如 Flash Attention 反向中的每行 `sum(dY*O)` 可以在满足对应梯度公式、shape 与 dtype 语义时预计算；后续阶段读取同一 query 行的结果。它减少重复计算，同时增加 workspace 写读及发布开销。

预计算结果的逻辑行号、物理布局和有效长度必须与消费端一致。不能因移出循环就改变 mask、广播、scale 或数值精度；若子结果依赖循环内变化的数据，也不满足前移条件。每个结果的写入应完成后再允许读取，并保留最后一轮和空输入的行为。

验证应同时覆盖阶段切换的资源峰值、旧引用失效、跨核发布和数值结果。阶段越多，启动与同步固定成本可能越高；只有被省去的重复工作足以抵消这些成本时，才形成实际收益。单块成功不能替代多块、多轮验证。
