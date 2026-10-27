---
type: Guide
title: S1 三槽流水与排空身份
description: 修正模3时间标签和伪任务索引，保留L1Reuse配对。
tags:
- task.implementation
- topic.attention
- topic.implementation.synchronization
- topic.precision
- topic.implementation.tiling
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/fa-sink-handcraft-recipe.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/fa-kernel-recipes.md
  role: supporting
created_at: '2026-09-11T05:27:29Z'
updated_at: '2026-09-11T05:27:29Z'
platforms: []
---

# S1 三槽流水与排空身份

S1选定路径每task覆盖一块query并一次处理该任务全部有效KV；s1s2则有KV内层。三槽extraInfo为调度元数据，S/P/O两槽为数据存储，两种深度不能混用。若当前taskId=t，(t+2)%3对应t-1，(t+1)%3对应t-2；来源将它们分别叫task-2和task-1颠倒。

按所列代码，提交QK(t)，消费QK(t-1)做softmax并提交PV(t-1)，等待/消费PV(t-2)做Vec2。每次复用同一Matmul对象、GM槽或metadata前应满足对应完成条件；只能保存三条extraInfo不证明允许无限未完成Matmul请求。WaitBmm1/2、End及异步模式需按目标API合同，而非仅以序号解释。

非L1Reuse多2个空迭代用于排空生产链；这些伪task只能消费已有工作，不得先ComputeAxisIdx(idx)和GetS2LoopRange读取超过totalSize的batch/head/长度。空核、单task、尾task以及奇数task都需独立推演，不能因notLast抑制了Iterate就认为前面的元数据访问安全。

L1Reuse来源以blockIdx/2给一对核分共同段、奇偶各自处理对应任务，tiling splitFactor和limit额外排空/补偶要整体匹配。单独乘2会重复或遗漏，单独开模板也会错；但D64–128不是所有FA必须开L1Reuse的数学条件。S1示例禁用、s1s2候选启用是其策略差异。

连续任务段按ceil(total/core)切可能有闲核；稀疏sparseStartIdx表按估算工作量分配，不可只看任务数。QK/softmax/PV在不同硬件单元执行，即使同一global入口也不表示AIC/AIV变为同一物理核或只需本核事件，框架消息协议仍存在。

来源s1s2章节称mm1Res和stage1Res全在UB，S1具体GlobalTensor和workspace公式却明确S/P落GM。应按实际实例的地址空间选择资源和同步；不能以“单核融合”标题证明无GM往返，也不能凭该标题归因2–3倍收益。
