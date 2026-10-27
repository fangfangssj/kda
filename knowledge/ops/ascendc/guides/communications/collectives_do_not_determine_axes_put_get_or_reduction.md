---
type: Guide
title: 集合通信不自动决定切轴、PUT/GET与求和
description: 纠正AllToAll示例中的分布混用，保存四种原语的数学边界。
tags:
- task.design
- topic.implementation.synchronization
- topic.implementation.tiling
- topic.performance.analysis
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/operators/collective-comm.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/requirement-analysis/classification.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/requirement-analysis/grill-protocol.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/requirement-analysis/quality-checklist.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/requirement-analysis/template.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/workflow/step2-investigation.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/workflow_integration.md
  role: supporting
created_at: '2026-09-11T07:06:47Z'
updated_at: '2026-09-11T07:06:47Z'
platforms: []
---

# 集合通信不自动决定切轴、PUT/GET与求和

AllGather将各来源分片按约定顺序拼接到每个rank；AllToAll使来源s的目标t分片交给t；AllReduce按reduceOp聚合后向所有参与者提供结果；ReduceScatter先聚合再把结果分片分给rank。M轴、K轴或token轴由具体布局定义，PUT/GET由实现选择，均不是原语固有属性。AllGather可通信在前拉取输入，也可计算在前收集结果；不能把GET等同计算在前、PUT等同通信在前。

若每rank持A[M,K/R]和B[K/R,N]，部分Ci=AiBi具有相同[M,N]坐标，sum_r Ci后可AllReduce完整输出或ReduceScatter分M段。若各rank持完整K的独立输入，跨rank求和也可合法但数学意义变为独立Ci求和，不能称全局K分片Matmul。只给每卡M/R无法判断指输入分片还是最终输出分布。

来源AllToAll段同时写Ai[Mlocal,K]、B可全量或K分片、收集后按rank Matmul累加及获得完整M，条件不闭合。若只是不同来源行段拼接，通常应拼接各行结果而非求和；只有各贡献对应同一输出坐标的K分量，才应跨rank累加。例R个source把各自Mlocal按R目标分，每个target收到R*(Mlocal/R)=Mlocal行，不会自动取得全局R*Mlocal行。token按expert路由通常计数变长，需要AllToAllv等协议语义，不能直接套均分AllToAll。

通信与计算两侧必须一致定义源rank、目标rank、局部/全局shape、有效offset及reduceOp。无归约通信不会替计算端累加；AllToAll PUT+本地规约可实现某些ReduceScatter，但需显式所有来源到达、初值、唯一输出归属与归约顺序，不是交换原语名即可等价。

源算子目录名提供AllReduce/AllToAll/AllGather/ReduceScatter/MoE及barrier的查找线索，不能作为任意dtype/调用形态已支持证据。数学golden应从实际分布推导，覆盖rank1/多、非均匀分片、空分片及各rank独特数据，避免所有输入相同掩盖源/目标错位。
