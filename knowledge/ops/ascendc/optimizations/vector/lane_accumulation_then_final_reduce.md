---
type: Optimization
title: 先按 Lane 累加再做最终归约
description: 将多块的逐 Lane 累计与最后一次横向归约分开，明确尾部掩码、状态寿命与累计顺序。
tags:
- task.optimization
- topic.compute.reduction
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/R6_low_latency_reduce.md
  role: primary
created_at: '2026-09-11T01:27:48Z'
updated_at: '2026-09-11T01:27:48Z'
platforms: []
---

# 先按 Lane 累加再做最终归约

对求和，可先把每个向量块的同一 lane 累计到向量状态 acc，完成全部块后再横向归约 acc。若向量宽度为 V，第 j 个状态表达 `acc[j]=sum_k x[k*V+j]`，只包括有效输入。最终 `sum_j acc[j]` 给出总和。这可减少反复把每个块归约到标量的依赖和指令，但不承诺固定 2–3 倍收益。

所有 acc lane 在第一次读取前需要零初值。尾块的无效 lane 必须不贡献新值，同时保留先前块的累计结果；不能因本轮 mask 较短就在最后把其余 acc lane 丢掉。具体 Add/Load 的 inactive lane 行为须按重载核对，必要时显式填零或合并状态。

完成累加后可在同一 VF 内横向归约，或把完整有效累计状态写到 UB 再由后续接口归约；后者增加一次存储和交接，且跨 VF 状态必须有真实的存储载体。一次全 mask Store 的安全性取决于已初始化状态和目标容量，不能由有效输入数直接推断。

这种组织改变浮点累加顺序，需核对累计 dtype、抵消误差和最终输出舍入。空输入、不到一个 V、V+1、多块尾部、正负抵消和大规模累计都要验证。mean 还要保存真实有效 count；max/min 需改用各自中性值及 NaN/Inf 规则，不能机械复用 sum 的零初始化。
