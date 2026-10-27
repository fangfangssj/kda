---
type: Guide
title: 反向 Cube 融合需保留全部梯度路径
description: 区分 dS 与概率矩阵，补齐累加器、转置输入和输出写回的寿命。
tags:
- task.optimization
- topic.attention
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P79_grad_cube23_fusion.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P79_grad_cube23_fusion.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P79.yaml
  role: supporting
created_at: '2026-09-11T02:13:08Z'
updated_at: '2026-09-11T02:13:08Z'
platforms: []
---

# 反向 Cube 融合需保留全部梯度路径

Flash Attention反向中，若dS代表分数梯度、P代表概率矩阵，相应矩阵链包括dQ=dS*K、dK=dS^T*Q、dV=P^T*dY；scale、dropout、mask和head归并仍按目标梯度定义处理。来源用S同时指代部分矩阵，不能把dS与P视为同一数据而错误共享。

来源headDim=64且sparseMode=0时调度Cube23，将dS与其转置分别载入L0A ping/pong，将K/Q载入L0B ping/pong，再维护dQ/dK各自结果。两种方向可以共享原始L1数据，但所示骨架仍各调用一次正常LoadData和转置LoadData，并未证明从两次搬运降为一次；实际收益可能来自复用、调度与重叠，需要指令/流量证据。

该片段只写出dQ，未展开dK写回及dV计算，不能当作完整Backward实现。所有必需输出都应保留对应任务坐标、有效M/N、归约维度和最后写回。每个L0C累加链首次初始化，其余贡献累加；交替L0A/B并不要求或允许把同一结果的部分和随意分散到不同L0C后只取一份。

来源四个L0C buffer、合计128KiB是特定块尺寸/份数预算，不能仅因d=64就认定必然需要或容得下。应计算dQ/dK/dV的真实累加dtype、矩阵布局、同时活跃结果和Fixpipe消费寿命，支持其他headDim的实现也可以采用另行证明的融合方案；来源条件只约束这条特化分支。

原子dQ写回还需要目的初始化、正确输出类型、累加范围与退出原子状态，不能由SetAtomicType一行推导完整正确性。验证三条梯度、K/query尾部、GQA共享head归并、多个partial与循环排空，比较误差及总流量/时间；源码省略处未被本批编译或设备验证。
