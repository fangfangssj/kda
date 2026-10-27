---
type: Guide
title: 自定义归约树工作量与 Welford 计数
description: 保存半区间和并行更新，纠正复杂度与浮点计数边界。
tags:
- task.optimization
- topic.precision
- topic.implementation.tiling
- topic.implementation.synchronization
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/cake-evo/references/meta_prompts/strategies/perf_06_multi_algo_selection.md
  role: primary
created_at: '2026-09-11T05:59:16Z'
updated_at: '2026-09-11T05:59:16Z'
platforms: []
---

# 自定义归约树工作量与 Welford 计数

半区间求和可令p为≤N最大2次幂，先把尾N-p与前N-p相加，剩p项再逐次折半，最后用匹配有效mask的WholeReduceSum归约。每层依赖前层完成，N0、N≤64、尾部和原地重叠按API处理。二叉树深度O(logN)，总标量加法仍N-1即O(N)工作量；4096项约12层不等于只12次算术或必比已有向量ReduceSum更少指令。

ReduceMaxInplace按64FP32块逐lane累积再WholeReduceMax，需要正确首块状态、offsetsFp32、末尾mask与repeat字段范围。源offsets/mask未定义，N<64时不能无条件读满64，N0没有Max中性结果默认值；跨repeat同dst依赖也须实际重载允许。只有参与count与布局完全等价时，才能将手写Max与通用ReduceMax比性能，source拿ReduceSum当Max对照不构成同一算子。

BatchNorm Welford可并行处理多个channel，源更新count=n、delta=x-mean_old、mean+=delta/n、M2+=delta²*(n-1)/n在实数域等价标准Welford增量，但浮点运算次序不同。count来源float在超过2^24后逐次+1可能不再变化；应以可表示计数或分块计数/宽整数设计完整大N域，不能声称任意长均安全。n1首次更新需mean/M2有定义，xTensor被用作delta/n临时后原输入已覆盖，下游需要原值时另保存/重读。

合并分块状态(n_a,mu_a,M2_a)、(n_b,mu_b,M2_b)时，delta=mu_b-mu_a，n=n_a+n_b，mu=mu_a+delta*n_b/n，M2=M2_a+M2_b+delta²*n_a*n_b/n；不能只相加M2而漏组间项。R0/R1切分与尾块计数要对应真实有效值，padding不能进入n。

WelfordInitialize→多次Update→Finalize需处理welfordUpdateTail，示例只循环整tile不能证明任意N覆盖；Finalize<true>的具体rstd/variance和epsilon参数按接口，不从模板true补猜。统计一遍不代表整个归一化只读输入一遍：若未保留原数据仍需归一化输出pass，GM流量不会自动减50%。中心化two-pass也可稳定，Welford不是所有分布下无误差最优。

Regbase BinaryAdd用LocalMemBar<VEC_STORE,VEC_LOAD>保障相应局部存取关系，不能替代跨核同步；uint16 loop计数须验证规模范围。Reduce/Brcb的64/128是来源dtype路径组织，广播scale需按quant block对应每元素，不能让多个块scale错配。Abs+Max有限值比较不因BF16低尾数自动丢相邻已表示值，选择FP32主要应依据实际指令支持与后续算术。
