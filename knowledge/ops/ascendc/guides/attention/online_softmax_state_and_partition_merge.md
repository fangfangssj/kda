---
type: Guide
title: 分块 Attention 的 online softmax 状态与归并
description: 保持 max、指数和与未归一化加权和一致，正确处理跨 KV 分块缩放、Split-KV 和分块 softmax 输出。
tags:
- task.design
- topic.compute.reduction
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-tiling-design/references/flashattention/foundation/fundamentals.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-best-practice/references/scenarios/cube-matmul-then-vector-softmax-reduce/online-softmax-reduce-epilogue-design.md
  role: supporting
created_at: '2026-09-10T14:01:59Z'
updated_at: '2026-09-10T14:01:59Z'
platforms: []
---

# 分块 Attention 的 online softmax 状态与归并

对一个 query 行，分块 Attention 可以保留当前最大 score m、指数和 l，以及未归一化加权和 a，逐个 KV 块更新，而不在 GM 中物化完整 attention score 矩阵。score 应包含该接口所需的 scale、mask 等变换；max、sum 和加权和必须对应同一组有效 key。

对新块的有限有效 score s 和对应 value V，令 m_new=max(m,max(s))、alpha=exp(m-m_new)、p=exp(s-m_new)，更新 l_new=alpha*l+sum(p)，a_new=alpha*a+p·V。处理完全部 KV 后再求 a/l。旧指数和和旧加权和必须一起重缩放；只更新 max/sum 而遗漏 a 的缩放会改变结果。

每个并行 KV 分区可以独立形成 (m_i,l_i,a_i)。归并时令 m=max_i(m_i)，再计算 l=sum_i(exp(m_i-m)*l_i)、a=sum_i(exp(m_i-m)*a_i)，最后输出 a/l。由此可见，Split-KV 需要显式归并，不能直接相加各分区已归一化的输出。该等式针对同一 query 的不重叠、完整 key 分区；重复或遗漏 key 会改变 attention。

空分区和全掩码行需要显式语义。直接对初始化的 m=-inf 做 -inf-(-inf) 会产生 NaN，不能只把 l 和 a 初始化为零就假定所有空路径正确。可按接口定义跳过无贡献分区，并为整行无有效 key、NaN/Inf score 等情况规定行为。上述有限有效 score 的公式本身不替接口决定这些特殊值语义。

对于需要输出完整 softmax 概率的 Matmul+softmax 场景，若某 tile 保存的是基于历史最大值 m_t 的 exp(s-m_t)，最终归一化需乘 exp(m_t-m_final)/l_final。保存了指数块却丢失其对应 m_t，会使后续跨核归一化用错缩放；这与 Attention 只保留加权和的存储方案不同。

状态应按 query/head/task 隔离，并与流水槽的生命周期一致。大 value 维度可能使 a 无法常驻 UB，可采用 GM 暂存，但每次读改写仍需遵守相同数学状态更新和同步条件。GQA 合并 query heads 是减少重复 KV 搬运的一种调度选择，不是 attention 正确性的必需条件。

Ascend C 的具体 Softmax 接口还有独立的 dtype、形状和临时空间约束，不能由数学公式推出任意调用均受支持。验证可用小规模明确计算的 attention 对照，改变 KV 分块和分区数，覆盖尾块、mask、空分区、不同 query/value 维度以及多个流水槽；浮点归并顺序变化需要按目标数值合同评估。
