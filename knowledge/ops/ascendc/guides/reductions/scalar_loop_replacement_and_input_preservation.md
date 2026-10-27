---
type: Guide
title: 标量归约替换与原输入保留
description: 保留 sum/sum_sq 的有效域、输出发布和空间约束，区分 API 名称与具体重载。
tags:
- task.optimization
- topic.compute.reduction
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P94_scalar_reduction_to_vector.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/code-performance-advisor/assets/rules/special_rules/R_REDUCE_MORE_CORES/R_REDUCE_MORE_CORES.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/code-performance-advisor/assets/rules/special_rules/R_REDUCE_MORE_CORES/code_snippets/case_0/good_code/good_code.md
  role: supporting
created_at: '2026-09-11T02:18:54Z'
updated_at: '2026-09-14T04:03:29Z'
platforms: []
---

# 标量归约替换与原输入保留

标量循环累加sum、sum_sq或max/min，可考虑用支持目标dtype/长度的Vector归约代替。sum与sum_sq可分别先ReduceSum原输入，再对平方结果ReduceSum；若平方原地覆写，所有仍需要原输入的归一化等消费者必须已经完成，或使用独立平方区。先得到sum不代表整个算子已不需要x。

Basic ReduceSum、WholeReduceSum与高阶按轴ReduceSum有不同重载及模板参数，不能将来源 `ReduceSum<float,true>` 的true普遍解释为isTail。高阶AR/RA的shape、srcInnerPad、scratch查询又是另一合同。来源固定8KiB scratch、64float输出区、8B对齐并非通用要求，应按所用接口查询/约束和实际物理范围分配。

Vector归约后由Scalar GetValue读取结果需有适当V→S完成关系，下一次归约覆写临时结果前先完成其消费者。跨tile/行累计需独立保存每个group的sum和sum_sq，尾部只纳入有效元素，空输入与mean/variance分母单独处理。Float加法重排改变舍入，E[x²]-E[x]²还可能产生消减误差，不能把向量化当作无条件精度等价。

来源混淆32字节DataBlock与256字节Vector repeat：FP32的8元素对应32B，64元素对应256B；这些宽度不直接给出算子加速比。树归约的逻辑深度也不表示任意n在固定资源上只需O(log n)总延迟。count>65536才采用折叠、3–5倍等都是待实测候选，任意长度分块均需完整覆盖。

后续增加double buffer时重算同时存活空间及真实可重叠任务，不能固定把tile减半，或认为每个归约结果都必须双缓冲。来源GroupNorm geomean1.84与TopK反复ReduceMax握手低效只保留为条件不完整案例。验证原输入后续用途、非满repeat、多group、极值和最终统计值，再比较含同步的总时间。

## 跨 tile 先累积再做最终归约

对加法归约，可把每 tile ReduceSum→GetValue→Scalar 累加改为 Vector 累积各 tile 的对应 lane，最后一次归约并读回。这减少重复的向量到标量消费点，但必须保存覆盖整个组的累积区，并在组开始用中性值初始化；尾 tile 的无效 lane 不能引入未初始化值或残留上一 tile 的贡献。

该重排改变浮点求和树，需验证目标误差或确定性合同。最终 GetValue 前仍需 Vector→Scalar 完成；缓冲复用与 Double Buffer 也需合法依赖。多核两阶段方案则要求每核 partial 的独立 workspace、有效核/组索引、发布完成和最终汇总，不能仅增加核数便省略阶段边界。来源关于固定 910B UB 容量和增核收益的泛化不作为容量与性能证明。
