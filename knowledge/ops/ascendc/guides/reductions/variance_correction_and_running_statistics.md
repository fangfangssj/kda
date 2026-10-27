---
type: Guide
title: Welford 方差修正与 Running 统计量的语义
description: 区分在线统计算法、总体/样本方差、EMA 系数及归一化消费，纠正把单样本兜底当作无偏方差的误用。
tags:
- task.design
- topic.compute.reduction
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/A2_welford_algorithm.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/A2_welford_algorithm.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/A2.yaml
  role: supporting
created_at: '2026-09-11T01:26:08Z'
updated_at: '2026-09-11T01:26:08Z'
platforms: []
---

# Welford 方差修正与 Running 统计量的语义

Welford 维护有效样本数 n、mean 和平方偏差和 M2；其逐样本及分组合并公式见[分块统计状态](streaming_statistics_and_group_merge.md)。新样本的 delta 使用旧均值，第二项 x-mean_new 使用更新后的均值。循环实际处理 count 个样本时，计数必须反映这 count 个样本及已有状态；不能使用一个不匹配的外部 N 作为最终分母。

M2/n 是总体方差，M2/(n-1) 是 n>1 时的样本方差。对同一 M2，从前者转换到后者可乘 n/(n-1)；这个系数修正并不等于已经采用 Welford。n=1 时把系数设成 1 只是在某种算子合同下采用有限值兜底，不能使单样本的无偏方差变得有定义。n=0、非法自由度和空分组也需分别处理。

BatchNorm 等算子的前向归一化、保存统计量和 running statistics 可能采用不同方差估计。EMA 可写作 `running_new = a*running_old + (1-a)*batch_stat`，但框架的 momentum 可能命名 a 或 1-a；batch_stat 采用总体方差还是修正后的方差也由具体算子决定。不能把来源“running 方差永远不做 N-1 校正”推广为通用规则。

需要区分 `E[x²]-E[x]²` 与先求 mean、再求 `sum((x-mean)²)` 的两遍中心化算法。前者在大均值小方差时容易发生消减误差；后者不应被仅凭“两遍”归类为同一个不稳定公式。Welford 有数据依赖和除法/合并成本，并非所有向量化场景都比两遍片上算法更快。

中间统计通常值得使用 FP32 或满足误差合同的更宽精度。微小负方差是否截零，应结合误差边界与下游开方合同；显著负值仍需调查数值、计数或同步错误，不能用 clamp 掩盖。RMSNorm 主要消费均方值，不因名称中有 Norm 就必须改算中心方差。

同一源材料夹带 AMSGrad 风格代码：先对历史最大二阶状态与当前二阶状态取 Max，再开方并乘修正系数。这一段表达“先选择状态，再执行非线性和缩放”的依赖，不能作为 Welford 实现或 Adam 收敛性的证据；完整优化器还需要其状态定义与更新顺序。

验证应覆盖 n=0/1/2、不同 correction、常量值、大偏置小扰动、不等长分组、EMA 一步手算及训练/推理各输出。原始样例没有证明其全部框架语义，本文不声称已完成对应算子编译或训练验证。
