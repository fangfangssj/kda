---
type: Guide
title: 分块统计归约中的 Welford 状态与跨组合并
description: 用 count、mean、M2 表达分块统计状态，说明流式更新、不同大小分组合并及 workspace 交接需要满足的条件。
tags:
- task.design
- topic.compute.reduction
- topic.precision
- topic.parallelism
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-tiling-design/references/reduction/alg-welford.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-tiling-design/references/reduction/alg-group-reduce.md
  role: supporting
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/adv_api/normalization/welford_update.md
  role: constraint
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/adv_api/normalization/welford_finalize.md
  role: constraint
created_at: '2026-09-10T12:29:14Z'
updated_at: '2026-09-10T12:29:14Z'
platforms:
- '950'
- a2
- a3
---

# 分块统计归约中的 Welford 状态与跨组合并

## 流式统计状态

当归约维无法一次装入片上缓冲时，可维护 `(n, mean, M2)`，其中 M2 是相对当前均值的平方偏差和。对新样本 x，使用更新前后的均值计算：

```text
n_new = n + 1
delta = x - mean
mean_new = mean + delta / n_new
M2_new = M2 + delta * (x - mean_new)
```

初始空状态为 n=0、mean=0、M2=0。最终方差是 `M2 / (n - correction)`；当分母不合法时应遵守算子的空输入或自由度规则，不能直接执行除法。

Welford 可以减少为计算统计量而重新读取输入的次数；若算子还要用最终统计量归一化全部输入，仍可能需要后续读取，不能把“统计单遍”表述为整个算子只访问输入一次。

## 不同大小的分组合并

对两个非空状态 a、b，令 `n = na + nb`、`delta = mean_b - mean_a`：

```text
mean = mean_a + delta * nb / n
M2 = M2_a + M2_b + delta * delta * na * nb / n
```

合并需要 count，不能直接平均两个均值或方差。空分组应作为无贡献状态处理；尾块按有效元素数计数。中间乘法与计数转换的类型也要明确，避免在进入浮点计算之前发生整数溢出。

## 跨核实现边界

当保留轴的独立工作量不足以利用多个核时，可沿归约轴分组。各核写出自己的 partial，合并阶段读取已完成的结果。workspace 的布局需区分输出组、归约分组及 mean/M2/count，按所用搬运接口进行对齐。

生产 partial 与读取 partial 之间需要目标平台支持的完成与可见性保证。若采用同一 Kernel 的全核同步，必须同时满足该同步方式的调度和参与者约束；单独写一条 SyncAll 不能证明整个协议正确。也可由明确有序的独立执行阶段完成交接。

## 选择与验证

全载后进行两遍片上统计通常实现更直接；分载时 Welford 值得评估，但吞吐、累计精度和合并树应实测。不存在对所有算子最优的“每 8 块合并一次”常量。

验证至少包括不同分组大小、非整块尾部、常量输入、大均值小方差，以及单组/多组合并一致性。具体高阶接口的布局、dtype 与空间要求见 [WelfordUpdate](../../apis/simd_api/adv_api/normalization/welford_update.md)和 [WelfordFinalize](../../apis/simd_api/adv_api/normalization/welford_finalize.md)。本文的数学状态不能直接替代接口内部布局。
