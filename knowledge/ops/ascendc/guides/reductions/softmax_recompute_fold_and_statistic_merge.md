---
type: Guide
title: Softmax 重算、折叠与统计量归并
description: 保存Softmax经验中的可复用状态，区分max、sum、方差和RMS。
tags:
- task.optimization
- topic.implementation.tiling
- topic.memory
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-perf-optimize/references/tiling/reduction/fallback/index.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-perf-optimize/references/tiling/reduction/fallback/tiling-fields.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-perf-optimize/references/tiling/reduction/fallback/tiling-flow.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-perf-optimize/references/tiling/reduction/fallback/script/reduction_tiling.py
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-perf-optimize/references/tiling/reduction/fallback/example/softmax/experience.md
  role: supporting
created_at: '2026-09-11T06:32:03Z'
updated_at: '2026-09-11T06:32:03Z'
platforms: []
---

# Softmax 重算、折叠与统计量归并

Softmax稳定形式为m=max(x)、s=sum(exp(x-m))、y=exp(x-m)/s。归约长度不能全驻留时可多遍重读，或保存在线(m,s)并在更大max出现时重标定旧sum。合并两分区时m=max(ma,mb)，s=sa*exp(ma-m)+sb*exp(mb-m)；最后每个原始x要按全局m重新计算exp再除s，不能把原始x直接除以s。

来源小R经验把max描述为累加，以及泛称二分Add合并统计值，均不适用于max与Softmax状态。sum可Add，max可Max，Welford需count/mean/M2专门合并，RMS使用平方和/计数；LayerNorm方差与RMS均方不同。argmax还须携带索引和tie规则。全mask、全负无穷、NaN、Inf及低精度累加的约定应沿用算子合同。

将R切为chunk时，完整循环q=floor(R/chunk)、余数r=R-q*chunk；r0表示没有额外尾循环。若用总循环ceil(R/chunk)及最后tile长度，则整除时最后长度为chunk。两种约定不能混写。来源fold base与floor/ceil循环混用，可能遗漏部分尾tile或产生负foldRemain；先列出实际有效tile，再用整数二叉合并层数和有效标志组织cache，不能对不存在的槽做Add/Max。

树归约cache深度与最多同时存活部分量相关，使用整数层级并处理0/1块；每层保存的统计量必须同一种语义且记录有效性。Softmax重算节省UB但增加GM读取和Exp次数，最后pass的真实流量、同步与精度应纳入测量，不能只依据更大tile断言更快。
