---
type: Guide
title: 归约 fallback 轴路由与序列化字段遗漏
description: 保留AR/ARA建模，指出非连续归约轴被错误合并。
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
created_at: '2026-09-11T06:32:03Z'
updated_at: '2026-09-11T06:32:03Z'
platforms: []
---

# 归约 fallback 轴路由与序列化字段遗漏

可按保留轴A与归约轴R合并相邻且布局可合并的同类轴，得到AR、RA、ARA等结构。对负axis先规范到rank内，明确重复axis、越界、空axis列表、rank0、空张量、keepdim与dtype规则。移除size1轴应保留语义映射，来源声称移除但代码只合并相邻类型。

源码route_mode遇首个R后把后续所有轴相乘成A0，后面再出现R时归约语义丢失。CPU例shape[2,3,4],axes[0,2]应归约8个值并保留3，脚本却给R2、A0=12。此类RARA结构需要保持多轴地址映射、合法转置或专门Kernel，不能强行压成一次连续ARA。

来源提供AR小R、AR全载、AR重算、ARA全载、ARA重算五类候选，算法能力与UB容量需分别判定。多个候选同时合法时可测量比较。var/std、LayerNorm、RMSNorm、Softmax、argmax不只是修改相同分母：其统计量、索引、重读输入与跨组归并合同不同。

enableWelford在脚本仅匹配var/std/norm字符串，未覆盖layernorm/rmsnorm且RMS也不需要方差；不能把“所有Norm用Welford”当数学要求。dichotomy仅force设置不代表其他分支不存在树归约，withIndex标志未计索引内存。group_r计算后未完整更新任务和launch字段。to_dict遗漏enableWelford/dichotomy/withIndex/tilesPerCore，类本身又缺文档一些tail/fold字段，输出JSON不是完整可用Kernel ABI。

按坐标编码输入，对非连续多R、退化A、不同axis顺序逐输出比较覆盖与结果；同时验证tiling序列化后所有Kernel消费字段完整、类型范围及默认值一致。
