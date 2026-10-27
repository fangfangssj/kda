---
type: Guide
title: Rsqrt Newton 修正的状态与数值域
description: 保存倒数平方根迭代公式及误差关系，明确原始输入、近似值、epsilon 与特殊值必须独立处理。
tags:
- task.design
- topic.precision
- topic.compute.normalization
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/A6_high_precision_rsqrt.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/A6_high_precision_rsqrt.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/A6.yaml
  role: supporting
created_at: '2026-09-11T01:26:08Z'
updated_at: '2026-09-11T01:26:08Z'
platforms: []
---

# Rsqrt Newton 修正的状态与数值域

对有限 x>0，设 y 近似 1/sqrt(x)，Newton 修正为 `y_new = y*(1.5-0.5*x*y*y)`，也可写为 `y_new = y+0.5*y*(1-x*y*y)`。归一化中的 x 可能是 var+eps 或 mean_square+eps，应先建立这个实际输入，再生成 y0。

实现必须同时保留 x 和当前 y。若把 var+eps 写在 output，又把 output 当作 y0，则乘法链已经不是上述公式；第二次迭代仍消费原始 x 和新的 y1。可按 `t=y*y; t=x*t; correction=1.5-0.5*t; y=y*correction` 理解依赖，再按实际 API 重叠规则安排寄存器或 UB。Mula 的累加目的地也必须预先初始化，不能把注释中的 1.5 当作已写入状态。

由公式可推导：若 y=(1+e)/sqrt(x)，在精确算术下 `e_new=-1.5*e²-0.5*e³`。这解释了小误差区域的二次修正，并不能保证任意初值收敛，也不能保证浮点下单次达到完整 FP32 精度。原材料约 8-bit 初值、一次 16-bit、两次 24-bit 仅是粗略估计，API 配置、输入范围、乘法舍入、FMA 和下溢都可能改变实际误差。

Rsqrt 初值与 Sqrt 后求倒数是可比较的候选，但后者也需区分向量 Reciprocal 与标量除法。已有 [a3 Rsqrt 精度案例](../../runbooks/precision/rsqrt_fast_approximation_use_sqrt_reciprocal.md)只支持其特定重载和输入探针，不能推广到所有 Rsqrt。Newton 增加依赖、临时存储和指令，不保证比高精度 API 或 Sqrt+倒数更快。

特殊值以 x 而非未经加 eps 的 var 判定：var=0 且 eps>0 时结果应遵守 1/sqrt(eps)；正无穷、零、负数、NaN、带符号零及非正规数按算子规定处理。正无穷和零直接进入迭代会出现 inf*0 等无效运算，不能只在结果端无条件改成 0 掩盖。`x>0 ? ... : 0` 会改变 NaN 传播。源文把 eps<1e-7 一概视为下溢也不成立，阈值依赖 dtype、舍入与实际运算。

验证包括 x=1、非整平方、数量级扫描、var=0+eps、特殊值，以及初值/每轮/最终输出误差和总耗时。本文仅保存公式与实现约束，没有声称修复了完整样例或验证了设备速度。
