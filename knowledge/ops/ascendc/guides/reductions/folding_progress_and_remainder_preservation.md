---
type: Guide
title: 折叠归约的进度与余项保留
description: 保存低延迟折叠的适用条件，并指出不缩短长度、尾项丢失和输出步长问题。
tags:
- task.optimization
- topic.compute.reduction
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P68_low_latency_reduction.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P68_low_latency_reduction.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P68.yaml
  role: supporting
created_at: '2026-09-11T02:07:16Z'
updated_at: '2026-09-11T02:07:16Z'
platforms: []
---

# 折叠归约的进度与余项保留

Sum归约可先用向量逐元素Add折叠较长区间，再对最后一个适合的向量用WholeReduceSum，或采用BlockReduce后WholeReduce。目标是减少串行归约路径，适用性依赖接口支持、长度和指令成本。每轮必须保留所有有效输入的贡献，并让有效长度严格下降，最终一次归约需使用实际剩余count。

来源playbook写 `halfNum=DivCeil(totalNum,16)*16`，这实际上是向上对齐，并非折半：totalNum=128得到halfNum=128，Add长度0且循环不前进；totalNum=65得到80，若无符号相减可能形成极大count。另一card使用不同乘数，两段不能视为等价模板。来源先把hLength向下截成16倍数，虽计算remaining却未消费，也会丢失尾项。

一个数学上的折叠描述是令0<h<n，将n-h个合法配对 `x[i]+x[i+h]` 写入前段，再把未配对的[i=n-h,h)范围保留到下轮（当该范围非空）。h应符合实际接口对齐、重叠和容量合同；分离dst时未配对值需要显式复制，不能只将src指向dst。沿用旧buffer也要保证没有覆盖尚未读取值。可采用独立尾部处理或中性值padding，但不能静默截掉尾数。

float每256B为64元素、half为128；来源16元素分组不是所有归约的统一硬件粒度。Counter区域按[mask作用域](../vector/counter_mask_scope_and_tail_counts.md)恢复模式和值，并核对归约重载支持。若输入包含bs行，紧凑bs×1输出的步长应为1；源 `dst[i*hLength]` 只有为这种物理间距分配输出时才合法。

跨核各自向同一GM位置普通写入会覆盖，不形成归约。Mean还需正确有效N，max/min应使用对应组合操作与特殊值规则，不能照搬Add折叠。Sum重排会改变浮点舍入顺序，要求按目标误差标准验证。来源30000个FP32的242→172cycles、8.44对13微秒等是未在本批复现的条件化案例，不能归纳固定2–5倍收益。

验证n=0/1、63/64/65、127/128/129、非16倍数、各轮奇数余项、分离/允许原地输出、行距和多核最终合并。可用每个输入独立贡献的计数证明无遗漏，再比较误差与完整时延。
