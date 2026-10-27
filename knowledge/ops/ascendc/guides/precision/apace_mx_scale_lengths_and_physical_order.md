---
type: Guide
title: APACE MX scale 长度与物理顺序
description: APACE MX scale 长度与物理顺序的来源实例与适用边界。
tags:
- task.design
- topic.implementation.synchronization
- topic.implementation.tiling
- topic.performance.analysis
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/operator-design/operator-anatomy.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/operator-design/development-guide.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/scenarios/put-all-gather-quant-matmul/design.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/scenarios/put-all-gather-quant-matmul/development.md
  role: supporting
created_at: '2026-09-11T07:21:06Z'
updated_at: '2026-09-11T07:21:06Z'
platforms: []
---

# APACE MX scale 长度与物理顺序

来源MX示例每64个K元素对应2字节e8m0 scale，实际scaleKLen=2*ceil(K/64)。表达成K/32仅在适当K对齐且无尾组时等价；ceil(K/64)为偶数不保证K为64倍数。例如K=96满足K%32=0与ceil(K/64)=2为偶数，但K/32=3、scaleKLen=4不同；K=32则ceil=1不满足偶数条件，不能当该合同合法最小正例。

A2A scaleA随本地Ka=globalK/R分片，scaleB随复制B保留globalK。源ScaleA Host有按全K额外分配的现象，分配更大不修复设备按错误行stride读取。每rank A还含全部globalM行，符号m是本地输出行还是总输入行必须确认，避免M与R重复/遗漏相乘。scale数量、字节数、文件长度、窗口段和DMA nonSplitAxisSize需要同一物理公式。

B逻辑[K,N]与DN物理[N,K]应分别描述；源gen_data按[N,K]的C-order写B、不再transpose。scaleB也按[N,scaleKLen]落盘，不能随意按[scaleKLen,N]再转置。数据布局与对应scale组索引必须一起验证。

AG Host只存数据commTile，Init根据mmTile.k重推scaleKLen并生成scaleComm；A2A则Host明确传两个通道。修改K/类型时必须沿真实生成和消费路径确认，不把缺重复字段判错。用每行、每组、每rank不同scale编码，覆盖K=96这类公式分歧探针，并按真实支持合同决定接受修正布局还是明确拒绝。
