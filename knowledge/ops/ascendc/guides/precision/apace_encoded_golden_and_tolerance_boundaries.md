---
type: Guide
title: APACE 编码 golden 与容差边界
description: APACE 编码 golden 与容差边界的来源实例与适用边界。
tags:
- task.design
- topic.implementation.synchronization
- topic.implementation.tiling
- topic.performance.analysis
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/operator-design/design-template.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/operator-design/plan-template.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/operator-design/development-guide.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/operator-design/operator-anatomy.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/scenarios/put-all-gather-quant-matmul/design.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/scenarios/put-all-gather-quant-matmul/development.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/scenarios/put-all-to-all-quant-matmul/design.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/scenarios/put-all-to-all-quant-matmul/development.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/scenarios/compute-first-reduce-scatter/design.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/scenarios/compute-first-reduce-scatter/development.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/troubleshooting/failure-navigation.md
  role: supporting
created_at: '2026-09-11T07:21:06Z'
updated_at: '2026-09-11T07:21:06Z'
platforms: []
---

# APACE 编码 golden 与容差边界

golden需从实际写入设备的FP8/MX字节和物理layout解码，再执行明确的每rank通信与Matmul语义。只对量化前随机float做Matmul会把量化误差错归因kernel。A/B独立E4M3/E5M2时有4组合，Host实际dispatch必须与两份字节解释一致；未知dtype字符串不能默落最后一个分支。固定seed利于复现，跨rank独立seed可揭示rank错配。

用FP32解码/累加后转输出是一个参考路径，但若设备每个rank先输出BF16 staging、再FP32聚合，golden必须复现中间BF16舍入点。concat_K(A_s)乘stack_K(B_s)与sum_s A_s B_s数学等价，浮点求和次序不同会改变误差；来源将前者称“非逐rank求和”不能解释成数学上不等价。复制bias若每个K片或每rank都加，再求和会乘次数，须按最终语义只加入一次或有等价分摊。

来源A2A用rtol=atol=1e-2；AG报告bit-exact/1 ULP。它们是样例判据，不是所有相同dtype算子的统一容差。raw uint16差要先拓宽再相减，BF16负值、正负零、跨符号、Inf/NaN须显式处理；无序位模式差不自动等价数学ULP距离。非有限值不能因比较返回false或np.nanmax而漏判。

非全零输出只说明有写入迹象，不证明正确，合法零输入也可能应全零。T1通过不能代表T>1、tail、slot绕回或多rank。测试矩阵应覆盖真实合法域及相邻拒绝边界，并检查支持域各约束的交集；条件列表各自可满足不意味着组合可行。误差全错/固定比例、随K增长、按rank/tile分布是诊断线索，不是唯一根因证明。
