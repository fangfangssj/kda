---
type: Guide
title: SwiGLU 双分支语义与高阶临时空间
description: 区别单独 SiLU 和双输入 SwiGLU，明确 beta、布局、别名及双缓冲范围。
tags:
- task.implementation
- topic.activation
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P50_swiglu_fusion.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P50_swiglu_fusion.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P50.yaml
  role: supporting
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/adv_api/activation_functions/swi_glu_interface/swi_glu.md
  role: constraint
created_at: '2026-09-11T01:56:34Z'
updated_at: '2026-09-11T01:56:34Z'
platforms: []
---

# SwiGLU 双分支语义与高阶临时空间

Swish/SiLU分支计算x*sigmoid(beta*x)，SwiGLU还需与另一门控分支逐元素相乘。来源手写序列Muls(-1)→Exp→Adds(1)→Div→Mul只展示beta=1时的SiLU，尚未包含另一分支乘法；第一条Muls也不是sigmoid(-x)。替换时必须确认完整计算、两分支顺序与实际beta，不能用SwiGLU替代任意一个SiLU。

若矩阵输出按两半存储，每行的两半起点、每半有效元素数及物理行跨度需分别计算。halfTokenLen只有等于实际要算的每分支范围才可作count；尾部、奇数分支总长和padding需要按算子合同处理。SwiGLU不限于MoE FFN，纯Vector或其他具有相同数学语义的上下文也可评估支持的接口。

已有SwiGLU卡的显式scratch重载要求两个LocalTensor源、LocalTensor目的、float beta、LocalTensor<uint8>临时区和calCount，记录half/float、ND以及源/目的与scratch不重叠等约束。源五参数片段若使用另一重载需单独核对，不能把workspace变量名自动解释为GM或已提供临时空间。isReuseSource在所记录接口中是预留参数，不能设为true就获取别名支持。

高阶调用可能减少手写算子与显式同步，但一个API不等于一条向量指令；源“5→1、减少80%指令、75%屏障”只比较表面写法，不能当汇编或实测结论。onesLocal可被省去，但高阶scratch可能重新占用相当或更多容量，需要按对应Host空间查询计算。

Cube输出的双槽和Vector计算临时区是不同生命周期。下一tile搬入另一槽之前，仍需真实ready/release及AIC/AIV交接，单个PIPE_V不能完成全部同步；双缓冲增加UB/workspace是否值得需测完整路径。验证不同beta、分支顺序、负大值/特殊值、非对齐slice、tail与源目的别名，数值等价后再比较时延。
