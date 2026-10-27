---
type: Guide
title: Rsqrt 初值溢出与特殊值收尾
description: 保存Newton状态要求，纠正倒数后开方与修正掩码的陷阱。
tags:
- task.implementation
- topic.precision
- topic.implementation.tiling
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/cake-evo/references/meta_prompts/strategies/acc_06_high_precision_rsqrt.md
  role: primary
created_at: '2026-09-11T05:41:33Z'
updated_at: '2026-09-11T05:41:33Z'
platforms: []
---

# Rsqrt 初值溢出与特殊值收尾

对正有限x，Newton候选y_next=y*(1.5-0.5*x*y*y)。初值精度、FMA/独立乘加顺序及迭代次数决定实际误差；来源“8bit初值、1%到0.001%、固定6条指令”未经目标实测不能作普遍保证。若Mula语义为dst+=src0*src1，则参与1.5-0.5*x*y²的目标必须预置1.5，源片段t1未初始化不构成可用实现；末尾rstd的Mula也需完整已定义初值和更新公式。

先Div(1,x)再Sqrt的初值仍包含除法，并不等于避免Div。其有限精度域与1/Sqrt(x)不同：FP32 x约1e-40时1/x可能溢出Inf，而1/sqrt(x)仍约1e20有限；大x的倒数下溢也会抹掉有效结果。不能从实数恒等式推断极端值行为等价。

若提前Select把0/Inf修成特殊结果，再无条件执行Newton，0*Inf等仍可能产生NaN并覆盖修正。特殊值分支应保护整个后续计算，或在最终结果应用有效掩码；比较对象是实际var+epsilon而非只比较原var。NaN、正负0、负值及epsilon导致的非正域需显式合同。

Maxs(var,var,0)不消除正subnormal，也可能掩盖上游方差负值错误；只检查==0与==Inf不能定义NaN策略。统计舍入产生轻微负方差时是否夹零须与参考一致。用正态、极小/极大、近0方差、特殊值和迭代误差曲线验证，不以“高精度rsqrt”名义省略输入域。
