---
type: Guide
title: Clipped SwiGLU 布局与非对称 Clip 语义
description: 保存奇偶Gather，纠正仅上限裁剪仍防所有Exp溢出。
tags:
- task.optimization
- topic.precision
- topic.implementation.tiling
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/cake-evo/references/meta_prompts/strategies/perf_13_special_algorithms.md
  role: primary
created_at: '2026-09-11T06:02:18Z'
updated_at: '2026-09-11T06:02:18Z'
platforms: []
---

# Clipped SwiGLU 布局与非对称 Clip 语义

前后排列输入为[A全行,B全行]，奇偶排列为[a0,b0,a1,b1,...]，两者需不同地址映射。来源先转换至FP32后ArithProgression产生byte offset0,8,16...，Gather base0取A、base4取B；这依赖float4B与SWI_FACTOR2及该Gather按byte解释。若直接对16位源Gather，步长/基址需改变；原始rowStride、tail pair及跨batch地址也独立核验。

索引缓冲长度至少ubMaxPair且每轮有效pairNum不超容量，int32等差数列和byte地址跨度不能溢出。两个Gather都消费完原xFloatLocal后，才可把该buffer改作Exp临时；如果tmpA/tmpB与源区重叠还需真实API允许。

数学上sigmoid(z)=1/(1+exp(-z))=0.5*(1+tanh(z/2))，有限精度两种实现误差/饱和及吞吐不同，不必然Exp更稳。alpha不为1时tanh输入也应alpha*x/2，不能对不一致公式比较。

source tmpB先clip[-L,L]再加gluBias，tmpA却只Mins到L，然后exp(-gluAlpha*tmpA)，是非对称裁剪示例：A负侧仍可能无界，正alpha下exp可能上溢。不能概括为两路都完整clip且始终防Inf，也不能擅自补Maxs改变模型公式。须确认真实ClippedSwiGLU合同是否正是min(A,L)/(1+exp(-alpha*min(A,L)))乘(clip(B,-L,L)+bias)；源片段未展示最终乘积，不是完整输出实现。

裁剪后的B再加bias可超过原±L，且前置clip输入与后置clip乘积不等价；GLU一般定义也不强制特定clip。负Inf/NaN、极大alpha、limit合法性、两种输入布局的逐pair一致性必须单独验证，不以“与现代芯片更适配”作为数值证据。
