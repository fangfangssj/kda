---
type: Guide
title: 向量索引生成的精确性与位宽保留
description: 保留等差数列、商余数和负索引转换，同时限定浮点与重新解释。
tags:
- task.optimization
- topic.implementation.tiling
- topic.memory
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/cake-evo/references/meta_prompts/strategies/perf_04_load_balance.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/cake-evo/references/meta_prompts/strategies/perf_10_vectorized_copy.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/cake-evo/references/meta_prompts/strategies/perf_07_data_alignment.md
  role: supporting
created_at: '2026-09-11T06:15:28Z'
updated_at: '2026-09-11T06:15:28Z'
platforms: []
---

# 向量索引生成的精确性与位宽保留

LinearIndex合轴例ArithProgression(float(offset),1)再Cast到int32，并用floor(floatIndex*(1/floatStride))求商、整数减q*stride求余。FP32只连续精确表示到2^24，offset较大时即使stride很小也会碰撞；倒数乘法还可能把整商边界舍入到错误一侧。需要覆盖完整索引范围并证明/校正q与r，不能仅因stride通常小就视作安全。序列缓冲和同步成本、具体指令吞吐需实测，不保证单cycle生成任意长度。

负索引向量化可以对INT32算术右移31得到0/-1，再减sign*D达到负时加D；源注释写“左移”与实际ShiftRight相反。仅在算子允许[-D,D)、D/索引与乘积不溢出时成立，规范化不替代越界检查。INT64输入先Cast INT32须有范围证明，barrier只能保护时序，不能恢复截断值。

Gather的byte offset=合法元素索引*sizeof(T_X)需在支持宽度内完成，转换后Gather uint32偏移须检查范围；把INT64 idxLocal ReinterpretCast<uint32_t>会交错低/高32位，不等于数值窄化或只取每个低位。合并x和index到一队列时，idx起点必须按实际字节对齐、完整容量和两者生命周期核算。

MicroAPI Arange+Div+余数分解可生成多维偏移，各num轴必须正，rate需真实元素/字节stride，整数Div与浮点商语义不能混用，最终地址积和必须宽域。模板泛型不代表64位索引重解释为32位总合法。

GM→UB后MTE2_V、相依Vector之间的正确屏障及真正Scalar读之前的完成须分别管理。将有效数量上对齐可以生成额外lane，但无效索引必须屏蔽，不能让dummy地址进入Gather/Scatter。验证相邻大索引、除数边界、负数、末轴和64位高位非零。
