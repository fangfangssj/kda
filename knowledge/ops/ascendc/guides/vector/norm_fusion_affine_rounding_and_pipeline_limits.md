---
type: Guide
title: Norm 融合的 Affine 重排与流水边界
description: 区分统计域、真实遍历、浮点重排和缓冲生命周期。
tags:
- task.implementation
- topic.precision
- topic.implementation.tiling
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-translator/references/ascendc_norm_fusion_patterns.md
  role: primary
created_at: '2026-09-11T04:38:57Z'
updated_at: '2026-09-11T04:38:57Z'
platforms: []
---

# Norm 融合的 Affine 重排与流水边界

GroupNorm通常每N×G组、LayerNorm按指定归一化轴、InstanceNorm按N×C组独立；RMSNorm不减均值，BatchNorm训练可能跨N/空间统计而推理使用已有统计。来源“全部Norm按组两遍、组间无依赖”需先确定该算子的实际统计域和状态更新合同。

在实数代数下scale_c=gamma_c*rstd，bias'_c=beta_c-gamma_c*mean*rstd后可写y=x*scale_c+bias'_c，减少apply阶段逐元素算术。FP浮点重排与FMA改变舍入，x≈mean时可能增加消减误差，不能称数值完全等价。RMSNorm可令均值项不存在，gamma/beta广播按真实归一化轴，不总是per-channel。

TQue输入/输出双缓冲可以表达CopyIn→Compute→CopyOut阶段依赖，实际重叠还需两个真实buffer及预取/消费顺序，depth=2本身不自动隐藏两次GM读取或把带宽成本减半。Pass2依赖本组最终统计，不能在Pass1未结束时随意应用；x驻留时也无需强制为第二次使用分配独立槽。最后消费者完成前不得Free或覆盖。

swish/silu可按x/(1+exp(-s*x))用Muls/Exp/Adds/Div表达，但与x*sigmoid(s*x)及Reciprocal链在舍入/溢出上可能不同，极负x导致exp溢出、±Inf/NaN需实际合同。来源Ln修正约8指令、1ULP及atol<1e-4阈值无通用保证；GELU的Phi不是sigmoid，查表/多项式替换还需独立近似误差，不能由silu等式直接继承。

小shape单块、大shape分片是可行分支，也可用同一正确自适应实现；单模板不是数学上无法覆盖全部。固定S_CHUNK例如2048可限制随空间S增长的UB，尾块只写chunkLen；CPerG、常驻gamma/beta和其他维度仍可能增长，不能仅检查InitBuffer不含S就证明安全。容量应逐项回代，(UB-in-out)/2只是未完整定义buffer情况下的估计。
