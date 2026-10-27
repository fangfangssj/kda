---
type: Guide
title: Attention 在线 Softmax 的空行与量化阶段合同
description: 修正有限 mask、初始 rescale 及 QK/PV 归约轴混淆。
tags:
- task.optimization
- topic.implementation.tiling
- topic.performance.analysis
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-tiling-design/references/flashattention/subfamilies/base_design.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-tiling-design/references/flashattention/subfamilies/quantization_design.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-tiling-design/references/flashattention/design/execution.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-tiling-design/references/flashattention/implementation_ref.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-tiling-design/references/flashattention/specialization/feature_flags.md
  role: supporting
created_at: '2026-09-11T04:27:49Z'
updated_at: '2026-09-11T04:27:49Z'
platforms: []
---

# Attention 在线 Softmax 的空行与量化阶段合同

每KV块在线更新为mNew=max(mOld,maxS)，alpha=exp(mOld-mNew)，P=exp(S-mNew)，lNew=alpha*lOld+sum(P)，ONew=alpha*OOld+P*V，末尾O/l。首个有限有效块mOld=-inf时alpha数学值为0，不是1；若跳过旧状态读取并直接写首结果，可以避免初始化旧GM，但不能通过0*未初始化NaN达到同等效果。

空或全mask行需显式定义中性状态和最终输出。将所有S填-1e30后减本行max会得到全0，exp全部为1，产生均匀权重，不是零权重；禁止-inf也不能修复此问题。可以使用有效性mask和对空行的分支，采用-inf时避免-inf-(-inf)。在至少一个有限有效score的相对max表示下l≥1；任意加epsilon会改变算子数学，不能当默认修复。Inf/NaN输入和完全不可见行需遵循接口合同。

GQA中mEff包含不同head，同一query在多个head重复；因果边界最大query为qOff+curBq-1，不能用qOff+mEff-1。SoftMaxShapeInfo逻辑行数、mPad物理行数及isBasicBlock需对应真实布局，padding行不得写成有效输出。

S和PV累加通常保留FP32，P可能转换为FP16/BF16/MX格式；V输入保持BF16不表示整个Vector阶段无需Cast。来源把量化BMM1结果写成1字节与其FP32累加路径矛盾，不能据此缩减S缓冲。Cast、Fixpipe转换及位置支持应查实际平台/重载，不能把某路径仅能UB转换扩大为所有转换指令限制。

MX中Q/K的归约轴是Dqk，P/V的归约轴是Sk，因此V scale的分组轴沿Sk，不能沿Dv替代。来源32元素分组、E8M0与特定存储是其路径合同；scale的head stride、padding及打包格式需独立于数据位宽核算。P及对应scale应在每个生成块交给C2前共同发布，并保持共同生命周期，不能只在最后KV迭代更新P scale。

来源c1v1Loop将BMM1 K写为S2是轴混淆：QK归约K=Dqk，S2是score的N列；PV归约K才是S2。ceil(actS2/256)可能描述N分块，不能当QK reduction循环。K_BASE、yStep和末块长度需有明确单位。输入量化不会让FP32 S/PV缓冲减半，也不自动使单缓冲调度安全；INT8反量化、MX与其他accumulator路径不能互换合同。
