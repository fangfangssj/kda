---
type: Guide
title: 统计修正与 Softmax 梯度合同
description: 保留BatchNorm及AMSGrad状态，纠正梯度基线和API内部推断。
tags:
- task.implementation
- topic.precision
- topic.implementation.tiling
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/cake-evo/references/meta_prompts/strategies/acc_02_welford_algorithm.md
  role: primary
created_at: '2026-09-11T05:41:33Z'
updated_at: '2026-09-11T05:41:33Z'
platforms: []
---

# 统计修正与 Softmax 梯度合同

Welford增量必须同时保留旧mean和新mean：delta=x-mean_old，mean_new=mean_old+delta/n，M2_new=M2_old+delta*(x-mean_new)。总体方差M2/n与样本方差M2/(n-1)适用不同合同；均值中心化的two-pass不等于E[x²]-E[x]²，不能泛称所有two-pass都比Welford不稳定。高阶LayerNorm的某个true模板值也不能单独证明内部采用Welford或特定求和算法。

BatchNorm样本数N=patternR0*patternR1应在足够宽的整数域乘法，再转换double；乘法溢出后再cast无法修复。来源N==1取系数1只是不除零的特定策略，不产生有定义的无偏方差；N0另须处理。saveVar是总体量时，batchVarScale=N/(N-1)转换为样本量。示例momentumVar=saveVar*batchVarScale*momentum明确将校正统计贡献给EMA，与其文字“running从不涉及N-1校正”冲突；前向、保存量和running各自采用哪一种由真实算子合同决定，不能凭框架名称互相替代momentum方向。

AMSGrad维护vhat=max(vhat_prev,v)，随后开方并应用相应bias correction；max_exp_avg_sq是跨step最大二阶矩状态，不是当前梯度范数。额外Max与状态读写有成本，但单凭vhat非降不能证明含bias correction或学习率schedule的有效步长永不增大，更不能保证所有训练收敛。该知识不属于Welford算法本身。

给定上游dy和保存的softmax概率y，dx_i=y_i*(dy_i-sum_j(dy_j*y_j))。源baseline只算y*(1-y)，仅为Jacobian对角项，缺少上游梯度与交叉项，属于数学语义缺失。含scale/mask的前向还需按链式法则乘scale并遵守可见域。保存y后该公式无需重新exp，不能从SoftmaxGrad名称推断内部一定有log-sum-exp、Kahan或重新归一化。

高阶API所需tiling、临时空间、physical paddedHeadDim与真实有效列须匹配。测试应包含非常量dy、不同维度、N0/1、低方差大均值与运行统计更新，不能用“专家实现”标题代替精度证据，1–2数量级收益也须数据与误差口径支持。
