---
type: Guide
title: 归一化类型矩阵与分发边界
description: 保存统计输出和分块Key，修正九组合、阈值及舍入时机。
tags:
- task.implementation
- topic.precision
- topic.implementation.tiling
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/cake-evo/references/meta_prompts/strategies/dtype_01_mixed_precision.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/cake-evo/references/meta_prompts/strategies/dtype_02_template_kernel.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/cake-evo/references/meta_prompts/strategies/dtype_03_tilingkey_dispatch.md
  role: supporting
created_at: '2026-09-11T05:51:16Z'
updated_at: '2026-09-11T05:51:16Z'
platforms: []
---

# 归一化类型矩阵与分发边界

LayerNorm/NormCommon将Tfm与Tweight分离，常见源约束为两者相同，或低精度输入配FP32权重；gamma/beta同型。dtypeKey=10*inputId+paramId，FP32/FP16/BF16 ID0/1/2。source列100、110、111、120、122仅5种SingleRead组合，不证明完整3×3矩阵；其15个key还可能包含算法维度，不能当15个dtype组合。GetDTypeKey子映射-1必须先拒绝，否则无效类型也可能进入整数编码。

源SingleRead按sizeof(Tfm)==2扩宽、BF16 CAST_ROUND降精等是某实现路径，别处RINT要求不应被覆盖。LayerNormV3的mean/rstd或variance输出M可能独立低精度并CastBatchMeanLastout，NormCommon又固定FP32，两者是不同ABI；均值、方差、rstd不可混用字段名，strided DMA必须匹配每repeat半值布局。

AddRMSNormCast key=dtype*10+mode：half1/BF163，normal0、splitD1、singleN3、multiN4；numCol>ubFactor先split，否则blockFactor1可single。DynamicQuant AddRMS则key1Normal/2SingleRow/3SliceD，dtype可能由DTYPE_X1另行确定；不能强制每个dtype再占一个key。

Gemma源dataPerBlock half/BF16为16、float为8，表示32B对应元素数；ubFactor B1610240/B328192是该buffer组合预算，非芯片UB总容量或任何算子通用tile。key0整行、1SplitD，SplitD需全D统计再归一化而非每chunk独立RMS。释放输入queue必须在Cast真正消费完成后，混合T/T_GAMMA仍有独立权重转换。

DeepNorm源模式Short、LELimit、GTLimit、Common按dtype与D/UB组合选择，给4096和8192/15360阈值；Short正文≤500而表头≤100冲突，需真实Host条件确定。示例FP32 key18/2/6/14对应其短/小/中/大分支，不能照数字推定未知边界。BatchNorm源FullReduce2000/2001、Welford1000..1013及Regbase300000+，R0*R1<8192选择FullReduce是历史启发式，未给全capability条件，不是普适性能定律。

NormCommon Welford key400+dtypeKey，tiling包含M/N/welfordUpdateTimes/welfordUpdateTail；writer、注册类与reader对应，Update计数和尾部必须真实覆盖N。Welford仍有FP32舍入，不是“避免所有累加误差”，中心化two-pass也不能泛称不稳定。

SoftmaxGrad以paddedHeadDim≤1024用1000系列，较大用2000系列，低位BF160/FP161/FP322；小路径高阶SoftmaxGrad，大路径WholeReduceSum仍须完整dy*y归约与最终链式公式。阈值针对padded维，不可误用原始维；多Cast路径实际两次扩宽加一次降精，不是仅两次转换。

ScaledMaskedSoftmax key0float/1half/2BF16。BF16先升FP32做scale/mask，half先在half做scale与mask再升FP32，存在不同舍入/overflow点。SelectWithBytesMask的离散屏蔽与baseline直接Add浮点mask不是任意输入下等价；mask数据类型、真值含义、sentinel、scale和全mask处理均由合同定义。不能因为scale是线性就断言低精度总足够。
