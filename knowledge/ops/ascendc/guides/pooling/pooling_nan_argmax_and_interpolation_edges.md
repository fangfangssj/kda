---
type: Guide
title: Pooling NaN、Argmax 与插值边缘
description: 补齐Ceil修正、Padding获胜和仅右角有效的分支。
tags:
- task.implementation
- topic.precision
- topic.implementation.tiling
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/cake-evo/references/meta_prompts/strategies/acc_05_softmax_stability.md
  role: primary
created_at: '2026-09-11T05:47:04Z'
updated_at: '2026-09-11T05:47:04Z'
platforms: []
---

# Pooling NaN、Argmax 与插值边缘

AdaptiveMaxPool3DGrad来源要求x/grad为5D NCDHW，索引域D*H*W在宽整数域检查INT32界。D方向Gcd(inputD,outputD)可帮助构造窗口周期/分组，但仅计算GCD不证明梯度窗口无重叠或可省同步。

FP32/FP16/BF16负Inf位模式分别0xFF800000/0xFC00/0xFF80，位填充应匹配真正类型；源函数只给float/half分支，BF16常量虽存在却无实现。用有限-3.402823e38代替-Inf会在所有输入为-Inf等情况下改变选择；初始argmax也须有定义，不能只有初值而永不记录合法索引。

CycleUpdate按val>max或IsNan(val)更新，使相等有限值保留先出现索引，每个新NaN更新为其索引；已有NaN后有限值比较为false，所以多NaN时最后遇到NaN获胜。此为明确遍历顺序下的政策，不能不核对实际框架/版本就称所有max-pool标准一致；并行归约顺序需保持对应tie/NaN规则。

Ceil输出尺寸用有效kernel=dilation*(ksize-1)+1，候选floor((dim+2pad-effectiveKernel+stride-1)/stride)+1，再对最后窗口起点(out-1)*stride≥dim+pad时减1。stride正、合法pad/kernel、宽整数乘法及负分子向下除法须明确；C++负整数除法向零不总等于DivRtn所需语义。源对称pad公式不能直接覆盖任意非对称padding合同。

从padded坐标减pad还原argmax时，将负坐标clip0只修下界，并不排除上越界、全padding窗口或padding与输入同为-Inf时的错误获胜。应在选择过程中追踪有效输入和索引，不能把无效获胜位置映到0便称结果正确。源baseline本已在有效坐标内记索引，不应为追求“专家式修正”重复减pad。

双线性插值若x1=x0+1，0<x1<w可同时加载左右两角；只有x0有效则单加载左角，y0/y1每行各自检查。源分级片段遗漏x0=-1而x1=0的“仅右角有效”情况，若完整函数无其他分支会丢掉有效贡献。跳过拷贝的槽必须本轮置零，不能保留上轮buffer数据；通道/角点双copy跨度也需有效。

越界角取零与把连续采样坐标clamp到边界得到不同插值值，选择由真实合同决定，不是一个普遍更精确。归一化坐标到像素坐标的定义与边界mask必须一致，无法仅凭一段DataCopy证明所有CV框架语义。
