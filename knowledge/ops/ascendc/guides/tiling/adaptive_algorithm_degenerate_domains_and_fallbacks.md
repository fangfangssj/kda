---
type: Guide
title: 自适应算法的退化域与 Fallback
description: 保存多路径选择，纠正D为1直通和未执行Tiling的路径。
tags:
- task.optimization
- topic.precision
- topic.implementation.tiling
- topic.implementation.synchronization
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/cake-evo/references/meta_prompts/strategies/perf_06_multi_algo_selection.md
  role: primary
created_at: '2026-09-11T05:59:16Z'
updated_at: '2026-09-11T05:59:16Z'
platforms: []
---

# 自适应算法的退化域与 Fallback

多算法选择应先满足capability与数学合同，再在相同输入输出语义下比较成本。固定8192、1024或SmallReduce等只是源实现阈值，dtype、scratch、core资源和统计输出可能改变可行域；拥有更多分支不证明全场景最优。

LayerNormV3源300系列TwoPass、400Welford、500TwoPassPerf、600NoReduce并结合dtypeKey。normalize长度1时，对有限x和正epsilon，mean=x、variance0、normalized0，带beta输出beta、无beta输出0，rstd=1/sqrt(epsilon)；不是简单pass-through x。epsilon0、非有限x和独立统计输出需真实合同，省reduce不能省这些结果。

可选gamma/beta四组合分别执行norm*gamma+beta、norm*gamma、norm+beta、norm；if constexpr条件须真编译参数，普通tiling字段不能直接当编译常量。FusedMulDstAdd与独立Mul/Add舍入不总相同。Welford/TwoPassPerf名称或更精确rsqrt也不代替完整输出比较。

Gather key0Scalar/1Transpose/2LastDim，Transpose把gather轴移动以适合局部向量布局，仍需还原输出轴顺序，计入搬运/转置/索引成本。源dim以uint64读后(dim+rank)%rank不能正确表达所有负axis，也会把非法大axis取模成合法；先在有符号域验证-rank≤axis<rank、rank>0，再规范化一次。GetAttrPointer片段的调用/解引用括号也不完整。LastDim分支仅构造tilingObject后return success，未调用实际SetKernelTiling时不能证明tiling已生成。

DequantBias N≤8192可缓存完整weightScale/bias在行间复用，较大按列块处理；8192FP32一份32KiB不代表全部输入/计算/输出均放得下，activateScale逐行与bias单位仍需对应。Gamma和bias缺省、混合dtype、尾列及N0各自参与预算。

Gemma Normal/SplitD/MergeN/SingleRow候选，MergeN来源仅numColAlign≤SmallReduce且isSoc910B，rowFactor=ubSize/oneRowBufSize还需扣固定scratch、检查≥1并cap真实行数；不能盲目把所有小D通用成此平台专用路径。RMSQuant numCol≤sliceSize走Fast否则Slice，gamma可跨同权重行缓存，但输入/输出offset必须随pid更新，例while只展示++pid不保证offset推进；slice统计应覆盖全行。

Scatter总包≤max_ub选择Small，否则input piece×indices piece两层处理。totalSize与max_ub单位必须相同且包含indices/updates及升精临时，不是只把input塞进UB就安全；每piece索引范围和重复更新语义仍需保持。source__CCE_AICORE__100自定义Reduce、else内置仅定义这个分支，不能据文字把__NPU_ARCH__3003当同一宏体系。
