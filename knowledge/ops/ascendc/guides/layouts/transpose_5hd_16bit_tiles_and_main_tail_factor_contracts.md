---
type: Guide
title: 5HD 二维转置的16位分块及主尾因子
description: 保存R/C/RC三种切法与16×16地址表，明确逻辑和物理长度。
tags:
- task.optimization
- topic.data_movement.layout_transform
- topic.implementation.tiling
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/conversion/transpose_vconv_5hd.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/conversion/templates/dav3510/transpose_transdata_5hd.template
  role: supporting
created_at: '2026-09-11T06:55:07Z'
updated_at: '2026-09-11T06:55:07Z'
platforms: []
---

# 5HD 二维转置的16位分块及主尾因子

此key10007模板只按16bit实例化，归约后2D perm[1,0]、R>5为Host选择。TransDataTo5HD在该类型路径组织16×16块；不能据这一实现限制宣称该API永远只支持16位。输入/输出各双buffer，单份AvailableUbSize=总预算/4，再按实际可用UB及保留空间约束。

若16行×C对齐宽能装下，则R分核、C全载；否则若16列×R对齐高能装下，则C分核、R全载；否则R分核且UB再沿C切，最小16×16仍不能装入时拒绝该方案。主核/尾核与主UB/尾UB四组r/c因子、对齐块数和循环次数必须一致。R/C分核单位16元素，UsedCoreNum-1只是本实现最后核定位，不代表任何尾核长度惯例通用。

ComputeRConv用repeat=r/16、外循环cAlign，源地址i*c+j*16，目的地址(i+j*16)*r；ComputeCConv对称，repeat=c/16、外循环rAlign。这里r/c是供Vector访问的物理对齐元素长度，rAlign/cAlign是16元素块数，必须满足对应关系。把真实非整除尾长度直接传入r/16会丢行，填对齐长度却不给UB空间也会越界。16×16转置只让有效元素重排，不自动使padding数值参与业务；仅输出真实有效范围可避免无效值泄漏。

CopyIn/Out按三种切法配置GM字节gap及UB连续对齐行距，C整齐的R分核输入可连续，R整齐的C分核输出可连续。最后核/最后loop通过原RLen/CLen减已处理量裁尾。所有非尾分支也需确实在GM内，DMA blockCount和TransData repeat/stride的窄字段限制需Host显式满足，不能把长边放repeat就无上限。

SetBasePocessData中循环偏移始终使用MainCoreUbFactor，尾核DMA长度却使用TailCoreUbFactor；若Host允许两者不同，多次尾核loop的起点将与消费长度不匹配，需证明二者相等或分别计算偏移。RC输出核偏移用rUbSplitPara.MainCoreUbFactor而非rSplitPara.BlockFactor，同样要求RC路径该两值相等。仅有Kernel和Host说明，不能假设缺失Host已经保证所有这些等式。

双buffer队列生命周期是Alloc→DMA/Vector→EnQue→DeQue→消费→Free，实际性能需测量；该实现的方向选择、R>5和满核偏好均不保证正收益。测试主尾核×主尾UB、R/C均不16对齐、RC双切及原始16位bitpattern。
