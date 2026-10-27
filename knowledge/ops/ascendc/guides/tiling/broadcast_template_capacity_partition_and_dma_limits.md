---
type: Guide
title: 广播模板容量、分核与DMA有效长度
description: 保存通用切分等式，指出全shape可放时内层积与保底容量问题。
tags:
- task.optimization
- topic.implementation.tiling
- topic.implementation.synchronization
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/broadcast/advanced_tiling.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/broadcast/broadcast_design.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/broadcast/code/broadcast_common.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/broadcast/code/broadcast_add_tiling.cpp
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/broadcast/code/broadcast_add_kernel.cpp
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/broadcast/datacopypad_design.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/broadcast/nddma_design.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/broadcast/ub_broadcast_design.md
  role: supporting
created_at: '2026-09-11T06:37:24Z'
updated_at: '2026-09-11T06:37:24Z'
platforms: []
---

# 广播模板容量、分核与DMA有效长度

给每槽容量E，选择split轴使inner=∏outputDims[split+1:]≤E，former=min(dim[split],floor(E/inner))，outer=ceil(dim/former)，tail=dim-(outer-1)*former。总tile数=∏outputDims[:split]*outer，真实tile元素=rows*inner。尾轴太大时inner1直接切尾轴；小shape全可放时split0且inner须排除dim0。

来源ComputeTiling循环在全shape可放时把dim0也乘入innerBelow，却仍声明它等于outputStrides[0]，导致过度缩块。如shape[4,8]、E32会former1、4tiles而非former4、1tile；该例覆盖仍完整，但浪费资源与核分发，不能解释成最佳通用切分。所有轴正、rank1..数组容量、乘积无溢出且E至少合法基本块后才能应用公式。

先F=ceil(totalTiles/C)再U=ceil(totalTiles/F)，tailCore=totalTiles-(U-1)*F可保证使用核的最后正且≤F。若先U=min(C,totalTiles)，再只把最后核当唯一tail，100tiles、C64会F2、末核-26并让前面部分核越界。每核可改为start=bid*F，count=max(0,min(F,total-start))并相应处理全局最后tile，或回收无效核，或q/r均分。

E=AlignDown(UB/(aliveBuf*dtypeBytes*2),32/dtypeBytes)只是相同dtype各槽保守模型。源x/y/z各两份+紧凑源一份实际7槽，按8槽估可保守；多输入紧凑源、Broadcast scratch及其他固定区仍须计入。算不出一块时强行E=alignEle会超预算，aliveBuf0/非法dtype/emptyshape不得靠除数夹1伪造合法结果。int64维度写int32 former/tile字段也需范围核验。

GM→UB DataCopyPad的blockLen是有效字节，UB须为隐式对齐后空间留足且padding值未必有数学意义；UB→GM只写真实有效长度。非对齐尾的API可用性、描述符字段上限、UB起点/行stride仍需遵守，不能说所有非对齐情况无需任何分支。Add/Broadcast的count和shape是否支持任意尾按具体dtype/接口验证，不因DataCopyPad可搬就推出所有计算可直接处理。
