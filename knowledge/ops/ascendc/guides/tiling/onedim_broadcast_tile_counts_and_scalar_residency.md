---
type: Guide
title: OneDim 广播tile数与标量驻留
description: 纠正负尾核循环，限定扁平化与标量一次填充的前提。
tags:
- task.optimization
- topic.implementation.tiling
- topic.implementation.synchronization
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/broadcast/code/broadcast_common.h
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/broadcast/code/onedim_add_kernel.cpp
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/broadcast/code/onedim_add_tiling.cpp
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/broadcast/onedim_design.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/broadcast/broadcast_design.md
  role: supporting
created_at: '2026-09-11T06:37:24Z'
updated_at: '2026-09-11T06:37:24Z'
platforms: []
---

# OneDim 广播tile数与标量驻留

OneDim适合每输入为满输出shape的连续Tensor或单元素标量，先验证逐轴广播与连续stride，不能仅以元素积相同判定；[2,3]与[3,2]即使积6也不自动逐元素语义等价。source TryOneDim只比积，需调用方提供上述前提，scalarFlag的bit数也受整数宽度限制。

满输入可双队列逐tile搬，标量可用一块TBuf首次Duplicate后驻留，每核只读取一次GM scalar；后续tile长度不大于首次已填长度才可复用。也可用标量专用Adds/Muls等合法API节省展开，故该方案不是所有算子最快。运行时标量属性可以经tiling传入，但需值变化、dtype/精度和生命周期合同。

ComputeOneDimTiling仅设U=min(coreNum,ceil(N/E))，OneDimCalcCore随后F=ceil(totalTiles/U)，tail=totalTiles-(U-1)*F。它没有多维版本的回收步骤：100tiles、64核得到tail=-26，非尾核仍处理2tile，部分核越界，真正最后数据tile也可能在非尾核按完整E处理。必须重算U或每核独立合法区间，且末数据tile按全局tileId而非最后物理核判断有效长度。

N0时源码强制一核、计算出的tail可能E而loops0；应明确空任务launch约定。资源不足时强制tileNum一个对齐块、int64转int32及core0夹1同样不能替代合法性验证。按真实槽份数核算：两满输入+输出为6份，一满+一标量+输出为5份，源统一IN_NUM+1再乘2可作保守上界。

Duplicate之后Add读取同TBuf需Vector内依赖/编译器合同，跨流水的Scalar读回也按接口确认。double按int64 bitcast铺值是特定API不支持double时的位级方案，不能数值cast成整数；样例模板未实现该特化，Add等计算本身的double支持也不能由拷贝可行推断。验证两输入标量组合、tile尾、100/64分核、NaN/负零及精度，通用固定FP32 1e-5/FP16 1e-3阈值不替代算子误差合同。
