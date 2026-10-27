---
type: Guide
title: CATLASS 开发实例的装配与 Bias 所有权
description: 保存实现层接口差异，限定槽位、累加类型和固定策略。
tags:
- task.implementation
- topic.implementation.synchronization
- topic.precision
- topic.implementation.tiling
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/architecture/00-overview.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/architecture/01-kernel-assembly.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/custom-epilogue.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/basic-matmul.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/with-epilogue.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/quant-matmul.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/rules.md
  role: supporting
created_at: '2026-09-11T05:12:50Z'
updated_at: '2026-09-11T05:12:50Z'
platforms: []
---

# CATLASS 开发实例的装配与 Bias 所有权

装配通常涉及ArchTag、DispatchPolicy、A/B/C及可选Bias的类型布局、L1/L0 TileShape、BlockMmad、BlockEpilogue和Scheduler，最终由真正调用这些组件的Kernel组合。来源Arch::AtlasA2、MmadAtlasA2Pingpong<true>、Preload/PreloadAsync/Callback及L1<128,256,256>/L0<128,256,64>是具体候选，不能仅因纯Matmul/激活/量化类别就断言整个CATLASS只有一种合法Kernel。Epilogue可以包含AIV Vector计算，不能把它统称为AIC尾处理。

Basic示例CType也绑定输出类型，不能把名字CType一律理解成硬件accumulator；改变CType可能改变GM输出合同。FP16输入、FP32累加及最终FP16输出要在实际Mmad/Fixpipe/Kernel路径上确认，FP32本身也不保证全部误差阈值。

自定义Tile模板参数Arch、Compute、ComputeLength以及ArchTag、ElementCompute、COMPUTE_LENGTH和operator()签名必须满足调用者；参数个数、const引用、原地别名和尾长同样属于接口。ElemWiseOneSource来源有2个计算槽与包含Copy的3项写法，不能把两种计数混为同一声明；PerTokenDequant的5槽后追加第6个Gelu也不等于已有特化可接受。可在实际兼容槽融合运算，或实现新的Block/Policy并重新证明资源和同步。

“BiasAdd+Activation”不能仅把需要双输入的操作替换成单输入Gelu；必须找到Bias被加载并加到正确广播轴的位置。若BlockMmad已加Bias，后续Epilogue再次加会翻倍；独立GM source、Mmad Bias及量化scale各自有不同拓扑。激活必须在完整K累加后且遵循合同规定的中间舍入。

computeLength16384并不直接等于L0<128,256,64>的M*N=32768；可能涉及AIV分片，但必须以实际Block的分片与有效长度为据。固定ComputeLength同时受UB容量、padding和尾mask约束。代码行数、禁止手写任何接口或固定交付文件名属于Skill开发约定，不作为库能力限制。
