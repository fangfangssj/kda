---
type: Guide
title: CATLASS 轮转 workspace 与 Callback 的收益条件
description: 按缓存、容量、事件和数值合同评估 Matmul 融合。
tags:
- task.design
- topic.implementation.tiling
- topic.performance.analysis
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-design/references/mmad-epilogue-selection.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-design/references/kernels/grouped-matmul.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-design/references/custom-epilogue.md
  role: supporting
created_at: '2026-09-11T04:33:09Z'
updated_at: '2026-09-11T04:33:09Z'
platforms: []
---

# CATLASS 轮转 workspace 与 Callback 的收益条件

整块C workspace的FP32写读流量名义为2*M*N*4字节；按(core,stage)轮转tile的基本容量为m1*n1*coreNum*stages*CBytes。轮转有机会缩小工作集并提高L2命中，L2是cache，不能显式分配，也不保证96–99%命中。整块C也可能命中cache，不能称其必然全部HBM；需计其他并发数据、双C及实际cache行为。来源5个shape的1.15–3.3倍和N≥8192仅是未附本批原始日志的历史经验。

PreloadAsyncWithCallback可把Matmul与Epilogue按槽交接并重叠，前提是匹配Kernel回调接口、缓冲和事件协议。它不是所有融合必然最优，Pingpong也不是不得交付的能力限制。非量化类比量化轮转Kernel时，去掉scale仍需重算dtype、参数、写回和数值合同。int8时enableUnitFlag=false、L0.M=L1.M均需按具体策略核验，不能扩大到全部CATLASS。

来源AtlasA2以L1约512KiB、L0C约128KiB估算：FP16的L1<128,256,512>双缓冲A/B需768KiB，超该容量；改K256需384KiB。INT8相同K512需384KiB。L0C单份128*256*4=128KiB已占满示例容量，额外buffer或更大M/N不能忽略。这不证明FP16 K全局上限256，缩小M/N、减少阶段或换策略会改变限制；A/B不同dtype要分项计算，L0C使用实际L0块而非无条件L1块。

Epilogue UB预算为各staged buffer乘真实份数加常驻计算/scratch，双C gate往往多出C1/C2、反量化、silu等区，别名复用需证明生命周期。来源某HardEvent同类最多8个、双C UB_STAGES=2需10个V_MTE2并挂死，是该平台/实现待核验约束；应同时计事件和容量，不能仅改UB_STAGES就保证无死锁。事件重用/预置/排空也属于协议。

激活/反量化放FP32后再Cast可避免早截断，golden需镜像合同规定的顺序；若合同规定中间FP16舍入，则全FP32路径未必bitwise等价。容量和文本配置通过仅构成静态证据，性能与精度仍需对应变体的实际执行。
