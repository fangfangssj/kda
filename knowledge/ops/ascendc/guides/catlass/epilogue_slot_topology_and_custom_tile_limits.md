---
type: Guide
title: CATLASS Epilogue 槽位与自定义 Tile 边界
description: 保留组件接口和操作数拓扑，修正任意 Tile 列表及零开销推断。
tags:
- task.design
- topic.implementation.tiling
- topic.performance.analysis
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-design/references/custom-epilogue.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-design/references/epilogue-components.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-design/references/kernels/epilogue-patterns.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-design/references/design-document.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-design/references/mmad-epilogue-selection.md
  role: supporting
created_at: '2026-09-11T04:33:09Z'
updated_at: '2026-09-11T04:33:09Z'
platforms: []
---

# CATLASS Epilogue 槽位与自定义 Tile 边界

BlockEpilogue特化具有固定模板槽位与调用顺序，并非可任意追加Tile的通用列表。来源同时把PerTokenDequant描述为5槽，又给出4个运算的概念流水；槽数量、Copy是否属于显式槽、模板形参顺序和实际调用必须以目标特化为准。“再插入一个TileElemWiseGelu”只有现有槽确实容纳它时才成立。

替换某槽需要对齐模板形参、operator()参数、地址/别名合同及Block读取的ArchTag、ElementCompute、TileShape、COMPUTE_LENGTH等成员。接口一致只是必要条件，数学单位、有效长度、同步和临时区也须满足；不能据此保证直接编译或相同性能。来源以TileOneBlkColumnBroadcastMulGelu替换列广播乘，在乘法后加GELU；声称UB增加0仅当GELU真实实现允许该原地别名且无需额外scratch，不能从in-place文字推导。

如现有槽的数据源、调用时点或数量无法表达依赖，可新增DispatchPolicy/BlockEpilogue特化，并重新定义完整Tile序列、UB分配、循环及跨核协议。约80行Tile、500–700行Block只是来源工作量估计，不是技术约束；“大多数融合都能单槽替换”也没有覆盖证据。

来源组件索引包括逐元素TileElemWiseGelu/Silu/Relu/Clamp/Add/Mul/Muls；广播TileBroadcastAdd/Mul、TileBroadcastInplaceByColumn/ByRow；TileCopy及CopyGMToUB/CopyUBToGM、TileCast、TileSwizzle或EpilogueTileSwizzle；量化TileRowBroadcastMul、TileBroadcastOneBlk、TileOneBlkColumnBroadcastMul、TilePerTokenDequant。不同名称可能是文档别名、类型或特化，需核验版本中的真实声明。无TileElemWiseTanh时可评估Exp/Muls/Adds/Div构造或AscendC::Tanh，但前者需数值稳定性，后者需scratch/长度支持，不能视为等精度且零成本替换。

Block索引列ElemwiseOneSource/NoSource、OnlineSoftmaxNoMask、PerTokenDequant、W4A4PerTokenPerChannelDequant、Dequant、GEMM/GEMV、FASoftmax/FARescaleO、MLASoftmax/MLARescaleO。来源EpilogueAtlasA2ElemWiseNoSource为1槽、OneSource为2槽、PerTokenDequant为5槽及950 OneSrcFixpipe均为版本候选。无源逐元素只依赖C；有源需明确额外GM操作数；bias、量化scale和配对gate不是同一种输入拓扑。

纯Matmul可无Epilogue，但有Epilogue必须选真正调用它的Kernel。computeLength=16384或TileShape::COUNT是例子，不是随意常量，也不一定只由L0TileShape唯一推出；要满足实际Tile的单位、尾块和分配。设计文档表格、文件分工和强制槽位审批是Skill流程，本卡仅保留接口知识。
