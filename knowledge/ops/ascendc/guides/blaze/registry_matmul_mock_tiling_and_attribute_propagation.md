---
type: Guide
title: Blaze 注册 Matmul 的 mock Tiling 与属性传递
description: 保存 GemmUniversal 组件连接，明确转置、形状、资源和参数初始化缺口。
tags:
- task.implementation
- topic.implementation.tiling
- topic.memory
- topic.implementation.operator_registration
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/matmul_blaze_example/README.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/matmul_blaze_example/op_api/aclnn_matmul_blaze_example.cpp
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/matmul_blaze_example/op_api/aclnn_matmul_blaze_example.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/matmul_blaze_example/op_api/matmul_blaze_example.cpp
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/matmul_blaze_example/op_api/matmul_blaze_example.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/matmul_blaze_example/op_graph/matmul_blaze_example_proto.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/matmul_blaze_example/op_host/arch35/matmul_blaze_example_tiling.cpp
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/matmul_blaze_example/op_host/arch35/matmul_blaze_example_tiling_data.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/matmul_blaze_example/op_host/arch35/matmul_blaze_example_tiling_key.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/matmul_blaze_example/op_host/matmul_blaze_example_def.cpp
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/matmul_blaze_example/op_host/matmul_blaze_example_infershape.cpp
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/matmul_blaze_example/op_kernel/matmul_blaze_example_arch35.cpp
  role: supporting
created_at: '2026-09-11T04:05:46Z'
updated_at: '2026-09-11T04:05:46Z'
platforms: []
---

# Blaze 注册 Matmul 的 mock Tiling 与属性传递

来源明确是普通2D BF16 GEMM骨架，A[M,K]、B[K,N]、C[M,N]，M/N/K≥1，目标DAV_3510/ascend950；它不含真实SWAT Tiling。C固定BF16，需输入dtype/format、rank、K匹配与输出shape在实际入口核对。OpDef的动态rank/format标志不使batch或其他dtype自然成立。

公开ACLNN第一段和图proto均含transA/transB，但L2未将它们传给L0，L0 launch未带OP_ATTR，InferShape与Tiling也始终按非转置[M,K]×[K,N]解析。应明确拒绝true或完整传递与实现四种转置形态，不能因参数出现在声明中就称支持。Contiguous使存储连续并不替代数学转置。

Tiling令baseM/N/K=min(align_up(dim,16),128)，不是README简写的永远128；L1块等于base块，l1BufferNum=2、l0cDB=1，tail参数多为1/0固定占位，核数min(ceil(M/baseM)*ceil(N/baseN),48)。48和tile上限未查询实际平台，tail参数必须符合真实scheduler语义。需要L1/L0容量、加载布局、核数域及尾块证明，不能把mock当优化解。

Host将uint64形状转uint32、AlignUp返回uint32、mTiles*nTiles用uint32相乘，均需预先限制范围和负值；GetInputShape/Dim未全部在Tiling判空/判rank。将0tile强制16、0核强制1不能自行实现M/N/K为空的合同，图InferShape当前明确拒绝非正维。workspace0只在实际Kernel无额外GM暂存需求时成立。

Kernel以ProblemShape(M,N,K,1)连接MatmulMultiBlockBasic、BlockMmad、BlockSchedulerMatmulBasic与BlockEpilogueEmpty到GemmUniversal；A/B/C使用NDExtLayout，BiasType=float但biasGmAddr=null，空epilogue不增加Bias/量化等功能。mmParams包含地址、L1/L0块和stage数；scheduler包含tile、tail、cache、stride/batch字段，两者必须一致。

BlockMmadParams、BlockSchedulerParams、EpilogueParams和总Params在来源未统一值初始化，若真实库增加或要求其他字段，未赋值成员可能被读到。仅给列出的字段赋值不能证明完整ABI；须使用匹配版本构造/默认参数并按实际定义核对。isHf32=0、srcNdStride=1、innerBatch=1属于这个模式，不是任意ND或batch布局默认。

TilingData有pack8/alignas8、uint32几何字段、uint8stage与reserved，Host/Device布局应有显式大小/偏移验证。空tiling_key头仅表示该例未使用模板键声明，不产生任何新key能力。Blaze/tensor_api实际库本次未由这些占位README提供，源码静态审查不能称该骨架已编译/上板通过。
