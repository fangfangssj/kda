---
type: Guide
title: CATLASS Matmul 模板索引与实例化合同
description: 保存来源示例能力及 Host/Device 参数边界，限制固定最优配置。
tags:
- task.design
- topic.implementation.tiling
- topic.performance.analysis
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-design/references/matmul-templates.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-design/references/kernels/matmul.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-design/references/kernels/quant-matmul.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-design/references/design-document.md
  role: supporting
created_at: '2026-09-11T04:33:09Z'
updated_at: '2026-09-11T04:33:09Z'
platforms: []
---

# CATLASS Matmul 模板索引与实例化合同

来源例子索引：00_basic_matmul为Multi Buffer基线；04_padding_matmul为ND padding；06_optimized_matmul组合预取、NZ padding、ShuffleK及小M搬运；09_splitk_matmul跨核切K；21_basic_matmul_preload_zN预取zN；22_padding_splitk_matmul结合ND padding；25_matmul_full_loadA全载A；31_small_matmul减少小shape标量开销；34_single_core_splitk_matmul切K与写出优化；12_quant_matmul为W8A8；27_matmul_gelu融合GELU；30_w8a16_matmul、32_w4a8_matmul、38_w4a4_matmul_per_token_per_channel_dequant为相应量化候选。20_matmul_bias、03_matmul_add及26/28激活例可提供不同源拓扑。这是固定reference的导航，未证明当前上游所有路径/符号存在。

Common在M/N分任务，忽略cache时单任务输入量约m1*K+K*n1个相应类型元素。MultiCoreSplitK增加K并行且需ReduceAdd；SingleCoreSplitK按其实现切K并累加，原子初始化/完成边界不能遗漏。Padding增加转换和存储开销，stride非512B只是来源性能场景，不是所有搬运合法性的界限；M<8逐行搬运、小shape任务少于核数、A全L1驻留均须结合资源和实际测量。

DispatchPolicy候选含MmadAtlasA2Pingpong、Preload、Small、SingleCoreSplitk、FullLoadA、PreloadAsyncWithCallback；Kernel候选BasicMatmul、MatmulActivation/MatmulEpilogue、SmallMatmul、OptimizedMatmul、SplitkMatmul/SingleCoreSlicekMatmul、MatmulFullLoadA及QuantMatmulMultiStageWorkspace。名称与模板参数按真实版本核验。量化要求scale的语义不等于只能使用一个Kernel/Callback形式，来源量化占位文件的“必须”只适用于其选定实现。

dtype、transA/transB、layout、Swizzle参数、激活和TileShape凡影响模板类型，都需要实际已编译的合法实例及Host dispatch。RowMajor/ColumnMajor与转置映射只在原始存储合同一致时成立，不能用一个bool替代leading dimension。示例M≥N选Swizzle<3,0>、否则<3,1>，及L1<128,256,256>/L0<128,256,64>是候选，不能保证所有形状最优。

Host main中的shape/Tiling/workspace管理与Device计算属于不同执行域。DeviceGemm是Host启动适配器，Device body应按对应Kernel::Params及Kernel{}(params)调用；Arguments/Params不保证同构。设计模板把所有非量化Kernel统一写成三个模板参数也仅是示意，BasicMatmul是否接收Epilogue要看声明。advanced/basic_matmul_aclnn提供宏TilingData、Host注册/推导/Tiling、Device封装及aclnn两段调用的适配例。

workspace应由实际Kernel::GetWorkspaceSize或完整等价公式给出；Split-K仅写M*N*sizeof(float)可能漏split份数或其他区。ptrScale/ptrPerTokenScale、stage数、缓冲dtype、指针是否含框架前缀均属于实际launch ABI。来源推荐FP32浮点累加、INT32整数累加及末尾Cast，应与真实精度合同保持一致，不能仅改变CType名称推断硬件累加模式。
