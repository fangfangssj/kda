---
type: Guide
title: 平台注册、数据类型与 Shape 限制
description: 保存具体配置差异，限定宏、动态标志与硬件能力推断。
tags:
- task.implementation
- topic.precision
- topic.implementation.tiling
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/cake-evo/references/meta_prompts/strategies/dtype_05_bf16_special_handling.md
  role: primary
created_at: '2026-09-11T05:47:04Z'
updated_at: '2026-09-11T05:47:04Z'
platforms: []
---

# 平台注册、数据类型与 Shape 限制

AddConfig注册可选平台路径，还需相应dtype、shape、tiling和Kernel实现才能运行；注册名称本身不自动产生最优性能。来源membase常用于ascend910b/910_93，regbase示例用于ascend910_95，kirinx90有GetKirinCoreConfig；这是算子版本配置，不能从名称推断全芯片统一功能。

AdaptiveAvgPool/SoftmaxGrad来源310p配置仅FLOAT/FLOAT16；RMSNormQuant在310p/310b/kirinx90仅FP16、云侧另BF16；Gather Kirin列FP16/FP32/INT32。应逐算子核验，不把“面向推理所以无需BF16”当硬件能力原因。Embedding 910_95配置grad多dtype、sort_indices INT32/INT64及AutoContiguous也须按合法组合保持索引ABI。

foreach_add_scalar regbase输入列表含FP16/FP32/INT32/BF16，但文字声称新增FP16输出，所示片段只证明输入列表，不能据此补齐输出矩阵。AutoContiguous可能涉及框架转换与额外流量，不自动等于Kernel支持任意stride。DynamicCompileStaticFlag、DynamicFormatFlag、DynamicRankSupportFlag、DynamicShapeSupportFlag是声明能力，仍需InferShape与tiling真实支持；不能保证任意新shape永不重编译，固定attr也不必然排除动态shape。

ScaledMaskedSoftmax源在__NPU_ARCH__3003屏蔽BF16 key2，Host也必须拒绝该组合，不能留下进入空分支而不写输出的合法tiling。ASCEND910_95路径maxDim8192、softmaxBuff64K，对照其他4096/32K是该算子限制/暂存预算，不是芯片UB总量或全部API维度上限。__CCE_AICORE__与__NPU_ARCH__并非可互换的编码体系。

TransQuantParam来源910_95允许(g,n)，其他仅(1,n)/(n,)；GetChipFeature查询AICoreintrinsicDtypeMap/Intrinsic_data_move_l12bt/bf16可作为该功能探测，返回状态与实际路径必须处理，单个搬运指令支持不证明所有BF16算术支持。不能硬编码8核/固定256 tile替代真实资源查询，但平台检测也不能替代输入范围校验。

所有平台与dtype标记在本批仍是来源候选范围，未运行硬件验证，知识卡保持draft及未定位平台；不把历史“最优”“50%以上精度提升”作为验证结果。
