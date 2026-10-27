---
type: Guide
title: 类型化 TensorList、可选参数与输出
description: 保存Foreach及双量化接口，补齐缺失参数状态和输出合同。
tags:
- task.implementation
- topic.precision
- topic.implementation.tiling
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/cake-evo/references/meta_prompts/strategies/dtype_01_mixed_precision.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/cake-evo/references/meta_prompts/strategies/dtype_02_template_kernel.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/cake-evo/references/meta_prompts/strategies/dtype_03_tilingkey_dispatch.md
  role: supporting
created_at: '2026-09-11T05:51:16Z'
updated_at: '2026-09-11T05:51:16Z'
platforms: []
---

# 类型化 TensorList、可选参数与输出

Foreach的ParamType(DYNAMIC)表示TensorList，不能因增加dtype就将普通单Tensor改为列表。KernelForeachBase负责逐Tensor地址、长度及分块；来源只说64位地址表，未给完整描述符ABI，不能假定裸数组就是框架TensorList或把相对offset当绝对GM指针。多列表和scalar-list需元素数、逐项shape/dtype及空Tensor合法性匹配，AutoContiguous产生的框架复制也应计入端到端成本。

foreach_add_scalar示例对应x[FP16,FP32,INT32,BF16]与scalar[FP16,FP32,INT32,FP32]，不是各列表笛卡尔积。文字另称scalar FP64/INT64或910D更多组合，所给表未证明。DtypeScalarToTensor2片段返回FP16→FP16、FP32→FP32、BF16→FP32、INT32→INT32；DtypeTensor2Scalar文字又为浮点→FP32、INT32→INT64，是不同方向/helper合同，不应合并为一张自动互逆表。Scalar先宽类型保存也不保证转回窄类型无溢出或舍入。

AddRMSNormCast来源y1固定FP32、y2随输入，另有残差与归一化舍入点；增加BF16分支不能把所有输出都改成T。AddRMSDynamicQuant按dst_type选择INT8/HIFLOAT8/E5M2/E4M3FN，InferDataType需先验证枚举在允许集合、平台及全部已启用输出一致，不可裸static_cast任意整数后SetOutputDataType。

双量化输出y1/y2可用两套smooth_scale，复用公共残差/统计后仍要分别计算smooth后的amax及outScale；output_mask至少验证长度与每位含义，不读取未启用输出/缺失smooth指针。smooth_scale1/2、beta是OPTIONAL，空shape、未知shape、nullptr不是同一状态；CheckOptionalShapeExisting用ShapeSize>0只适合已知合法shape，不能把动态-1自动判作不存在。标量rank0的ShapeSize通常不等于rank，需按实际描述符规则。

Modulate文字四种scale/shift状态，代码只实现both、no_scale、no_shift三分支；两者皆空被归到no_scale，若该函数仍读shift就错误。应明确both absent的identity/default公式或在Host拒绝，分别推导UB数量和GM访问。可选参数不只是少分配buffer，还决定实际数学公式，15–20%收益需测量。

AdaGrad来源要求var/accum/lr/grad与输出dtype一致，Host读取descriptor前检查存在性，shape还需明确lr标量/广播及update_slots合同。模板updateSlots=false的DAG Init没有accum_out，若外部输出仍为required，须已有别名或显式copy保证其值，不能因为不更新状态就留下未写输出。

TransQuantParam的scale为FP32、输出INT64/UINT64、offset可选；两种输出是相同位打包载荷的不同有符号解释，不能用empty_like(scale)的FP32输出代替。支持更多dtype、shape或可选分支是能力扩展，不能把同类型全FP32对照自动判为更差精度。
