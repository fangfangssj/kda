---
type: Guide
title: RegBase API 的操作数、掩码与存储合同
description: 补全 API 汇总中的模板、有效 lane、分组归约和局部存储约束，区分候选目录与实现证据。
tags:
- task.implementation
- topic.memory
- topic.implementation.synchronization
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-regbase-best-practice/references/api/regbase_api_reference.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-regbase-best-practice/references/api/regbase_api_whitelist.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-regbase-best-practice/references/reference-ops/open_source_operator_table.md
  role: supporting
created_at: '2026-09-11T03:30:43Z'
updated_at: '2026-09-11T03:30:43Z'
platforms: []
---

# RegBase API 的操作数、掩码与存储合同

来源以 CANN 9.0.0 的 RegBase API 组织方式描述普通 Kernel 外壳 basic_api、VF 内 reg_compute 和部分 adv_helpers，并指出该版本 kernel_macros 中 MicroAPI 与 Reg 的别名关系。命名空间、头文件与支持平台仍由实际 SDK 决定；汇总表和白名单提供候选接口族，不证明所有版本存在同名符号，也不证明未收录接口不可用。

RegTensor 承载 VF 的寄存器操作数，MaskReg 表达有效 lane 或比较条件；LoadDist/StoreDist 控制搬运分布。三者不能代替同步。CastTrait 的 RegLayout、SatMode、MaskMergeMode、RoundMode 影响解释、饱和、非活动 lane 和舍入，应作为数值合同而非默认常量隐藏。

UpdateMask(uint32_t&) 的来源版本会更新引用参数。循环进度只能有一个明确的维护者：保存本轮 active=min(remaining,VL)，用稳定 offset 推进；不能既让 UpdateMask 消耗 remaining 又重复递减，或用修改后的 remaining 反推本轮地址。active mask 不保证整向量无掩码 LoadAlign 的物理访问合法，tail 的逻辑长度与实际可读写空间需分别证明。

Add/Sub/Mul 一类按寄存器参数类型推导，来源签名的 MaskMergeMode 默认 ZEROING；Div 另有精度模式参数。Adds/Muls 的标量类型和 RegTensor 类型参数不是同一角色。Compare/Compares 输出的条件掩码与控制参与比较 lane 的掩码不同，Select 的选择条件也不等于数据已同步。

Cast 的输入/输出寄存器类型与 const CastTrait 引用共同约束转换；Truncate 的模板与舍入模式另有约定。Reduce、ReduceDataBlock、PairReduce 的分组范围、结果位置和类型合同不同，不能按名字视为等价归约。数学、位运算、比较选择、索引重排、归约和转换族均需按具体重载核对输入输出类型及有效 lane。

LoadAlign(U& reg, __ubuf__ T* addr) 的来源基础重载没有 mask；StoreAlign 有 mask。带 dataBlockStride/dataMode 的重载改变地址布局，不能仅按元素数复用 stride。Load/Store、count Store、非对齐搬运的 Pre/Post 状态以及 Gather/GatherB/Scatter 的索引类型、范围和掩码各有独立约束。Pre/Post 状态要属于实际搬运链，不能跨不相容地址流随意复用。

LocalMemBar<src,dst> 在来源中为 __simd_callee__ 的局部 UB 访问排序，MemType 区分向量/标量读写；它不是 TQue、跨流水事件或跨核屏障。普通外壳的 TQue/LocalTensor 与 VF RegTensor 职责见[数据流层次](dataflow_layers.md)，完成与资源复用见[同步层次](synchronization_layers.md)。

算子目录的七条技术提示可用作实例检索方向：foreach_add_scalar 的路由与 scalar，foreach_addcmul_scalar 的多输入 VF 链，foreach_norm 的归约 tail/partial，add_rms_norm_dynamic_quant 的 FP32 中间量及量化写回，add_rms_norm_quant 的 split-reduce 共用结构，group_norm_v2 的 Welford/two-pass/generalized 路线，max_pool3d_with_argmax_v2 的多路由复杂 Tiling。它们是目录描述，未在本批读取各算子实现；不能据此声称所有候选都实现了 RegBase 或上述路线已验证。其余名称/分类/路径仅为导航，留在 Skill。
