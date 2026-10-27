---
type: Guide
title: 算子文档模板中的接口事实与占位值边界
description: 提取两段调用、输入状态、图模式和 Tiling 的知识，避免把示例产品表与容量表当成能力证据。
tags:
- task.design
- topic.implementation.operator_registration
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-docs-gen/references/aclnn-api-doc-template.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-docs-gen/references/detailed-design-template.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-docs-gen/references/operator-readme-template.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-docs-gen/references/operator-readme-template_bak.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-docs-gen/references/requirement-analysis-template.md
  role: supporting
created_at: '2026-09-11T03:15:06Z'
updated_at: '2026-09-11T03:15:06Z'
platforms: []
---

# 算子文档模板中的接口事实与占位值边界

来源两段式 ACLNN 合同中，GetWorkspaceSize 生成所需 workspace 字节数及包含计算流程的 executor，执行接口使用匹配的 workspace/size/executor/stream。具体参数与生命周期以目标头文件和实现为准；模板中的算子名、参数表与产品勾选都不能证明一个实际接口已支持。

输入状态需区分 Optional 缺省空指针、存在但零元素 Tensor、合法零维标量，以及不连续 storage。类型推导、广播、rank、逻辑/物理 shape、stride/offset 和输出数量共同限定接口。原地广播结果需适配真实可写目标的形状与别名规则；不能用新建输出的规则默认允许扩展原地 storage。

模板 Add 示例自身存在 alpha 标为预留且传空、错误表又要求 alpha 非空，以及空输入/输出说明不完全一致等冲突。这些样例不能作为 aclnnAdd 的真实版本签名、dtype 全集、产品支持或错误码证据。y=x1+alpha*x2 仅保留为所给数学示意；实际 alpha 默认值、类型转换与非有限行为要由目标合同确定。

op_api、op_host、op_kernel、op_graph 分别承担接口封装、Host 定义/推导/Tiling、设备计算和图适配等责任；它们的目录拓扑是工程约定。eager、torch.compile、GE 静态/动态图模式是不同调用路径，某一路通过不证明其余可用。GM→UB→计算→UB→GM 只描述常见 Vector 路径，Cube 还涉及自己的 L1/L0 数据流，不能按模板图把 Cube 当 UB ALU。

TilingKey 选择已编译的实现，TilingData 承载运行数据；模板片段未完整设置 blockDim/workspace、未消费 tileLength 或未给出模板声明，不是可直接运行的证明。数据类型、字段宽度、分支实例、参数初始化和最终消费者必须闭合，详见[分发合同](tiling_dispatch_and_data_contracts.md)。

UB 预算按实际分配的输入、输出、临时、padding 和每个缓冲份数计算，并计入阶段同时存活资源。模板固定 184/248KiB 及未给定义的 GetUBSizeInBytes/GetRuntimeUBSize 名称不构成跨版本资源查询/API 支持证据。具体平台容量与系统预留要按实际实现确认。

确定性取决于算子、模式与累加顺序，不能从文档中的默认勾选推导；提高精度与固定累加顺序也有不同资源/性能代价。“FP16 双千分之一、FP32 双万分之一”是模板默认示意，容差、非有限比较及逐位要求必须来自目标验收语义。产品排版、表格字段、质量清单和迭代规划不迁入知识正文。
