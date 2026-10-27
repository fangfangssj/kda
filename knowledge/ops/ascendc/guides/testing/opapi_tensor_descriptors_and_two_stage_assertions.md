---
type: Guide
title: OP API UT 的 Tensor 描述与两段式断言
description: 保留非连续视图、Scalar 描述和 dtype 合法组合，界定 WorkspaceSize 测试的覆盖范围。
tags:
- task.testing
- topic.implementation.operator_registration
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-ut-develop/references/ut-guide/op-api-ut-guide.md
  role: primary
created_at: '2026-09-11T03:30:43Z'
updated_at: '2026-09-11T03:30:43Z'
platforms: []
---

# OP API UT 的 Tensor 描述与两段式断言

来源 OP_API_UT 框架通过 TensorDesc 描述 shape、dtype、format、数据范围与精度，并可给出 storage offset/stride；ScalarDesc 的 float、整数和 bool 类型要与接口合同匹配。数据范围是生成条件，Precision 是配置，二者自身不证明已执行比较。

非连续视图需要逻辑 shape、stride、offset 和底层 storage 的一致地址范围。例如 shape={5,4}、stride={1,5}、offset=0 的最大访问元素索引为 4+3*5=19，至少需要 20 个元素空间；这是一种布局示例，不是任意转置视图的通用空间公式。须另查负 stride、别名及目标框架支持。

TestGetWorkspaceSize 只对两段调用的第一阶段及其返回状态提供证据。来源的“正常场景”即使设了 Precision 并得到 ACLNN_SUCCESS，只调用该方法仍不能证明执行了 Kernel 或比较了数值。执行接口、workspace 生命周期、stream 完成、所有有效输出和原地状态需独立断言。

非法输入测试应以实际算子的 null、dtype、rank、format、广播及属性限制为准；不能从 Abs 示例推导 DOUBLE 永远非法、rank>8 永远失败。错误返回应对应预期阶段和原因，不能因为某个实现当前拒绝就自动缩小公开支持域。

OpDef 的 DataType 同位置输入输出按完整元组配套，DataTypeList 则有自由组合语义，见[类型组合](../registration/opdef_dtype_combinations_and_binary_config.md)。合法测试应覆盖实际组合，数量 M≥N 不证明逐组对应；随意跨列组合也可能碰巧属于另一条合法元组。来源要求至少三种非法组合不是完整覆盖证明，遇到声明合法但失败的组合需定位接口或实现矛盾。

GE dtype、ACL dtype、CSV token 属不同类型域，如 ge::DT_FLOAT、ACL_FLOAT 与 FLOAT 的文本关系不等于可直接混用枚举。映射和头文件要以框架版本为准，不能把某个 Kernel 类型头路径当所有 Host/ACL 枚举的定义处。
