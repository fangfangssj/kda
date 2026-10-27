---
type: Guide
title: OpDef 类型组合与二进制配置的一致性
description: 区分按列组合与自由类型组合，并核对目标 SoC 的有效原型、Kernel 输入输出和编译配置。
tags:
- task.design
- topic.implementation.operator_registration
- topic.compilation
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/binary-json-dtype-guide.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/basic-guide.md
  role: supporting
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/utils_api/prototype_register_management/op_param_def/data_type_list.md
  role: constraint
created_at: '2026-09-10T13:38:09Z'
updated_at: '2026-09-10T13:38:09Z'
platforms: []
---

# OpDef 类型组合与二进制配置的一致性

注册算子的支持范围由输入输出基数、dtype、format、属性和目标 SoC 配置共同决定。C++ 模板能编译某种类型，不等于该组合已经通过算子接口暴露；仅修改 OpDef 也不证明对应 Kernel 二进制已经生成。

使用 `DataType` 显式列举组合时，同一位置应解释为一组完整的输入输出类型及格式关系。例如 x 的类型为 `[half,float]`、y 为 `[float,float]`、out 为 `[half,float]`，表达两组配套关系，不能独立做三列的笛卡尔积。使用 `DataTypeList` 时则允许该参数与其他参数的类型/格式自由组合，应按该机制展开，不能仍套按列取值公式。具体约束见 [DataTypeList](../../apis/utils_api/prototype_register_management/op_param_def/data_type_list.md)。

一个 SoC 有差异化配置时，应使用该平台实际生效的原型，再核对源码 binary 配置、生成任务、最终二进制映射及 Kernel 类型。来源中某个示例的 A2/A3 三组、950 五组仅是该算子的配置，不代表平台固定支持数；不同 SoC 的配置完全相同也不必然错误。

binary 配置的 dtype 名称、输入输出顺序、format 和属性应符合所用 OPC 构建系统的实际格式。遗漏合法组合会使支持域不完整；错误扩展组合则可能生成与 Kernel 假设不符的调用路径。不能仅以 JSON 条目数量等于某个列表的最大长度验收。

编译结果还应能区分各变体的产物身份。多个任务写入相同文件名可能覆盖先前产物；已有 [bin_filename 为空的覆盖案例](../../runbooks/compilation/binary_json_bin_filename_empty_overwrites_multi_dtype_o.md)说明了一个具体失败机制，其命名规则须按实际构建版本理解。

验证时，逐组比对最终的类型/格式元组和对应二进制，再用每个支持组合运行输入输出检查；同时测试接口拒绝不支持组合的行为。REQUIRED、OPTIONAL、DYNAMIC 表达参数基数，不能为了放宽 dtype 支持而互相替换。
