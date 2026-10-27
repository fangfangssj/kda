---
type: Guide
title: 白盒用例描述的可选、列表与输出语义
description: 保存 Mapper-v1 的数据含义，避免空值、嵌套和顺序转换丢失。
tags:
- task.testing
- topic.implementation.tiling
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-whitebox-design/references/case-mapper/S5_case_json_schema.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-whitebox-design/references/case-mapper/S5_mapped_case_schema.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-whitebox-design/references/case-mapper/02-generate-mapper.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-whitebox-design/references/case-mapper/01-variable-semantics.md
  role: supporting
created_at: '2026-09-11T04:54:25Z'
updated_at: '2026-09-11T04:54:25Z'
platforms: []
---

# 白盒用例描述的可选、列表与输出语义

来源 Mapper-v1 case 的7个顶层字段为id/source/attributes/const_inputs/inputs/outputs/meta。这是该工具版本的数据合同，不是 Ascend C 通用硬件格式。attributes 表达真实算子属性，const_inputs 表达 ValueDepend 等逻辑常量输入值，路由/path/key与测试溯源属于 meta；不能因某字段叫 axis 就否定它可能是实际 Attr。常量值与真实输入槽的关联应显式保存。

shape=null 表达缺省可选输入时不同于 rank0 的[]、长度为0的[0]或count0的TensorList。列表中的 null 子项仍占对应位置，不能过滤后改变其他参数位置；是否允许子项缺省由算子接口决定，而非 JSON 能表示就合法。null 项也要保持 dtype/format 与参数身份。

来源单 Tensor 输入6字段、输出5字段，TensorList外层输入7/输出6字段，子项输入5/输出4字段；输入含data_range、输出不含，子项不重复param_type。count需与实际子项数一致。外层dtype/format不能覆盖允许异构的子项；范围共享规则若有也需逐项校验，不能仅存在同名字段就假设一致。

TensorList是一个参数槽中的子序列，与多个普通输入是不同嵌套结构。source/meta保留原path或network身份，low shape 变体用base_id追溯；唯一ID只能证明标签无冲突，不能证明每个rank或参数组合出现。一个源case只生成一个shape变体时，不能宣称自动覆盖所有等价rank分解。

输出完整shape/dtype/format/param_type应由最终输入、attrs和const值及接口合同决定；同shape不意味着同dtype或inplace。默认ND仅适用于真实支持ND的分支，格式受限或物理storage shape不同的算子不能靠默认填充。JSON对象顺序在通用数据模型中不应当作唯一ABI定义，转换到位置参数时需按显式签名顺序；列表成员顺序始终保留。

只读前30行样本不能验证异构用例集合，最终每条记录仍需要完整结构与语义校验。未知变量不能由名称猜shape、dtype或输出；无可证明映射的案例应记录未覆盖及所缺合同，不能用猜测字段使schema通过。
