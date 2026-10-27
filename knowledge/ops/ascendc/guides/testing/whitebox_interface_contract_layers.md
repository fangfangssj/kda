---
type: Guide
title: 白盒接口模型的层次与关联字段
description: 区分注册、推导、ACLNN 和框架绑定，保存可选、动态、标量及别名语义。
tags:
- task.testing
- topic.implementation.tiling
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-whitebox-design/references/source-analysis/02-interface-analyzer.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-whitebox-design/references/source-analysis/03-param-derivation.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-whitebox-design/references/case-mapper/01-variable-semantics.md
  role: supporting
created_at: '2026-09-11T04:54:25Z'
updated_at: '2026-09-11T04:54:25Z'
platforms: []
---

# 白盒接口模型的层次与关联字段

OpDef 的 Input/Output/Attr、ParamType、AddConfig 描述注册模型；proto/infer 定义约束或推导；ACLNN 签名与实际框架 binding 决定调用者能提供什么。一个层面的字段不能直接当另一层面的全部合法域。REQUIRED/OPTIONAL/DYNAMIC 的默认及约束须按目标实现；aclTensor* 参数并不自动等于 REQUIRED，也可能允许 nullptr。

DYNAMIC 是张量序列的模型，不足以推出每个列表1≤N≤50、全部子张量同 shape/dtype 或多个列表长度独立。最小/最大数量、配对输入长度、各成员类型及输出数量需源约束支持。输出数量可依赖多个输入、属性或数据，不能统一跟随一个输入 count。输出 TensorList 与可选子项也各有接口合法性。

aclScalar 与 rank-0 Tensor 是不同 ABI 对象；测试模型可用标量形状表示逻辑值，但不能据此改写参数类型、指针或 dtype 转换。DtypeScalarToTensor2 之类 helper 的出现不是全部 scalar 合法类型证明。输入输出 dtype 的合法组合应以元组关系保存，不能只存 primary_dtype 再任意交叉乘积。

input/caller/internal 表示变量来源，不是是否依赖输入的同义词。size、align(size)、reduce 轴归约等内部量仍由输入决定；调用者可选属性与内部编码也应分别建模。sync_with 仅在字段存在严格依赖时适用，shape 相同不代表 dtype/format 也相同；依赖图需排除任意长度环，不能只查相互两节点。

缺少显式 Output 或名称相同不能单独证明 inplace；实际修改、别名及 API 返回合同才是依据。测试中建模的隐式输出不等于源码新增了注册 Output。输出重命名要保留绑定映射，否则按名字推导别名可能改变意义。

框架 schema 的默认值、重载、别名和 wrapper 派生逻辑共同决定曝光接口。一次名称搜索 not_found 不证明 API 不存在，默认 schema 也可能漏 overload。Attr 默认值不仅是 number/null，还可为 bool/string/list；JSON 模型不得为适配狭窄 schema 静默改类型。由实际 Attr 声明决定 axis 等是否属性，不能按字段名一概禁止。
