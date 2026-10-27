---
type: Guide
title: Host UT 的属性、CompileInfo 与推导上下文
description: 明确 Tiling、InferShape 和 InferDataType 各自验证内容及上下文实例的一致性。
tags:
- task.testing
- topic.implementation.tiling
- topic.implementation.operator_registration
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-ut-develop/references/ut-guide/op-host-ut-guide.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-ut-develop/references/coverage-guide/test-implementation.md
  role: supporting
created_at: '2026-09-11T03:30:43Z'
updated_at: '2026-09-11T03:30:43Z'
platforms: []
---

# Host UT 的属性、CompileInfo 与推导上下文

Tiling UT 可验证状态、blockDim、TilingKey、TilingData 与 workspace；InferShape UT 验证形状推导，InferDataType UT 验证类型推导。前者不会因为传入 dtype 描述就自动执行后者；若 InferShape 未检查 dtype，其 DOUBLE 用例不应照抄为必然 GRAPH_FAILED。

CompileInfo 应匹配实际 TilingParse 的类型、命名空间、字段与注册路径。来源 BroadcastCompileInfo 位于 Ops::Base，某些自定义类型位于 optiling；名称不具有普适性。示例 {64,245760}、key=102、16MiB workspace 都是输入或占位期望，不是平台常量，也不能在未知字段顺序时直接聚合初始化。

Faker 上下文需保持 OpType、NodeIoNum、IrInstanceNum、Tensor 描述/shape、输入输出位置和 NodeAttrs 一致。调用 GetAttr/GetAttrValue 的路线若缺少必需属性会失败；bool、int64、float 和 string 必须采用实际 IR 类型。某些算子没有属性，不能规定所有 InferShape 必须有 NodeAttrs，也不能把所有 GRAPH_FAILED 判成属性缺失。

Tiling 需要的 dtype、原始/存储 format 与 shape 要由真实上下文提供。属性名相同但位置、类型、默认处理不同仍可能改变路线。上下文构建成功不证明属性读取正确；预期输出应来自接口语义与可解释的 Tiling 行为。

成功测试同时检查有意义的 key/data/workspace 和各输出；失败测试检查目标错误分支确实触发。仅用省略号作 expectTilingData 或仅比较 output shape 的示意片段不是完整可执行验证。包含路径以源文件实际目录计算，不能按架构名称猜层数。
