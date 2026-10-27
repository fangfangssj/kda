---
type: API
title: GetShapeInfo
description: 获取LocalTensor的Shape信息。注意：Shape信息没有默认值，只有通过SetShapeInfo设置过Shape信息后，才可以调用该接口获取正确的Shape信息。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.development_mode.tensor_programming
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/data_structures/LocalTensor/GetShapeInfo.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- '950'
- a2
- a3
- 310p
- '910'
source_headers:
- include/basic_api/kernel_tensor.h
---

# GetShapeInfo<a name="ZH-CN_TOPIC_0000002330927394"></a>

## 功能说明<a name="section618mcpsimp"></a>

获取LocalTensor的Shape信息。注意：Shape信息没有默认值，只有通过[SetShapeInfo](set_shape_info.md)设置过Shape信息后，才可以调用该接口获取正确的Shape信息。

## 函数原型<a name="section620mcpsimp"></a>

```cpp
__aicore__ inline ShapeInfo GetShapeInfo() const
```

## 参数说明<a name="section622mcpsimp"></a>

无

## 返回值说明<a name="section640mcpsimp"></a>

LocalTensor的Shape信息，[ShapeInfo](../../aux_data_structures/shape_info.md)结构体类型。

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="section17531157161314"></a>

```cpp
// 示例获取Tensor的ShapeInfo信息
AscendC::ShapeInfo maxShapeInfo = maxUb.GetShapeInfo();
uint32_t orgShape0 = maxShapeInfo.originalShape[0];
uint32_t orgShape1 = maxShapeInfo.originalShape[1];
uint32_t orgShape2 = maxShapeInfo.originalShape[2];
uint32_t orgShape3 = maxShapeInfo.originalShape[3];
uint32_t shape2 = maxShapeInfo.shape[2];
```
