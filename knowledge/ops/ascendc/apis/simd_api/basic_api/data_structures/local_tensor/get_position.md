---
type: API
title: GetPosition
description: 获取LocalTensor所在的TPosition逻辑位置，支持TPosition为VECIN、VECOUT、VECCALC、A1、A2、B1、B2、CO1、CO2。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.development_mode.tensor_programming
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/data_structures/LocalTensor/GetPosition.md
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

# GetPosition<a name="ZH-CN_TOPIC_0000002364845741"></a>

## 功能说明<a name="section618mcpsimp"></a>

获取LocalTensor所在的TPosition逻辑位置，支持TPosition为VECIN、VECOUT、VECCALC、A1、A2、B1、B2、CO1、CO2。

## 函数原型<a name="section620mcpsimp"></a>

```cpp
__aicore__ inline int32_t GetPosition() const
```

## 参数说明<a name="section622mcpsimp"></a>

无

## 返回值说明<a name="section640mcpsimp"></a>

LocalTensor所在的[TPosition](../../aux_data_structures/t_position.md)逻辑位置。

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="section17531157161314"></a>

```cpp
// 示例
AscendC::TPosition srcPos = (AscendC::TPosition)inputLocal.GetPosition();
if (srcPos == AscendC::TPosition::VECCALC) {
    // 处理逻辑1
} else if (srcPos == AscendC::TPosition::A1) {
    // 处理逻辑2
} else {
    // 处理逻辑3
}
```
