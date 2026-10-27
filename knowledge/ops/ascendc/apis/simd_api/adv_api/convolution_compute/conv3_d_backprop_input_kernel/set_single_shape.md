---
type: API
title: SetSingleShape
description: 设置Conv3DBackpropInput在单核上计算的形状，单位为元素个数。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.compute.neighborhood.convolution
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/convolution_compute/Conv3DBackpropInput_Kernel/SetSingleShape.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- a2
- a3
---

# SetSingleShape

## 功能说明

设置Conv3DBackpropInput在单核上计算的形状，单位为元素个数。

## 函数原型

```
__aicore__ inline void SetSingleShape(uint64_t singleShapeM, uint64_t singleShapeK, uint32_t singleShapeN)
```

## 参数说明

**表1**  接口参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| singleShapeM | 输入 | 单核上M的大小，单位为元素。 |
| singleShapeK | 输入 | 单核上K的大小，单位为元素。 |
| singleShapeN | 输入 | 单核上N的大小，单位为元素。 |

## 返回值说明

无

## 约束说明

无

## 调用示例

```
const Conv3DBackpropInputTilingData* tilingData;
// ...初始化tilingData
ConvBackpropApi::Conv3DBackpropInput<weightDxType, inputSizeDxType, gradOutputDxType, gradInputDxType> gradInput_;
// ...设置其它参数
gradInput_.SetSingleShape(singleShapeM_, singleShapeK_, singleShapeN_); // 设置单核计算的形状
```
