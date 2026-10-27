---
type: API
title: SetInputShape
description: 设置特征矩阵Input的形状：Batch、Channel、Depth、Height、Width。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.compute.neighborhood.convolution
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/convolution_compute/Conv3DBackpropFilter_Tiling/SetInputShape.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-04T01:25:11Z'
platforms: []
---

# SetInputShape

## 功能说明

设置特征矩阵Input的形状：Batch、Channel、Depth、Height、Width。

## 函数原型

```
void SetInputShape(int64_t n, int64_t c, int64_t d, int64_t h, int64_t w)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| n | 输入 | 输入Input的Batch值。 |
| c | 输入 | 输入Input的Channel值。 |
| d | 输入 | 输入Input的Depth值。 |
| h | 输入 | 输入Input的Height值。 |
| w | 输入 | 输入Input的Width值。 |

## 返回值说明

无

## 约束说明

无

## 调用示例

```
auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
ConvBackpropApi::Conv3dBpFilterTiling conv3dBpDwTiling(*ascendcPlatform);
conv3dBpDwTiling.SetInputShape(n, c, d, h, w);
```
