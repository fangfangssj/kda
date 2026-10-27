---
type: API
title: SetWeightShape
description: 设置权重矩阵Weight的形状。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.compute.neighborhood.convolution
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/convolution_compute/Conv3DBackpropFilter_Tiling/SetWeightShape.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-04T01:25:11Z'
platforms: []
---

# SetWeightShape

## 功能说明

设置权重矩阵Weight的形状。

## 函数原型

```
void SetWeightShape(int64_t cout, int64_t cin, int64_t d, int64_t h, int64_t w)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| cout | 输入 | 设置GradOutput的Channel值。 |
| cin | 输入 | 设置Input的Channel值。 |
| d | 输入 | 设置Weight的Depth值。 |
| h | 输入 | 设置Weight的Height值。 |
| w | 输入 | 设置Weight的Width值。 |

## 返回值说明

无

## 约束说明

无

## 调用示例

```
optiling::Conv3DBackpropFilterTilingData tilingData;
auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
ConvBackpropApi::Conv3dBpFilterTiling conv3dBpDwTiling(*ascendcPlatform);
conv3dBpDwTiling.SetWeightShape(cout, cin, d, h, w);
```
