---
type: API
title: SetSingleWeightShape
description: 设置单核上权重矩阵Weight的形状。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.compute.neighborhood.convolution
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/convolution_compute/Conv3D_Tiling/SetSingleWeightShape.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-04T01:25:11Z'
platforms: []
source_headers:
- include/adv_api/conv/conv3d/conv3d_tiling_base.h
---

# SetSingleWeightShape

## 功能说明

设置单核上权重矩阵Weight的形状。

## 函数原型

```
void SetSingleWeightShape(int64_t singleCi, int64_t singleKd, int64_t singleKh, int64_t singleKw)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| singleCi | 输入 | 单核上输入通道的大小。 |
| singleKd | 输入 | 单核上Weight D维度大小。 |
| singleKh | 输入 | 单核上Weight H维度大小。 |
| singleKw | 输入 | 单核上Weight W维度大小。 |

## 返回值说明

无

## 约束说明

无

## 调用示例

```
// 实例化Conv3D Api
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
Conv3dTilingApi::Conv3dTiling conv3dApiTiling(ascendcPlatform);
conv3dApiTiling.SetSingleWeightShape(singleCi, singleKd, singleKh, singleKw);
```

