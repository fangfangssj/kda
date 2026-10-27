---
type: API
title: SetSingleOutputShape
description: 设置单核上结果矩阵Output的形状。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.compute.neighborhood.convolution
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/convolution_compute/Conv3D_Tiling/SetSingleOutputShape.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-04T01:25:11Z'
platforms: []
---

# SetSingleOutputShape

## 功能说明

设置单核上结果矩阵Output的形状。

## 函数原型

```
void SetSingleOutputShape(int64_t singleCo, int64_t singleDo, int64_t singleM)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| singleCo | 输入 | 单核上输出通道的大小。 |
| singleDo | 输入 | 单核上Output D维度大小。 |
| singleM | 输入 | 单核上Output M维度大小。 |

## 返回值说明

无

## 约束说明

无

## 调用示例

```
// 实例化Conv3D Api
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
Conv3dTilingApi::Conv3dTiling conv3dApiTiling(ascendcPlatform);
conv3dApiTiling.SetSingleOutputShape(singleCo, singleDo, singleM);
```

