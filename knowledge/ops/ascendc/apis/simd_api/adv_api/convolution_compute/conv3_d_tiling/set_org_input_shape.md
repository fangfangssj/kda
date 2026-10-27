---
type: API
title: SetOrgInputShape
description: 设置特征矩阵Input的原始形状。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.compute.neighborhood.convolution
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/convolution_compute/Conv3D_Tiling/SetOrgInputShape.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-04T01:25:11Z'
platforms: []
---

# SetOrgInputShape

## 功能说明

设置特征矩阵Input的原始形状。

## 函数原型

```
void SetOrgInputShape(int64_t orgCi, int64_t orgDi, int64_t orgHi, int64_t orgWi)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| orgCi | 输入 | 原始输入通道的大小。 |
| orgDi | 输入 | 原始Input D维度大小。 |
| orgHi | 输入 | 原始Input H维度大小。 |
| orgWi | 输入 | 原始Input W维度大小。 |

## 返回值说明

无

## 约束说明

无

## 调用示例

```
// 实例化Conv3D Api
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
Conv3dTilingApi::Conv3dTiling conv3dApiTiling(ascendcPlatform);
conv3dApiTiling.SetOrgInputShape(orgCi, orgDi, orgHi, orgWi);
```

