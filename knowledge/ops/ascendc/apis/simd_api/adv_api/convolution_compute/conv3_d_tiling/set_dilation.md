---
type: API
title: SetDilation
description: 设置Dilation信息。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.compute.neighborhood.convolution
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/convolution_compute/Conv3D_Tiling/SetDilation.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-04T01:25:11Z'
platforms: []
---

# SetDilation

## 功能说明

设置Dilation信息。

## 函数原型

```
void SetDilation(int64_t dilationD, int64_t dilationH, int64_t dilationW)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| dilationD | 输入 | D方向Dilation大小。 |
| dilationH | 输入 | H方向Dilation大小。 |
| dilationW | 输入 | W方向Dilation大小。 |

## 返回值说明

无

## 约束说明

在调用GetTiling接口前，本接口可选调用。若未调用本接口，默认dilationD=1, dilationH=1, dilationW=1。

## 调用示例

```
// 实例化Conv3D Api
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
Conv3dTilingApi::Conv3dTiling conv3dApiTiling(ascendcPlatform);
conv3dApiTiling.SetDilation(dilationD, dilationH, dilationW);
```

