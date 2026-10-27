---
type: API
title: SetPadding
description: void SetPadding(int64_t padHead, int64_t padTail, int64_t padUp, int64_t
  padDown, int64_t padLeft, int64_t padRight)
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.compute.neighborhood.convolution
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/convolution_compute/Conv3D_Tiling/SetPadding.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-04T01:25:11Z'
platforms: []
---

# SetPadding

## 功能说明

设置Pad信息。

## 函数原型

```
void SetPadding(int64_t padHead, int64_t padTail, int64_t padUp, int64_t padDown, int64_t padLeft, int64_t padRight)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| padHead | 输入 | D方向前Padding大小。 |
| padTail | 输入 | D方向后Padding大小。 |
| padUp | 输入 | H方向上Padding大小。 |
| padDown | 输入 | H方向下Padding大小。 |
| padLeft | 输入 | W方向左Padding大小。 |
| padRight | 输入 | W方向右Padding大小。 |

## 返回值说明

无

## 约束说明

在调用GetTiling接口前，本接口可选调用。若未调用本接口，默认padHead=0, padTail=0, padUp=0, padDown=0, padLeft=0, padRight=0。

## 调用示例

```
// 实例化Conv3D Api
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
Conv3dTilingApi::Conv3dTiling conv3dApiTiling(ascendcPlatform);
conv3dApiTiling.SetPadding(padHead, padTail, padUp, padDown, padLeft, padRight);
```

