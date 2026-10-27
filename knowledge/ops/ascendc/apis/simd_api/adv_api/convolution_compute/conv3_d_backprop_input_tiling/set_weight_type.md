---
type: API
title: SetWeightType
description: 设置权重矩阵Weight的位置、数据格式、数据类型信息，这些信息必须与Kernel侧的设置保持一致。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.compute.neighborhood.convolution
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/convolution_compute/Conv3DBackpropInput_Tiling/SetWeightType.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-04T01:25:11Z'
platforms: []
---

# SetWeightType

## 功能说明

设置权重矩阵Weight的位置、数据格式、数据类型信息，这些信息必须与核函数（Kernel）侧的设置保持一致。

## 函数原型

```
void SetWeightType(ConvCommonApi::TPosition pos, ConvCommonApi::ConvFormat format, ConvCommonApi::ConvDtype dtype)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| pos | 输入 | Weight在内存上的[位置](../../../general_description_and_constraints.md#table07372185712)。当前仅支持TPosition::GM。 |
| format | 输入 | Weight的数据格式。当前仅支持ConvFormat::FRACTAL_Z_3D。 |
| dtype | 输入 | Weight的数据类型。当前仅支持ConvDtype::FLOAT16或者ConvDtype::BF16。 |

## 返回值说明

无

## 约束说明

无

## 调用示例

```
auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
ConvBackpropApi::Conv3DBpInputTiling conv3DBpDxTiling(*ascendcPlatform);
conv3DBpDxTiling.SetWeightType(
    ConvCommonApi::TPosition::GM, ConvCommonApi::ConvFormat::FRACTAL_Z_3D, ConvCommonApi::ConvDtype::FLOAT16);
```
