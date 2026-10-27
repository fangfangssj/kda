---
type: API
title: SetSingleShape
description: 设置Conv3DBackpropFilter在单核上计算的形状。用于运行时修改shape。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.compute.neighborhood.convolution
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/convolution_compute/Conv3DBackpropFilter_Kernel/SetSingleShape.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- a2
- a3
---

# SetSingleShape

## 功能说明

设置Conv3DBackpropFilter在单核上计算的形状。用于运行时修改shape。

## 函数原型

```
__aicore__ inline void SetSingleShape(uint64_t singleCoreM, uint64_t singleCoreN, uint64_t singleCoreK)
```

## 参数说明

**表1**  接口参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| singleCoreM | 输入 | 单核上M的大小，单位为元素个数。 |
| singleCoreN | 输入 | 单核上N的大小，单位为元素个数。 |
| singleCoreK | 输入 | 单核上K的大小，单位为元素个数。 |

## 返回值说明

无

## 约束说明

无

## 调用示例

```
const Conv3DBackpropFilterTilingData* tilingData;
// ...初始化tilingData
ConvBackpropApi::Conv3DBackpropFilter <inputType, weightSizeType, gradOutputType, gradWeightType > gradWeight_;
....
gradWeight_.Init(&(tilingData->dwTiling));
gradWeight_.SetInput(inputGm_[offsetB_]);
gradWeight_.SetGradOutput(gradOutputGm_[offsetA_]);
gradWeight_.SetSingleShape(singleCoreM, singleCoreN, singleCoreK);// 设置单核计算量
....
```
