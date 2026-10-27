---
type: API
title: 卷积计算Init
description: Init主要用于对Conv3DBackpropFilter对象中的Tiling数据进行初始化，根据Tiling参数进行资源划分，Tiling参数的具体介绍请参考Conv3DBackpropFilter
  Tiling侧接口。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.compute.neighborhood.convolution
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/convolution_compute/Conv3DBackpropFilter_Kernel/Init.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- a2
- a3
---

# Init

## 功能说明

Init主要用于对Conv3DBackpropFilter对象中的Tiling数据进行初始化，根据Tiling参数进行资源划分，Tiling参数的具体介绍请参考[Conv3DBackpropFilter Tiling侧接口](../conv3_d_backprop_filter_tiling/index.md)。

## 函数原型

```
__aicore__ inline void Init(const TConv3DBpFilterTiling* __restrict tiling)
```

## 参数说明

**表1**  接口参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| tiling | 输入 | Conv3DBackpropFilter对象的Tiling参数，Conv3DBackpropFilterTilingData结构体定义请参见[TConv3DBpFilterTiling结构体](../conv3_d_backprop_filter_tiling/t_conv3_d_bp_filter_tiling_struct.md)。<br><br>Tiling参数可以通过Host侧[GetTiling](../conv3_d_backprop_filter_tiling/get_tiling.md)接口获取，并传递到核函数（Kernel）侧使用。 |

## 返回值说明

无

## 约束说明

Init接口必须在Iterate、GetTensorC、End接口前调用，且只能调用一次Init接口，调用顺序如下。

```
Init(...);
...
Iterate(...);
GetTensorC(...);
End();
```

## 调用示例

```
const Conv3DBackpropFilterTilingData* tilingData;
// ...初始化tilingData，创建Conv3DBackpropFilter对象，调用init接口
ConvBackpropApi::Conv3DBackpropFilter <inputType, weightSizeType, gradOutputType, gradWeightType > gradWeight_;
gradWeight_.Init(&(tilingData->dwTiling));
```
