---
type: API
title: Conv3DBackpropFilterTiling
description: 提供获取Conv3DBackpropFilter Kernel计算所需Tiling参数的API，用户传入输入输出参数后调用GetTiling接口获取Tiling信息。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.compute.neighborhood.convolution
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/convolution_compute/Conv3DBackpropFilter_Tiling/Conv3DBackpropFilter_Tiling_usage.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-04T01:25:11Z'
platforms: []
---

# Conv3DBackpropFilter Tiling使用说明

Ascend C提供一组Conv3DBackpropFilter Tiling API，方便用户获取Conv3DBackpropFilter核函数（Kernel）计算时所需的Tiling参数。用户只需要传入Input/GradOutput/GradWeight的Position位置、Format格式和DType数据类型及相关参数等信息，调用API接口，即可获取[Init](../conv3_d_backprop_filter_kernel/init.md)中TConv3DBpFilterTiling结构体中的相关参数。

Conv3DBackpropFilter Tiling API提供一个GetTiling接口获取Tiling参数，获取Tiling参数的流程如下：

1.  创建一个单核Tiling对象。
2.  设置Input、GradOutput、GradWeight的参数类型信息以及Shape信息，如果存在Padding、Stride参数，通过[SetPadding](set_padding.md)、[SetStride](set_stride.md)接口设置。
3.  调用[GetTiling](get_tiling.md)接口，获取Tiling信息。

使用Conv3DBackpropFilter Tiling接口获取Tiling参数的样例如下：

```
#include "tiling/conv_backprop/conv3d_bp_filter_tiling.h"

optiling::Conv3DBackpropFilterTilingData tilingData;
auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
ConvBackpropApi::Conv3dBpFilterTiling conv3dBpDwTiling(*ascendcPlatform);

conv3dBpDwTiling.SetWeightType(
    ConvCommonApi::TPosition::CO1, ConvCommonApi::ConvFormat::FRACTAL_Z_3D, ConvCommonApi::ConvDtype::FLOAT32);
conv3dBpDwTiling.SetInputType(
    ConvCommonApi::TPosition::GM, ConvCommonApi::ConvFormat::NDC1HWC0, ConvCommonApi::ConvDtype::FLOAT16);
conv3dBpDwTiling.SetGradOutputType(
    ConvCommonApi::TPosition::GM, ConvCommonApi::ConvFormat::NDC1HWC0, ConvCommonApi::ConvDtype::FLOAT16);
conv3dBpDwTiling.SetGradOutputShape(n, c, d, h, w);
conv3dBpDwTiling.SetInputShape(n, c, d, h, w);
conv3dBpDwTiling.SetWeightShape(cout, cin, d, h, w);
conv3dBpDwTiling.SetPadding(padFront, padBack, padUp, padDown, padLeft, padRight);
conv3dBpDwTiling.SetStride(strideD, strideH, strideW);
conv3dBpDwTiling.SetDilation(dilationD, dilationH, dilationW);
int ret = conv3dBpDwTiling.GetTiling(tilingData); // 如果ret = -1,获取tiling结果失败
```

## 需要包含的头文件

```
#include "lib/conv_backprop/conv3d_bp_filter_tiling.h"
```
