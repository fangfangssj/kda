---
type: API
title: SetInputStartPosition
description: 设置单核上特征矩阵Input载入数据的起始位置。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.compute.neighborhood.convolution
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/convolution_compute/Conv3D_Kernel/SetInputStartPosition.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- a2
- a3
source_headers:
- include/adv_api/conv/conv3d/conv3d_intf.h
---

# SetInputStartPosition

## 功能说明

设置单核上特征矩阵Input载入数据的起始位置。

## 函数原型

```
__aicore__ inline void SetInputStartPosition(int64_t diStartPos, int64_t mStartPos)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| diStartPos | 输入 | 单核上Din方向起始位置。 |
| mStartPos | 输入 | 单核上M方向起始位置。 |

## 返回值说明

无

## 约束说明

无

## 调用示例

```
conv3dApi.SetInputStartPosition(diIdxStart, mIdxStart);
```

