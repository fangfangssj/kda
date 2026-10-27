---
type: API
title: SetStartPosition
description: 设置单核上GradOutput载入数据的起始位置。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.compute.neighborhood.convolution
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/convolution_compute/Conv3DBackpropInput_Kernel/SetStartPosition.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- a2
- a3
---

# SetStartPosition

## 功能说明

设置单核上GradOutput载入数据的起始位置。

## 函数原型

```
__aicore__ inline void SetStartPosition(uint32_t curDinStartIdx, int32_t curHoStartIdx)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| curDinStartIdx | 输入 | 当前核D方向起始位置。 |
| curHoStartIdx | 输入 | 当前核H方向起始位置。 |

## 返回值说明

无

## 约束说明

无

## 调用示例

```
gradInput_.SetStartPosition(dinStartIdx_, curHoStartIdx_); // 设置单核上GradOutput载入的起始位置
```

