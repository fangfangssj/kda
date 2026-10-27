---
type: API
title: GetOffsetC
description: 预留接口，为后续功能做预留。 获取本次计算时当前分片在整个C矩阵中的位置。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.programming_model.execution_unit.cube
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/cube_compute/Matmul_Kernel/GetOffsetC.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-04T01:25:11Z'
platforms: []
---

# GetOffsetC

## 功能说明

预留接口，为后续功能做预留。

获取本次计算时当前分片在整个C矩阵中的位置。

## 函数原型

```
__aicore__ inline MatrixOffset GetOffsetC()
```

## 参数说明

无

## 返回值说明

MatrixOffset结构体如下：

```
struct MatrixOffset {
    int32_t offset;
    int32_t row, col;
    int32_t height, width;
};
```

## 约束说明

无

