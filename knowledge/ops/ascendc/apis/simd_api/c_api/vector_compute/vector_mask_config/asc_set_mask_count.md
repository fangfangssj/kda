---
type: API
title: asc_set_mask_count
description: 设置Mask模式为Counter模式。该模式下，不需要开发者去感知迭代次数、处理非对齐的尾块等操作，可直接传入数据计算数量，实际迭代次数由Vector计算单元自动推断。在前n个数据计算类的接口内部会自动设置该模式，不需要自行设置。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/vector_compute/vector_mask_config/asc_set_mask_count.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-09-12T03:09:26Z'
platforms:
- a2
- a3
source_headers:
- include/c_api/vector_compute/compute/vector_mask_config.h
---

# asc_set_mask_count

## 功能说明

头文件路径为：`"c_api/vector_compute/compute/vector_mask_config.h"`。

设置Mask模式为Counter模式。该模式下，不需要开发者去感知迭代次数、处理非对齐的尾块等操作，可直接传入数据计算数量，实际迭代次数由Vector计算单元自动推断。在前n个数据计算类的接口内部会自动设置该模式，不需要自行设置。

## 函数原型

```cpp
__aicore__ inline void asc_set_mask_count()
```

## 参数说明

无

## 返回值说明

无

## 流水类型

PIPE_S

## 约束说明

该API需和[asc_set_vector_mask](asc_set_vector_mask.md)配合使用。

## 调用示例

```cpp
asc_set_mask_count();
asc_set_vector_mask(0, static_cast<uint64_t>(64)); // 设置前64个元素参与计算
... // 计算操作
asc_set_mask_norm();    // 恢复为Normal模式。此步骤可根据需求，按需设置。
```