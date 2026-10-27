---
type: API
title: asc_get_cmp_mask
description: 此接口用于获取Compare操作的比较结果。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/vector_compute/vector_compare/asc_get_cmp_mask.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-09-12T03:09:26Z'
platforms:
- a2
- a3
source_headers:
- include/c_api/vector_compute/compute/vector_compare.h
---

# asc_get_cmp_mask

## 功能说明

头文件路径为：`"c_api/vector_compute/compute/vector_compare.h"`。

此接口用于获取Compare操作的比较结果。

## 函数原型

```cpp
__aicore__ inline void asc_get_cmp_mask(__ubuf__ void* dst)
```

## 参数说明

**表1** 参数说明

|参数名|输入/输出|描述|
| ------------ | ------------ | ------------ |
|dst|输出|存放比较操作结果的地址。|

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

需和Compare操作配合使用。

## 调用示例

```cpp
constexpr uint32_t total_length = 128;
__ubuf__ int8_t dst[total_length];
...     // 进行Compare操作
asc_get_cmp_mask(dst);
```