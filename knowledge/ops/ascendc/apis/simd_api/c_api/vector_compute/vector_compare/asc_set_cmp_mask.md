---
type: API
title: asc_set_cmp_mask
description: 为asc_select操作设置作用于选择的Mask掩码。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/vector_compute/vector_compare/asc_set_cmp_mask.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-09-12T03:09:26Z'
platforms:
- a2
- a3
source_headers:
- include/c_api/vector_compute/compute/vector_compare.h
---

# asc_set_cmp_mask

## 功能说明

头文件路径为：`"c_api/vector_compute/compute/vector_compare.h"`。

为[asc_select](../vector_permute_sel/asc_select.md)操作设置作用于选择的Mask掩码。

## 函数原型

```cpp
__aicore__ inline void asc_set_cmp_mask(__ubuf__ void* sel_mask)
```

## 参数说明

**表1** 参数说明

|参数名|输入/输出|描述|
| ------------ | ------------ | ------------ |
|sel_mask|输入|用于选择的Mask掩码的起始地址。|

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

需和[asc_select](../vector_permute_sel/asc_select.md)操作配合使用。

## 调用示例

```cpp
constexpr uint32_t total_length = 256;
constexpr uint32_t sel_length = 8;
__ubuf__ uint8_t sel_mask[sel_length];
__ubuf__ float dst[total_length];
__ubuf__ float src0[total_length];
__ubuf__ float src1[total_length];
asc_set_cmp_mask(sel_mask);
asc_sync_pipe(PIPE_V);
asc_select(dst, src0, src1, total_length);
```
