---
type: API
title: asc_set_gm2ub_pad（废弃）
description: 头文件路径为："c_api/vector_datamove/vector_datamove.h"。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/deprecated_interface/asc_set_gm2ub_pad_deprecated.md
  role: primary
created_at: '2026-09-12T03:09:26Z'
updated_at: '2026-09-12T03:09:26Z'
platforms:
- '950'
source_headers:
- include/c_api/vector_datamove/vector_datamove.h
---

# asc_set_gm2ub_pad（废弃）

## 功能说明

头文件路径为：`"c_api/vector_datamove/vector_datamove.h"`。

**此接口已废弃，源数据从GM非对齐搬运到Unified Buffer（UB）设置填补数据值请使用[asc_set_copy_pad_val](../vector_datamove/asc_set_copy_pad_val.md)进行设置。**

将数据从Global Memory (GM) 非对齐搬运到 UB时，通过调用该接口设置连续搬运数据块左右两侧需要填补的数据值。

## 函数原型 

```cpp 
__aicore__ inline void asc_set_gm2ub_pad(uint32_t pad_val)
``` 

## 参数说明 

**表1** 参数说明

|参数名|输入/输出|描述| 
|------------|------------|-----------| 
| pad_val    | 输入       | 左右两侧需要填补的数据值。| 

## 返回值说明 

无 

## 流水类型 

PIPE_S 

## 约束说明 

无 

## 调用示例 

```cpp 
asc_set_gm2ub_loop_size(2, 2); 
asc_set_gm2ub_loop1_stride(96, 128); 
asc_set_gm2ub_loop2_stride(192, 288); 
// 源操作数非对齐，需要填补数据 
asc_set_gm2ub_pad(0); 
asc_copy_gm2ub_align(dst, src, 2, 46 * sizeof(int8_t), 1, 1, true, asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM, 48, 48); 
asc_set_gm2ub_loop_size(1, 1); 
```
