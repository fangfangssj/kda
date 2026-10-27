---
type: API
title: asc_get_program_counter
description: 获取程序计数器的指针，程序计数器用于记录当前程序执行的位置。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/utils/sys_var/asc_get_program_counter.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-09-12T03:09:26Z'
platforms:
- '950'
- a2
- a3
source_headers:
- include/c_api/utils/sys_var.h
---

# asc_get_program_counter

## 功能说明

头文件路径为：`"c_api/utils/sys_var.h"`。

获取程序计数器的指针，程序计数器用于记录当前程序执行的位置。常用于调试时定位当前执行位置，也可配合[asc_icache_preload](../../cache_ctrl/asc_icache_preload.md)等接口，将当前指令地址作为预取的起始地址。

## 函数原型

```c
__aicore__ inline int64_t asc_get_program_counter()
```

## 参数说明

无

## 返回值说明

返回int64_t类型的程序计数器指针。

## 流水类型

PIPE_S

## 约束说明

本接口返回64位程序计数器值，其中有效地址位为0~47位，高16位为无效位。如下代码片段所示，使用时需通过掩码`0xFFFFFFFFFFFF`（低48位全1）将高位无效位清零，以得到当前指令在GM中的有效48位地址。

```cpp
int64_t prefetch_length = 6;
// 通过asc_get_program_counter获取当前程序计数器（PC）值。
// PC为64位返回值，有效地址位为0~47位，通过掩码0xFFFFFFFFFFFF（低48位全1）将高位无效位清零，
// 得到当前指令在GM中的有效48位地址，作为预加载的起始地址。
int64_t pc = asc_get_program_counter() & 0xFFFFFFFFFFFF;
asc_icache_preload(reinterpret_cast<void *>(pc), prefetch_length);
```

## 调用示例

获取PC后，可将其作为指令预加载起始地址，完整的预加载调用方式请参考[asc_icache_preload调用示例](../../cache_ctrl/asc_icache_preload.md#调用示例)。
