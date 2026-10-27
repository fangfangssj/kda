---
type: API
title: asc_brcb_sync（废弃）
description: 头文件路径为："c_api/composite/vector_compute_composite.h"。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/deprecated_interface/asc_brcb_sync_deprecated.md
  role: primary
created_at: '2026-09-12T03:09:26Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- a2
- a3
source_headers:
- include/c_api/composite/vector_compute_composite.h
---

# asc_brcb_sync（废弃）

## 功能说明

头文件路径为：`"c_api/composite/vector_compute_composite.h"`。

**`asc_brcb_sync`接口已废弃，请使用[对应的非同步接口](../vector_compute/vector_broadcast/asc_brcb.md)和同步接口[asc_sync](../sync/intra_core_sync/asc_sync.md)替代。**

给定一个输入矢量，每一次取输入矢量中的8个数填充到结果矢量的8个datablock（32Bytes）中去，每个数对应一个datablock。

## 函数原型

- 同步高维切分计算

    ```cpp
    __aicore__ inline void asc_brcb_sync(__ubuf__ uint16_t* dst, __ubuf__ uint16_t* src, uint16_t dst_block_stride, uint16_t dst_repeat_stride, uint8_t repeat)
    __aicore__ inline void asc_brcb_sync(__ubuf__ uint32_t* dst, __ubuf__ uint32_t* src, uint16_t dst_block_stride, uint16_t dst_repeat_stride, uint8_t repeat)
    ```

## 参数说明

**表1** 参数说明

| 参数名 | 输入/输出 | 描述 |
| :--- | :--- | :--- |
| dst | 输出 | 目的操作数（矢量）的起始地址。 |
| src | 输入 | 源操作数（矢量）的起始地址。 |
| dst_block_stride | 输入 | 目的操作数单次迭代内不同DataBlock间地址步长。 |
| dst_repeat_stride | 输入 | 目的操作数相邻迭代间相同DataBlock的地址步长。 |
| repeat | 输入 | 迭代次数。 |

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

- 不支持src与dst为同一块内存地址。

## 调用示例

```cpp
constexpr uint32_t src_length = 16;
constexpr uint32_t dst_length = 256;
__ubuf__ uint16_t src[src_length];
__ubuf__ uint16_t dst[dst_length];
asc_brcb_sync(dst, src, 1, 8, 2);
```

结果示例：

```
输入数据src：[1 2 3 ... 16]
输出数据dst：[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 ... 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16]
```
