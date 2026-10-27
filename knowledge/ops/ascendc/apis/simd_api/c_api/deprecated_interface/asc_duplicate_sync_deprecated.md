---
type: API
title: asc_duplicate_sync（废弃）
description: 头文件路径为："c_api/composite/vector_compute_composite.h"。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/deprecated_interface/asc_duplicate_sync_deprecated.md
  role: primary
created_at: '2026-09-12T03:09:26Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- a2
- a3
source_headers:
- include/c_api/composite/vector_compute_composite.h
---

# asc_duplicate_sync（废弃）

## 功能说明

头文件路径为：`"c_api/composite/vector_compute_composite.h"`。

**`asc_duplicate_sync`接口已废弃，请使用[对应的非同步接口](../vector_compute/vector_broadcast/asc_duplicate.md)和同步接口[asc_sync](../sync/intra_core_sync/asc_sync.md)替代。**

执行矢量复制（Duplicate）操作，将标量值复制填充到矢量中。

## 函数原型

- 同步计算

    ```cpp
    __aicore__ inline void asc_duplicate_sync(__ubuf__ half* dst, half src, uint32_t count)
    __aicore__ inline void asc_duplicate_sync(__ubuf__ int16_t* dst, int16_t src, uint32_t count)
    __aicore__ inline void asc_duplicate_sync(__ubuf__ uint16_t* dst, uint16_t src, uint32_t count)
    __aicore__ inline void asc_duplicate_sync(__ubuf__ bfloat16_t* dst, bfloat16_t src, uint32_t count)
    __aicore__ inline void asc_duplicate_sync(__ubuf__ float* dst, float src, uint32_t count)
    __aicore__ inline void asc_duplicate_sync(__ubuf__ int32_t* dst, int32_t src, uint32_t count)
    __aicore__ inline void asc_duplicate_sync(__ubuf__ uint32_t* dst, uint32_t src, uint32_t count)
    ```

## 参数说明

**表1** 参数说明

| 参数名 | 输入/输出 | 描述 |
| :--- | :--- | :--- |
| dst | 输出 | 目的操作数（矢量）的起始地址。 |
| src | 输入 | 源操作数（标量）。 |
| count | 输入 | 参与连续复制的元素个数。 |

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

- dst的起始地址需要32字节对齐。

## 调用示例

```cpp
constexpr uint32_t dst_length = 128;
__ubuf__ uint16_t dst[dst_length];
uint16_t val = 18;
asc_duplicate_sync(dst, val, dst_length);
```

结果示例：

```
输出数据dst：[18 18 18 ... 18]
```
