---
type: API
title: asc_pair_reduce_sum_sync（废弃）
description: 头文件路径为："c_api/composite/vector_compute_composite.h"。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/deprecated_interface/asc_pair_reduce_sum_sync_deprecated.md
  role: primary
created_at: '2026-09-12T03:09:26Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- a2
- a3
source_headers:
- include/c_api/composite/vector_compute_composite.h
---

# asc_pair_reduce_sum_sync（废弃）

## 功能说明

头文件路径为：`"c_api/composite/vector_compute_composite.h"`。

**`asc_pair_reduce_sum_sync`接口已废弃，请使用[对应的非同步接口](../vector_compute/vector_reduce/asc_pair_reduce_sum.md)和同步接口[asc_sync](../sync/intra_core_sync/asc_sync.md)替代。**

对输入数据做归约操作，得到数据总和。相邻两个（奇偶）元素求和，例如（a1，a2，a3，a4，a5，a6…），相邻两个数据求和为（a1+a2，a3+a4，a5+a6…）。

## 函数原型

- 同步计算

    ```cpp
    __aicore__ inline void asc_pair_reduce_sum_sync(__ubuf__ half* dst, __ubuf__ half* src, uint32_t count)
    __aicore__ inline void asc_pair_reduce_sum_sync(__ubuf__ float* dst, __ubuf__ float* src, uint32_t count)
    ```

## 参数说明

**表1** 参数说明

|参数名|输入/输出|描述|
|------------|------------|-----------|
| dst     | 输出     | 目的操作数（矢量）的起始地址。 |
| src     | 输入     | 源操作数（矢量）的起始地址。 |
| count   | 输入     | 参与连续计算的元素个数。 |

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

- 操作数地址重叠约束请参考[通用地址重叠约束](../general_description_and_constraints.md#通用地址重叠约束)。
- dst、src的起始地址需要32字节对齐。

## 调用示例

```cpp
// 示例计算256个half的相邻元素求和，分2次迭代完成，每次迭代处理128个元素，输出64个结果
constexpr uint32_t src_length = 256;
constexpr uint32_t dst_length = 128;
__ubuf__ half src[src_length];
__ubuf__ half dst[dst_length];

// 同步接口使用 count 指定参与计算的元素总数，无需手动设置 mask。
asc_pair_reduce_sum_sync(dst, src, src_length);
```

结果示例：

```
输入数据src：[1 2 3 ... 16 17 ... 32 ... 225... 256]
输出数据dst：[3 7 ... 511]
```
