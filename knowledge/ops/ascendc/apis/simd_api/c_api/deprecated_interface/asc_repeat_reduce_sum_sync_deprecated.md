---
type: API
title: asc_repeat_reduce_sum_sync（废弃）
description: 头文件路径为："c_api/composite/vector_compute_composite.h"。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/deprecated_interface/asc_repeat_reduce_sum_sync_deprecated.md
  role: primary
created_at: '2026-09-12T03:09:26Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- a2
- a3
source_headers:
- include/c_api/composite/vector_compute_composite.h
---

# asc_repeat_reduce_sum_sync（废弃）

## 功能说明

头文件路径为：`"c_api/composite/vector_compute_composite.h"`。

**`asc_repeat_reduce_sum_sync`接口已废弃，请使用[对应的非同步接口](../vector_compute/vector_reduce/asc_repeat_reduce_sum.md)和同步接口[asc_sync](../sync/intra_core_sync/asc_sync.md)替代。**

对每个Repeat内所有元素求和。

源操作数相加采用二叉树的方式，两两相加。以256个half类型数据求和为例，每个Repeat可以计算128个half类型数据，分成2个Repeat计算；每个Repeat内，通过二叉树的方式，两两相加。

需要注意的是，两两相加计算过程中，计算结果大于65504时结果保存为65504。例如，源操作数为[60000,60000,-30000,100]，首先60000+60000溢出，结果为65504，然后计算-30000+100=-29900，最后计算65504-29900=35604。

## 函数原型

- 同步计算

    ```cpp
    __aicore__ inline void asc_repeat_reduce_sum_sync(__ubuf__ half* dst, __ubuf__ half* src, uint32_t count)
    __aicore__ inline void asc_repeat_reduce_sum_sync(__ubuf__ float* dst, __ubuf__ float* src, uint32_t count)
    ```

## 参数说明

**表1** 参数说明

| 参数名 | 输入/输出 | 描述 |
|----|-----|-----|
| dst | 输出 | 目的操作数（矢量）的起始地址。 |
| src  | 输入 | 源操作数（矢量）的起始地址。|
| count | 输入 | 参与计算的元素个数。 |

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

- 操作数地址重叠约束请参考[通用地址重叠约束](../general_description_and_constraints.md#通用地址重叠约束)。
- dst的起始地址，输入类型为half时需要2字节对齐，输入类型为float时需要4字节对齐。
- src的起始地址需要32字节对齐。

## 调用示例

```cpp
// 示例计算256个half的和，分2次迭代完成，计算结果为2组求和的值
constexpr uint32_t total_length = 256;
__ubuf__ half src[total_length];
constexpr uint32_t dst_length = 2;
__ubuf__ half dst[dst_length];

// 同步接口使用 count 指定参与计算的元素总数，无需手动设置 mask。
asc_repeat_reduce_sum_sync(dst, src, total_length);
```

结果示例：

```
输入数据src：[1, 1, 1, ..., 1]
输出数据dst：[128, 128]
```
