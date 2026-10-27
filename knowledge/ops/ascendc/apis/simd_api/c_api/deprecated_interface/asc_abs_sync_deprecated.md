---
type: API
title: AscAbsSync
description: 已废弃的按元素取绝对值同步计算接口，请改用asc_abs和asc_sync实现精细化调优。
tags:
- task.lookup
- topic.quality.compatibility.deprecation
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/deprecated_interface/asc_abs_sync_deprecated.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- a2
- a3
source_headers:
- include/c_api/composite/vector_compute_composite.h
---

# asc_abs_sync（废弃）

## 功能说明

头文件路径为：`"c_api/composite/vector_compute_composite.h"`。

该接口废弃，请通过[asc_abs](../vector_compute/vector_arith/asc_abs.md)和同步接口[asc_sync](../sync/intra_core_sync/asc_sync.md)实现精细化调优，达成最优性能。

按元素取绝对值，计算公式如下：
$$
dst_i = |src_i|
$$

## 函数原型

- 同步计算

    ```cpp
    __aicore__ inline void asc_abs_sync(__ubuf__ half* dst, __ubuf__ half* src, uint32_t count)
    __aicore__ inline void asc_abs_sync(__ubuf__ float* dst, __ubuf__ float* src, uint32_t count)
    ```

## 参数说明

**表1** 参数说明

| 参数名  | 输入/输出 | 描述 |
| :----- | :------- | :------- |
| dst | 输出 | 目的操作数（矢量）的起始地址。 |
| src | 输入 | 源操作数（矢量）的起始地址。 |
| count | 输入 | 参与计算的元素个数。 |

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

- dst、src的起始地址需要32字节对齐。
- 操作数地址重叠约束请参考[通用地址重叠约束](../general_description_and_constraints.md#通用地址重叠约束)。

## 调用示例

```cpp
//total_length指参与计算的数据长度
constexpr uint32_t total_length = 128;
__ubuf__ half src[total_length];
__ubuf__ half dst[total_length];
asc_abs_sync(dst, src, total_length);
```
