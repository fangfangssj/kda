---
type: API
title: asc_copy_l12l0c_sync（废弃）
description: 头文件路径为："c_api/composite/cube_datamove_composite.h"。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/deprecated_interface/asc_copy_l12l0c_sync_deprecated.md
  role: primary
created_at: '2026-09-12T03:09:26Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- a2
- a3
source_headers:
- include/c_api/composite/cube_datamove_composite.h
---

# asc_copy_l12l0c_sync（废弃）

## 功能说明

头文件路径为：`"c_api/composite/cube_datamove_composite.h"`。

**`asc_copy_l12l0c_sync`接口已废弃，请使用[对应的非同步接口](../cube_datamove/cube_compute_load/asc_copy_l12l0c.md)和同步接口[asc_sync](../sync/intra_core_sync/asc_sync.md)替代。**

将矩阵由L1 Buffer搬运到L0C Buffer中。该搬运支持的数据类型如下：
- bfloat16_t->bfloat16_t
- half->half
- float->half
- float->bfloat16_t
- float->float
- int32_t->int32_t
- uint32_t->uint32_t

## 函数原型

- 同步搬运

    ```cpp
    __aicore__ inline void asc_copy_l12l0c_sync(__cc__ bfloat16_t* dst, __cbuf__ bfloat16_t* src, uint16_t n_burst, uint16_t len_burst, uint16_t src_gap, uint16_t dst_gap)
    __aicore__ inline void asc_copy_l12l0c_sync(__cc__ half* dst, __cbuf__ half* src, uint16_t n_burst, uint16_t len_burst, uint16_t src_gap, uint16_t dst_gap)
    __aicore__ inline void asc_copy_l12l0c_sync(__cc__ half* dst, __cbuf__ float* src, uint16_t n_burst, uint16_t len_burst, uint16_t src_gap, uint16_t dst_gap)
    __aicore__ inline void asc_copy_l12l0c_sync(__cc__ bfloat16_t* dst, __cbuf__ float* src, uint16_t n_burst, uint16_t len_burst, uint16_t src_gap, uint16_t dst_gap)
    __aicore__ inline void asc_copy_l12l0c_sync(__cc__ float* dst, __cbuf__ float* src, uint16_t n_burst, uint16_t len_burst, uint16_t src_gap, uint16_t dst_gap)
    __aicore__ inline void asc_copy_l12l0c_sync(__cc__ int32_t* dst, __cbuf__ int32_t* src, uint16_t n_burst, uint16_t len_burst, uint16_t src_gap, uint16_t dst_gap)
    __aicore__ inline void asc_copy_l12l0c_sync(__cc__ uint32_t* dst, __cbuf__ uint32_t* src, uint16_t n_burst, uint16_t len_burst, uint16_t src_gap, uint16_t dst_gap)
    ```

## 参数说明

**表1** 参数说明

| 参数名 | 输入/输出 | 描述                                                              |
| :--- |:------|:----------------------------------------------------------------|
| dst | 输出    | 目的操作数的起始地址。|
| src | 输入    | 源操作数的起始地址。|
| n_burst | 输入    | 待搬运的连续传输数据块个数。取值范围：[1, 4095]。 |
| len_burst | 输入    | 待搬运的每个连续传输数据块的长度，单位为32个字节。取值范围：[1, 65535]。 |
| src_gap | 输入    | 表示相邻迭代间，源操作数前一个迭代第一个分形的结束地址到下一个迭代第一个分形起始地址的间隔。 |
| dst_gap | 输入    | 表示相邻迭代间，目的操作数前一个迭代第一个分形的结束地址到下一个迭代第一个分形起始地址的间隔。 |

## 返回值说明

无

## 流水类型

PIPE_MTE1

## 约束说明

- 各存储单元的空间大小和对齐要求请参考[存储单元说明](../general_description_and_constraints.md#存储单元说明)。
- 如果需要执行多条asc_copy_l12l0c指令，且asc_copy_l12l0c指令的目的地址存在重叠，需要插入同步指令，保证多个asc_copy_l12l0c指令的串行化，防止出现异常数据。

## 调用示例

```cpp
//待搬运的连续传输数据块为1个
constexpr uint16_t n_burst = 1;
//待搬运的每个连续传输数据块的长度为32个字节
constexpr uint16_t len_burst = 1;
//相邻迭代间，源操作数前一个分形与后一个分形起始地址的间隔为64个字节
constexpr uint16_t src_gap = 1;
//相邻迭代间，目的操作数前一个分形与后一个分形起始地址的间隔为32个字节
constexpr uint16_t dst_gap = 0;
__cbuf__ half src[256];
__cc__ half dst[256];
asc_copy_l12l0c_sync(dst, src, n_burst, len_burst, src_gap, dst_gap);
```
