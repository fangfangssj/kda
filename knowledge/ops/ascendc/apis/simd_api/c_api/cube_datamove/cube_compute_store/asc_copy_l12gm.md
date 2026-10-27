---
type: API
title: asc_copy_l12gm
description: 头文件路径为："c_api/cube_datamove/cube_datamove.h"。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/cube_datamove/cube_compute_store/asc_copy_l12gm.md
  role: primary
created_at: '2026-10-20T07:46:54Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- a2
- a3
source_headers:
- include/c_api/cube_datamove/cube_datamove.h
---

# asc_copy_l12gm

## 功能说明

头文件路径为：`"c_api/cube_datamove/cube_datamove.h"`。

将数据从L1 Buffer搬运到Global Memory。

## 函数原型

- 常规搬运

    ```cpp
    __aicore__ inline void asc_copy_l12gm(__gm__ void* dst, __cbuf__ void* src, uint16_t n_burst, uint16_t len_burst, uint16_t src_gap, uint16_t dst_gap)
    ```

## 参数说明

**表1** 参数说明

| 参数名 | 输入/输出 | 描述                                                              |
| :--- |:------|:----------------------------------------------------------------|
| dst | 输出    | 目的操作数的起始地址。|
| src | 输入    | 源操作数的起始地址。|
| n_burst | 输入    | 待搬运的连续传输数据块个数。取值范围：[1, 4095]。 |
| len_burst | 输入    | 待搬运的每个连续传输数据块的长度，单位为32个字节。取值范围：[1, 65535]。 |
| src_gap | 输入    | 相邻迭代间，源操作数前一个迭代第一个分形的结束地址到下一个迭代第一个分形起始地址的间隔。 |
| dst_gap | 输入    | 相邻迭代间，目的操作数前一个迭代第一个分形的结束地址到下一个迭代第一个分形起始地址的间隔。单位为32字节。 |

## 返回值说明

无

## 流水类型

PIPE_MTE3

## 约束说明

- 各存储单元的空间大小和对齐要求请参考[存储单元说明](../../general_description_and_constraints.md#存储单元说明)。
- 如果本指令与其他指令存在数据依赖，需要插入[同步指令](../../sync/system_sync_overview.md)，保证多个指令串行化，防止出现异常数据。

## 调用示例

```cpp
//待搬运的连续传输数据块为1
constexpr uint16_t n_burst = 1;
//待搬运的每个连续传输数据块的长度为32个字节
constexpr uint16_t len_burst = 1;
//相邻迭代间，源操作数前一个分形与后一个分形起始地址的间隔为32个字节
constexpr uint16_t src_gap = 0;
//相邻迭代间，目标操作数前一个分形与后一个分形起始地址的间隔为64个字节
constexpr uint16_t dst_gap = 1;
__cbuf__ half src[256];
// dst表示目的操作数的起始地址
asc_copy_l12gm(dst, src, n_burst, len_burst, src_gap, dst_gap);
```
