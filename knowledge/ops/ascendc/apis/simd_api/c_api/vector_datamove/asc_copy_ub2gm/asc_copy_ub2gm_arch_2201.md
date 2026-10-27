---
type: API
title: asc_copy_ub2gm
description: 将数据从Unified Buffer（UB）搬运到Global Memory (GM)。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/vector_datamove/asc_copy_ub2gm/asc_copy_ub2gm_arch_2201.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- a2
- a3
source_headers:
- include/c_api/vector_datamove/vector_datamove.h
---

# asc_copy_ub2gm

## 功能说明

头文件路径为：`"c_api/vector_datamove/vector_datamove.h"`。

将数据从Unified Buffer（UB）搬运到Global Memory (GM)。

## 函数原型

- 前n个数据搬运

```cpp
__aicore__ inline void asc_copy_ub2gm(__gm__ void* dst, __ubuf__ void* src, uint32_t size)
```

- 高维切分搬运

```cpp
__aicore__ inline void asc_copy_ub2gm(__gm__ void* dst, __ubuf__ void* src, uint16_t burst_count, uint16_t burst_len, uint16_t src_gap, uint16_t dst_gap)
```

## 参数说明

**表1** 参数说明

| 参数名 | 输入/输出 | 描述 |
| :--- | :--- | :--- |
| dst | 输出 | 目的GM地址。 |
| src | 输入 | 源UB地址。 |
| size | 输入 | 搬运数据大小（字节）。 |
| burst_count | 输入 | 待搬运的连续传输数据块个数。取值范围：[1, 4095]。 |
| burst_len | 输入 | 待搬运的每个连续传输数据块的长度，单位为DataBlock（32字节）。取值范围：[1, 65535]。 |
| src_gap | 输入 | 源操作数相邻连续数据块的间隔（前面一个数据块的尾与后面一个数据块的头的间隔）。<br>单位为DataBlock（32字节）。 |
| dst_gap | 输入 | 目的操作数相邻连续数据块的间隔（前面一个数据块的尾与后面一个数据块的头的间隔）。<br>单位为DataBlock（32字节）。 |

## 返回值说明

无

## 流水类型

PIPE_MTE3

## 约束说明

- 各存储单元的空间大小和对齐要求请参考[存储单元说明](../../general_description_and_constraints.md#存储单元说明)。
- 如果本指令与其他指令存在数据依赖，需要插入[同步指令](../../sync/system_sync_overview.md)，保证多个指令串行化，防止出现异常数据。
- 当采用前n个数据搬运接口时，搬运数据大小要求32字节对齐。

## 调用示例

```cpp
// total_length指参与搬运的数据总长度。dst是外部输入的half类型的GM内存。
constexpr uint64_t total_length = 128;
__ubuf__ half src[total_length];
asc_copy_ub2gm(dst, src, total_length * sizeof(half));
```

## 配套 API 与常见组合

- [asc_copy_gm2ub](../asc_copy_gm2ub/asc_copy_gm2ub_arch_2201.md)——同一算子GM↔UB数据通路的镜像两端（入搬运-计算-出搬运），P(a|b)=1.00，证据中同批UB缓冲区首尾相接
