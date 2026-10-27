---
type: API
title: asc_copy_ub2gm_align
description: 将数据从Unified Buffer（UB）搬运到Global Memory (GM)，支持8位/16位/32位分块拷贝操作。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/vector_datamove/asc_copy_ub2gm_align/asc_copy_ub2gm_align_arch_2201.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- a2
- a3
source_headers:
- include/c_api/vector_datamove/vector_datamove.h
---

# asc_copy_ub2gm_align

## 功能说明

头文件路径为：`"c_api/vector_datamove/vector_datamove.h"`。

将数据从Unified Buffer（UB）搬运到Global Memory（GM），支持搬运长度不是32字节整数倍的数据。搬运过程中按32字节的整数倍进行补齐处理，写入GM时丢弃补齐数据，仅写入指定长度的有效数据。该接口支持8位/16位/32位数据类型搬运，搬运数据长度仍需满足对应数据类型的字节对齐要求；源、目的起始地址的对齐要求请参见[参数说明](#参数说明)。

本接口支持以下两种数据搬运方式，均不支持设置填充值：

- 前n个数据搬运

    若搬运数据长度非32字节对齐，搬出时会在读取UB数据时自动补充dummy假数据，将搬运数据补齐至32字节对齐。dummy假数据的值为数据块的第一个元素的值。写入GM时会丢弃补充的dummy假数据。

- 高维切分搬运

    高维切分搬运采用Normal模式。若`burst_len`指定的单个数据块长度非32字节对齐，则`burst_count`指定的每个数据块在读取UB数据时都会分别补充dummy假数据至32字节对齐。dummy假数据的值为数据块的第一个元素的值。写入GM时会丢弃补充的dummy假数据。

## 函数原型

- 前n个数据搬运

    ```cpp
    __aicore__ inline void asc_copy_ub2gm_align(__gm__ uint8_t* dst, __ubuf__ uint8_t* src, uint32_t size)
    __aicore__ inline void asc_copy_ub2gm_align(__gm__ int8_t* dst, __ubuf__ int8_t* src, uint32_t size)
    __aicore__ inline void asc_copy_ub2gm_align(__gm__ half* dst, __ubuf__ half* src, uint32_t size)
    __aicore__ inline void asc_copy_ub2gm_align(__gm__ uint16_t* dst, __ubuf__ uint16_t* src, uint32_t size)
    __aicore__ inline void asc_copy_ub2gm_align(__gm__ int16_t* dst, __ubuf__ int16_t* src, uint32_t size)
    __aicore__ inline void asc_copy_ub2gm_align(__gm__ bfloat16_t* dst, __ubuf__ bfloat16_t* src, uint32_t size)
    __aicore__ inline void asc_copy_ub2gm_align(__gm__ uint32_t* dst, __ubuf__ uint32_t* src, uint32_t size)
    __aicore__ inline void asc_copy_ub2gm_align(__gm__ float* dst, __ubuf__ float* src, uint32_t size)
    __aicore__ inline void asc_copy_ub2gm_align(__gm__ int32_t* dst, __ubuf__ int32_t* src, uint32_t size)
    __aicore__ inline void asc_copy_ub2gm_align(__gm__ double* dst, __ubuf__ double* src, uint32_t size)
    __aicore__ inline void asc_copy_ub2gm_align(__gm__ int64_t* dst, __ubuf__ int64_t* src, uint32_t size)
    __aicore__ inline void asc_copy_ub2gm_align(__gm__ uint64_t* dst, __ubuf__ uint64_t* src, uint32_t size)
    ```

- 高维切分搬运

    ```cpp
    __aicore__ inline void asc_copy_ub2gm_align(__gm__ uint8_t* dst, __ubuf__ uint8_t* src, uint16_t burst_count, uint32_t burst_len,uint8_t left_padding_num, uint8_t right_padding_num, uint32_t src_gap, uint32_t dst_gap)
    __aicore__ inline void asc_copy_ub2gm_align(__gm__ int8_t* dst, __ubuf__ int8_t* src, uint16_t burst_count, uint32_t burst_len,uint8_t left_padding_num, uint8_t right_padding_num, uint32_t src_gap, uint32_t dst_gap)
    __aicore__ inline void asc_copy_ub2gm_align(__gm__ half* dst, __ubuf__ half* src, uint16_t burst_count, uint32_t burst_len,uint8_t left_padding_num, uint8_t right_padding_num, uint32_t src_gap, uint32_t dst_gap)
    __aicore__ inline void asc_copy_ub2gm_align(__gm__ uint16_t* dst, __ubuf__ uint16_t* src, uint16_t burst_count, uint32_t burst_len,uint8_t left_padding_num, uint8_t right_padding_num, uint32_t src_gap, uint32_t dst_gap)
    __aicore__ inline void asc_copy_ub2gm_align(__gm__ int16_t* dst, __ubuf__ int16_t* src, uint16_t burst_count, uint32_t burst_len,uint8_t left_padding_num, uint8_t right_padding_num, uint32_t src_gap, uint32_t dst_gap)
    __aicore__ inline void asc_copy_ub2gm_align(__gm__ bfloat16_t* dst, __ubuf__ bfloat16_t* src, uint16_t burst_count, uint32_t burst_len,uint8_t left_padding_num, uint8_t right_padding_num, uint32_t src_gap, uint32_t dst_gap)
    __aicore__ inline void asc_copy_ub2gm_align(__gm__ uint32_t* dst, __ubuf__ uint32_t* src, uint16_t burst_count, uint32_t burst_len,uint8_t left_padding_num, uint8_t right_padding_num, uint32_t src_gap, uint32_t dst_gap)
    __aicore__ inline void asc_copy_ub2gm_align(__gm__ float* dst, __ubuf__ float* src, uint16_t burst_count, uint32_t burst_len,uint8_t left_padding_num, uint8_t right_padding_num, uint32_t src_gap, uint32_t dst_gap)
    __aicore__ inline void asc_copy_ub2gm_align(__gm__ int32_t* dst, __ubuf__ int32_t* src, uint16_t burst_count, uint32_t burst_len,uint8_t left_padding_num, uint8_t right_padding_num, uint32_t src_gap, uint32_t dst_gap)
    __aicore__ inline void asc_copy_ub2gm_align(__gm__ double* dst, __ubuf__ double* src, uint16_t burst_count, uint32_t burst_len,uint8_t left_padding_num, uint8_t right_padding_num, uint32_t src_gap, uint32_t dst_gap)
    __aicore__ inline void asc_copy_ub2gm_align(__gm__ int64_t* dst, __ubuf__ int64_t* src, uint16_t burst_count, uint32_t burst_len,uint8_t left_padding_num, uint8_t right_padding_num, uint32_t src_gap, uint32_t dst_gap)
    __aicore__ inline void asc_copy_ub2gm_align(__gm__ uint64_t* dst, __ubuf__ uint64_t* src, uint16_t burst_count, uint32_t burst_len,uint8_t left_padding_num, uint8_t right_padding_num, uint32_t src_gap, uint32_t dst_gap)
    ```

## 参数说明

**表1** 参数说明

| 参数名 | 输入/输出 | 描述 |
| :--- | :--- | :--- |
| dst | 输出 | 目的GM的起始地址。需要1字节对齐。 |
| src | 输入 | 源UB的起始地址。需要32字节对齐。 |
| size | 输入 | 搬运数据大小，单位为字节。取值范围：[0, 2097151]。 |
| burst_count | 输入 | 待搬运的连续传输数据块个数。取值范围：[0, 4095]。 |
| burst_len | 输入 | 待搬运的每个连续传输数据块的长度，单位为字节。取值范围：[0, 2097151]。 |
| left_padding_num | 输入 | 保留参数，必须设置为0。 |
| right_padding_num | 输入 | 保留参数，必须设置为0。 |
| src_gap | 输入 | 源操作数相邻连续数据块的距离（前面一个数据块的尾与后面一个数据块的头的间隔），单位为DataBlock（32字节）。|
| dst_gap | 输入 | 目的操作数相邻连续数据块的距离（前面一个数据块的尾与后面一个数据块的头的间隔），单位为字节。 |

## 返回值说明

无

## 流水类型

PIPE_MTE3

## 约束说明

- 各存储单元的空间大小和对齐要求请参考[存储单元说明](../../general_description_and_constraints.md#存储单元说明)。
- 当`burst_count`、`burst_len`中任意一个值为0时，该接口被视为NOP（空操作）。
- 当`size`值为0时，该接口被视为NOP（空操作）。
- 如果本指令与其他指令存在数据依赖，需要插入[同步指令](../../sync/system_sync_overview.md)，保证多个指令串行化，防止出现异常数据。

## 调用示例

```cpp
//total_length指参与搬运的数据总长度。dst是外部输入的uint8_t类型的GM内存。
constexpr uint64_t total_length = 128;
__ubuf__ uint8_t src[total_length];
asc_copy_ub2gm_align(dst, src, total_length * sizeof(uint8_t));
```

## 配套 API 与常见组合

- [asc_init](../../utils/sys_init/asc_init.md)——init 卡明示复位 GM↔UB 搬运循环次数与原子累加状态直接决定 copy 行为基线，卡片示例演示 init 复位原子模式后 ub2gm 变普通搬运
- [asc_copy_gm2ub_align](../asc_copy_gm2ub_align/asc_copy_gm2ub_align_arch_2201.md)——算子数据流的搬入/回写闭环（MTE2 入、MTE3 出），P(b|a)=1.00，两侧对齐与 padding 语义成对互补
- [asc_sync（同步）](../../sync/intra_core_sync/asc_sync.md)——异步UB→GM搬出发起后须asc_sync等待落盘方可结束kernel，abs.asc中两行相邻构成kernel末段确定步骤
