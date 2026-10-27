---
type: API
title: asc_fill_l0a
description: 头文件路径为："c_api/cube_datamove/cube_datamove.h"。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/cube_datamove/cube_load_aux_config/asc_fill_l0a.md
  role: primary
created_at: '2026-10-20T07:46:54Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- a2
- a3
source_headers:
- include/c_api/cube_datamove/cube_datamove.h
---

# asc_fill_l0a

## 功能说明

头文件路径为：`"c_api/cube_datamove/cube_datamove.h"`。

将value的二进制值赋值给dst，对L0A Buffer的Local Memory进行初始化。

填充操作用于准备矩阵输入缓冲区，后续数据搬入及矩阵乘加的衔接参见[矩阵计算流程](../../cube_compute/overview/cube_compute_flow.md)；输入缓冲区布局参见[L1 Buffer、L0A Buffer/L0B Buffer内存结构介绍](../cube_compute_load/l1_l0_a_b_memory_structure_intro.md)。

## 函数原型

- 常规计算

    ```cpp
    __aicore__ inline void asc_fill_l0a(__ca__ int16_t* dst, half value, const asc_fill_value_config& config)
    __aicore__ inline void asc_fill_l0a(__ca__ int16_t* dst, uint32_t value, const asc_fill_value_config& config)
    __aicore__ inline void asc_fill_l0a(__ca__ uint16_t* dst, half value, const asc_fill_value_config& config)
    __aicore__ inline void asc_fill_l0a(__ca__ uint16_t* dst, uint32_t value, const asc_fill_value_config& config)
    __aicore__ inline void asc_fill_l0a(__ca__ half* dst, half value, const asc_fill_value_config& config)
    __aicore__ inline void asc_fill_l0a(__ca__ half* dst, uint32_t value, const asc_fill_value_config& config)
    __aicore__ inline void asc_fill_l0a(__ca__ bfloat16_t* dst, bfloat16_t value, const asc_fill_value_config& config)
    __aicore__ inline void asc_fill_l0a(__ca__ bfloat16_t* dst, half value, const asc_fill_value_config& config)
    __aicore__ inline void asc_fill_l0a(__ca__ bfloat16_t* dst, uint32_t value, const asc_fill_value_config& config)
    __aicore__ inline void asc_fill_l0a(__ca__ int32_t* dst, half value, const asc_fill_value_config& config)
    __aicore__ inline void asc_fill_l0a(__ca__ int32_t* dst, uint32_t value, const asc_fill_value_config& config)
    __aicore__ inline void asc_fill_l0a(__ca__ uint32_t* dst, half value, const asc_fill_value_config& config)
    __aicore__ inline void asc_fill_l0a(__ca__ uint32_t* dst, uint32_t value, const asc_fill_value_config& config)
    __aicore__ inline void asc_fill_l0a(__ca__ float* dst, half value, const asc_fill_value_config& config)
    __aicore__ inline void asc_fill_l0a(__ca__ float* dst, uint32_t value, const asc_fill_value_config& config)
    ```

## 参数说明

**表1** 参数说明

| 参数名  | 输入/输出 | 描述 |
| :----- | :------- | :------- |
| dst | 输出 | 目的操作数（矢量）的起始地址。 |
| value | 输入 | 源操作数（标量）。 |
| config | 输入 | 使用的初始化相关参数，详细说明请参考[asc_fill_value_config](../../defs/union/asc_fill_value_config.md)。 |

## 返回值说明

无

## 流水类型

PIPE_MTE1

## 约束说明

- 各存储单元的空间大小和对齐要求请参考[存储单元说明](../../general_description_and_constraints.md#存储单元说明)。
- 操作数地址重叠约束请参考[通用地址重叠约束](../../general_description_and_constraints.md#通用地址重叠约束)。
- 若dst和value的数据类型不相同，则需要根据dst的数据类型对value的值进行转换，具体请参考调用示例。

## 调用示例

```cpp
constexpr uint64_t total_length = 256;    // total_length指参与搬运的数据总长度
asc_fill_value_config config;

// dst和value数据类型不相同
__ca__ float dst0[total_length];
uint32_t value0 = 0x3F800000;  // float类型的1的二进制表示为0x3F800000
config.repeat = 1;
config.blk_num = total_length * sizeof(float) / 512;
config.dst_gap = 0;
asc_fill_l0a(dst0, value0, config);    // 将dst0中256个float类型的元素初始化为1

// dst和value数据类型相同
__ca__ half dst1[total_length];
half value1 = 1;  // 不用做二进制转换，直接填1即可
config.repeat = 1;
config.blk_num = total_length * sizeof(half) / 512;
config.dst_gap = 0;
asc_fill_l0a(dst1, value1, config);    // 将dst1中256个half类型的元素初始化为1
```
