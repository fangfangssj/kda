---
type: API
title: Min
description: 返回指定数据类型的最小有限值，浮点类型返回最小规约正数，支持标量和矢量两种接口。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/tool_interface/NumericLimits_utils/Min.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/basic_api/kernel_operator_limits_intf.h
---

# Min

## 功能说明

头文件路径为：`"basic_api/kernel_operator_limits_intf.h"`。

返回指定数据类型的最小有限值，对于浮点数类型来说，返回其最小规约正数。

## 函数原型

- 标量接口，返回值为标量：

    ```cpp
    constexpr __aicore__ static inline T Min()
    ```

- 矢量接口，为dst前count个元素赋最小有限值：

    ```cpp
    __aicore__ static inline void Min(const LocalTensor<T>& dst, uint32_t count)
    ```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| dst | 输出 | 目的操作数。<br><br>类型为[LocalTensor](../../data_structures/local_tensor/index.md)，支持的TPosition为VECIN/VECCALC/VECOUT。<br><br>LocalTensor的起始地址需要32字节对齐。 |
| count | 输入 | 输入数据元素个数。 |

## 数据类型


针对Ascend 950PR&950DT系列产品，支持的数据类型为：int8_t、uint8_t、int16_t、uint16_t、half、bfloat16_t、int32_t、uint32_t、float。


## 返回值说明

标量接口返回值为对应数据类型的最小有限值。

矢量接口无返回值。

**表 2**  Min各数据类型输出

| 数据类型 | 十进制数 | 二进制各bit位 |
| --- | --- | --- |
| uint8_t | 0 | 0000 0000 |
| int8_t | -128 | 1000 0000 |
| uint16_t | 0 | 0000 0000 0000 0000 |
| int16_t | -32768 | 1000 0000 0000 0000 |
| half | 6.1e-5 | 0 00001 0000000000 |
| bfloat16_t | 1.18e-38 | 0 00000001 0000000 |
| uint32_t | 0 | 0000 0000 0000 0000 0000 0000 0000 0000 |
| int32_t | -2147483648 | 1000 0000 0000 0000 0000 0000 0000 0000 |
| float | 1.18e-38 | 0 00000001 00000000000000000000000 |

## 约束说明

标量接口无约束，矢量接口使用[duplicate](../../memory_vector_compute/data_padding/duplicate.md)填充前count个元素赋最小有限值，参照Memory矢量计算下的数据填充接口duplicate约束。

## 调用示例

- 标量接口：

    ```cpp
    int32_t value = AscendC::NumericLimits<int32_t>::Min();
    ```

- 矢量接口：

    ```cpp
    AscendC::NumericLimits<int32_t>::Min(dstLocal, 256); // 返回256个int32_t类型的最小有限值
    ```
