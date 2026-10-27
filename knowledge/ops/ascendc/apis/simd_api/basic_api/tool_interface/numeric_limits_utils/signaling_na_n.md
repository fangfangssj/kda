---
type: API
title: SignalingNaN
description: 返回指定数据类型的发信NaN值（浮点尾数最高位为0），提供标量和矢量两种接口。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/tool_interface/NumericLimits_utils/SignalingNaN.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/basic_api/kernel_operator_limits_intf.h
---

# SignalingNaN

## 功能说明

头文件路径为：`"basic_api/kernel_operator_limits_intf.h"`。

返回指定数据类型的[发信NaN值](../../data_structures/builtin_data_types.md)（浮点尾数最高位为0）。

## 函数原型

- 标量接口，返回值为标量：

    ```cpp
    constexpr __aicore__ static inline T SignalingNaN()
    ```

- 矢量接口，为dstLocal前count个元素赋发信NaN值：

    ```cpp
    __aicore__ static inline void SignalingNaN(const LocalTensor<T>& dstLocal, uint32_t count)
    ```

## 参数说明

**表1** 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| dstLocal | 输出 | 目的操作数。<br><br>类型为[LocalTensor](../../data_structures/local_tensor/index.md)，支持的TPosition为VECIN/VECCALC/VECOUT。<br><br>LocalTensor的起始地址需要32字节对齐。 |
| count | 输入 | 输入数据元素个数。 |

## 数据类型


针对Ascend 950PR&950DT系列产品，支持的数据类型为：half、bfloat16_t、float。


## 返回值说明

标量接口返回值为对应数据类型的发信NaN值。

矢量接口无返回值。

**表 2**  SignalingNaN各数据类型输出

| 数据类型 | 十进制数 | 二进制各bit位 |
| --- | --- | --- |
| half | nan | 0 11111 0100000000 |
| bfloat16_t | nan | 0 11111111 0100000 |
| float | nan | 0 11111111 01000000000000000000000 |

## 约束说明

标量接口无约束，矢量接口使用[duplicate](../../memory_vector_compute/data_padding/duplicate.md)填充前count个元素赋发信NaN值，参照Memory矢量计算下的数据填充接口duplicate约束。

## 调用示例

- 标量接口：

    ```cpp
    float value = AscendC::NumericLimits<float>::SignalingNaN();
    ```

- 矢量接口：

    ```cpp
    AscendC::NumericLimits<float>::SignalingNaN(dstLocal, 256); // 返回256个float类型的NaN值
    ```
