---
type: API
title: QuietNaN
description: 返回指定数据类型的安静NaN值，支持标量和矢量两种接口，矢量接口为dstLocal前count个元素赋安静NaN值。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/tool_interface/NumericLimits_utils/QuietNaN.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/basic_api/kernel_operator_limits_intf.h
---

# QuietNaN

## 功能说明

头文件路径为：`"basic_api/kernel_operator_limits_intf.h"`。

返回指定数据类型的[安静NaN值](../../data_structures/builtin_data_types.md)（浮点尾数最高位为1）。

## 函数原型

- 标量接口，返回值为标量：

    ```cpp
    constexpr __aicore__ static inline T QuietNaN()
    ```

- 矢量接口，为dstLocal前count个元素赋安静NaN值：

    ```cpp
    __aicore__ static inline void QuietNaN(const LocalTensor<T>& dstLocal, uint32_t count)
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

标量接口返回值为对应数据类型的安静NaN值。

矢量接口无返回值。

**表 2**  QuietNaN各数据类型输出

| 数据类型 | 十进制数 | 二进制各bit位 |
| --- | --- | --- |
| half | nan | 0 11111 1000000000 |
| bfloat16_t | nan | 0 11111111 1000000 |
| float | nan | 0 11111111 10000000000000000000000 |

## 约束说明

标量接口无约束，矢量接口使用[duplicate](../../memory_vector_compute/data_padding/duplicate.md)填充前count个元素赋安静NaN值，参照Memory矢量计算下的数据填充接口duplicate约束。

## 调用示例

- 标量接口：

    ```cpp
    float value = AscendC::NumericLimits<float>::QuietNaN();
    ```

- 矢量接口：

    ```cpp
    AscendC::NumericLimits<float>::QuietNaN(dstLocal, 256); // 返回256个float类型的NaN值
    ```

- [AddRmsNormDynamicMxQuant 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/norm/add_rms_norm_dynamic_mx_quant_950.md)
- [BatchNorm 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/norm/batch_norm_950.md)
- [RmsNormDynamicMxQuant 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/norm/rms_norm_dynamic_mx_quant_950.md)
- [RmsNormGrad 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/norm/rms_norm_grad_950.md)
- [RmsNormGradQuant 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/norm/rms_norm_grad_quant_950.md)
