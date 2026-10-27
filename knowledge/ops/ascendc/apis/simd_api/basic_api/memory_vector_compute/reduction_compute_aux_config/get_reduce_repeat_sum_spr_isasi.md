---
type: API
title: GetReduceRepeatSumSpr
description: 获取ReduceSum接口前n个数据连续计算的结果，以全局变量形式存储。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.execution_unit.vector
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/memory_vector_compute/reduction_compute_aux_config/GetReduceRepeatSumSpr_ISASI.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-12T03:09:26Z'
platforms:
- '950'
- a2
- a3
source_headers:
- include/basic_api/kernel_operator_vec_reduce_intf.h
---

# GetReduceRepeatSumSpr(ISASI)

## 功能说明

头文件路径为：`"basic_api/kernel_operator_vec_reduce_intf.h"`。

获取[ReduceSum](../reduction_compute/reduce_sum.md)接口的计算结果，仅支持tensor前n个数据连续计算接口使用。计算结果以全局变量形式存储，可以随时调用获取。

## 函数原型

```cpp
template <typename T>
__aicore__ inline T GetReduceRepeatSumSpr()
```

## 参数说明

**表1**  模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 操作数数据类型。 |

## 数据类型

操作数支持的数据类型为：`half`、`float`。

## 返回值说明

`ReduceSum`接口的计算结果。

## 约束说明

仅支持配合`ReduceSum`的tensor前n个数据连续计算接口使用。

## 调用示例

具体可参考[GetReduceRepeatSumSpr样例](https://gitcode.com/cann/asc-devkit/tree/639b2b5c1031188107d23d785d52b935f1b4f13b/examples/01_simd_cpp_api/03_basic_api/01_memory_vector_compute/reduce_computation)。

```cpp
AscendC::LocalTensor<float> src;
AscendC::LocalTensor<float> work;
AscendC::LocalTensor<float> dst;
AscendC::ReduceSum(dst, src, work, 128);
float res = AscendC::GetReduceRepeatSumSpr<float>(); // 返回前128个数的和
```
