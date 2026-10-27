---
type: Guide
title: 矩阵乘输出的量化/反量化
description: 对于特定输入输出数据类型，Matmul支持将计算结果从L0C Buffer（CO1）搬出到Global Memory时，对输出C矩阵元素执行数据量化或反量化操作。
tags:
- task.lookup
- topic.programming_model.execution_model.simd
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/guide/operator_practice/simd_operator_impl/matrix_advanced_api/feature_scenarios/matmul_output_quant_dequant.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms: []
---

# 矩阵乘输出的量化/反量化

## 功能介绍

对于[特定输入输出数据类型](#zh-cn_topic_0000002298767893_table1996113269499)，Matmul支持将计算结果从L0C Buffer（CO1）搬出到Global Memory时，对输出C矩阵元素执行数据量化或反量化操作。

-   Matmul量化场景：Matmul计算时左矩阵A、右矩阵B为half或bfloat16\_t数据类型，输出C矩阵为int8\_t数据类型。该场景下，C矩阵的数据从L0C Buffer（CO1）搬出到Global Memory时，会执行量化操作，将最终结果量化为int8\_t类型，如下图所示。

    **图1**  Matmul量化场景示意图  
    ![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/guide/figures/mm_quant.png "Matmul量化场景示意图")

-   Matmul反量化场景：Matmul计算时左矩阵A、右矩阵B为int8\_t或int4b\_t数据类型，输出C矩阵为half数据类型，或者左矩阵A、右矩阵B为int8\_t数据类型，输出C矩阵为int8\_t数据类型。该场景下，C矩阵的数据从L0C Buffer（CO1）搬出到Global Memory时，会执行反量化操作，将最终结果反量化为对应的half类型或int8\_t类型，如下图所示。

    **图2**  Matmul反量化场景示意图  
    ![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/guide/figures/mm_dequant.png "Matmul反量化场景示意图")

Matmul量化/反量化包含两种模式：同一系数的量化/反量化模式、向量的量化/反量化模式，开发者在算子Tiling侧调用[SetDequantType](../../../../../apis/simd_api/adv_api/cube_compute/matmul_tiling/set_dequant_type.md)接口设置量化或反量化模式，这两种模式的具体区别为：

-   同一系数的量化/反量化模式（PER\_TENSOR模式）：整个C矩阵对应一个量化参数，量化参数的shape为\[1\]。开发者在算子核函数（Kernel）侧调用接口[SetQuantScalar](../../../../../apis/simd_api/adv_api/cube_compute/matmul_kernel/set_quant_scalar.md)设置量化参数。
-   向量的量化/反量化模式（PER\_CHANNEL模式）：C矩阵的shape为\[m, n\]，每个channel维度即C矩阵的每一列，对应一个量化参数，量化参数的shape为\[n\]。开发者在算子核函数（Kernel）侧调用接口[SetQuantVector](../../../../../apis/simd_api/adv_api/cube_compute/matmul_kernel/set_quant_vector.md)设置量化参数。

**表1**  量化/反量化模式对应的接口配置

| 模式 | Tiling侧接口 | 核函数（Kernel）侧接口 |
| --- | --- | --- |
| 同一系数的量化/反量化 | SetDequantType(DequantType::SCALAR) | SetQuantScalar(gmScalar) |
| 向量的量化/反量化 | SetDequantType(DequantType::TENSOR) | SetQuantVector(gmTensor) |

## 使用场景

需要对矩阵计算结果进行量化/反量化操作的场景，当前该场景下，Matmul输入输出矩阵支持的数据类型如下表所示。

**表2**  Matmul量化/反量化支持的数据类型<a name="zh-cn_topic_0000002298767893_table1996113269499"></a>

| A矩阵 | B矩阵 | C矩阵 | 支持平台 |
| --- | --- | --- | --- |
| half | half | int8_t |  |
| bfloat16_t | bfloat16_t | int8_t |  |
| int8_t | int8_t | half |  |
| int4b_t | int4b_t | half |  |
| int8_t | int8_t | int8_t |  |
| int8_t | int8_t | bfloat16_t |  |
| fp8_e4m3fn_t/fp8_e5m2_t | fp8_e4m3fn_t/fp8_e5m2_t | fp8_e4m3fn_t/half/bfloat16_t/float |  |
| hifloat8_t | hifloat8_t | hifloat8_t/half/bfloat16_t/float | 注意：<br><br>输出为hifloat8_t时，采用Half to Away Round方式量化。<br><br>量化场景的输出为float类型时，该量化模式精度无法达到双万分之一，可以达到双千分之一。如果有双万分之一的精度要求，建议使用AscendDeQuant高阶API。 |

## 约束说明

-   [SetQuantScalar](../../../../../apis/simd_api/adv_api/cube_compute/matmul_kernel/set_quant_scalar.md)和[SetQuantVector](../../../../../apis/simd_api/adv_api/cube_compute/matmul_kernel/set_quant_vector.md)接口必须在[Iterate](../../../../../apis/simd_api/adv_api/cube_compute/matmul_kernel/iterate.md)或者[IteratAll](../../../../../apis/simd_api/adv_api/cube_compute/matmul_kernel/iterate_all.md)接口前调用。

-   在核函数（Kernel）侧与Tiling侧设置的量化/反量化模式需要保持一致：
    -   核函数（Kernel）侧调用SetQuantScalar接口设置同一系数的量化/反量化模式，对应Tiling侧调用SetDequantType接口配置模式为DequantType::SCALAR。
    -   核函数（Kernel）侧调用SetQuantVector接口设置向量的量化/反量化模式，对应Tiling侧调用SetDequantType接口配置模式为DequantType::TENSOR。

-   当A、B矩阵为int8\_t或int4b\_t类型，C矩阵为half时，本节特性的输出结果不支持INF\_NAN模式。若结果需要以INF\_NAN输出，建议在调用Matmul API时将结果输出到TPosition::VECIN，同时将输出的数据类型设置为int32\_t，再基于AIV核使用高阶API  [AscendDequant](../../../../../apis/simd_api/adv_api/quantization/ascend_dequant.md)将该结果反量化为half类型。

## 调用示例

完整的算子样例请参考[matmul\_quant样例](https://gitcode.com/cann/asc-devkit/tree/639b2b5c1031188107d23d785d52b935f1b4f13b/examples/01_simd_cpp_api/04_advanced_api/00_matmul/matmul_quant)。

-   Tiling实现

    调用[SetDequantType](../../../../../apis/simd_api/adv_api/cube_compute/matmul_tiling/set_dequant_type.md)接口设置量化或反量化模式，其他实现内容与基础场景相同。

    ```
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    matmul_tiling::MatmulApiTiling tiling(ascendcPlatform);
    tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT8);
    tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT8);
    tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    tiling.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT32);
    tiling.SetShape(M, N, K);
    tiling.SetOrgShape(M, N, K);
    tiling.EnableBias(true);
    tiling.SetDequantType(DequantType::SCALAR);  // 设置同一系数的量化/反量化模式
    // tiling.SetDequantType(DequantType::TENSOR);  // 设置向量的量化/反量化模式
    ... // 执行其他配置
    ```

-   核函数（Kernel）实现

    根据具体量化模式场景，调用[SetQuantScalar](../../../../../apis/simd_api/adv_api/cube_compute/matmul_kernel/set_quant_scalar.md)或[SetQuantVector](../../../../../apis/simd_api/adv_api/cube_compute/matmul_kernel/set_quant_vector.md)接口设置量化参数。其他实现内容与基础场景相同。

    -   同一系数的量化/反量化模式

        ```
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
        float tmp = 0.1;  // 输出gm时会乘以0.1
        uint64_t ans = static_cast<uint64_t>(*reinterpret_cast<int32_t*>(&tmp)); // 浮点值量化系数转换为uint64_t类型进行设置
        mm.SetQuantScalar(ans);
        mm.SetTensorA(gm_a);
        mm.SetTensorB(gm_b);
        mm.SetBias(gm_bias);
        mm.IterateAll(gm_c);
        ```

    -   向量的量化/反量化模式

        ```
        GlobalTensor gmQuant;
        ...
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
        mm.SetQuantVector(gmQuant);
        mm.SetTensorA(gm_a);
        mm.SetTensorB(gm_b);
        mm.SetBias(gm_bias);
        mm.IterateAll(gm_c);
        ```
