---
type: API
title: Init
description: 对Matmul对象的Tiling数据进行初始化，根据Tiling参数进行资源划分。Tiling参数可变场景下可多次调用。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.programming_model.execution_unit.cube
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/cube_compute/Matmul_Kernel/Init.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
- a2
- a3
- 310p
- kirin_x90
- kirin_9030
---

# Init

## 功能说明

Init主要用于对Matmul对象中的Tiling数据进行初始化，根据Tiling参数进行资源划分，Tiling参数的具体介绍请参考[Matmul Tiling侧接口](../matmul_tiling/index.md)。

开发者可以先通过[REGIST\_MATMUL\_OBJ](regist_matmul_obj.md)不传入Tiling参数对单个Matmul对象进行初始化，后续通过Init接口单独传入Tiling参数，对Matmul对象中的Tiling数据进行调整。比如，Tiling参数可变的场景下，可以通过多次调用Init来重新设置Tiling参数。

不需要Tiling变更的场景下，推荐使用[REGIST\_MATMUL\_OBJ](regist_matmul_obj.md)传入Tiling参数进行初始化。

## 函数原型

-   Tiling参数传入栈地址

    ```
    __aicore__ inline void Init(const TCubeTiling* __restrict cubeTiling, TPipe* tpipe = nullptr)
    ```

-   Tiling参数传入GM地址

    ```
    __aicore__ inline void Init(const __gm__ TCubeTiling* gmCubeTiling, TPipe* tpipe = nullptr)
    ```

## 参数说明

**表1**  Tiling参数传入栈地址接口参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| cubeTiling | 输入 | Matmul Tiling参数，TCubeTiling结构体定义请参见[表1 TCubeTiling结构说明](../matmul_tiling/t_cube_tiling_struct.md#tcubetiling-struct)。<br><br>Tiling参数可以通过host侧[GetTiling](../matmul_tiling/get_tiling.md)接口获取，并传递到核函数（Kernel）侧使用。在核函数（Kernel）侧调用[GET_TILING_DATA](../../../basic_api/kernel_tiling/get_tiling_data.md)实现将Tiling参数搬运到AI Core内的栈空间中，本接口传入Tiling参数中TCubeTiling结构体的栈地址。 |
| tpipe | 输入 | Tpipe对象。 |

**表2**  Tiling参数传入GM地址接口参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| gmCubeTiling | 输入 | Matmul Tiling参数，该参数指向gm上的一块内存地址，其中的数据类型是TCubeTiling结构体，TCubeTiling结构体定义请参见[表1 TCubeTiling结构说明](../matmul_tiling/t_cube_tiling_struct.md#tcubetiling-struct)。<br><br>Tiling参数可以通过host侧[GetTiling](../matmul_tiling/get_tiling.md)接口获取，并传递到核函数（Kernel）侧使用。在核函数（Kernel）侧调用[GET_TILING_DATA_PTR_WITH_STRUCT](../../../basic_api/kernel_tiling/get_tiling_data_ptr_with_struct.md)获取gm上Tiling参数的指针，本接口传入Tiling参数中TCubeTiling结构体的GM地址。 |
| tpipe | 输入 | Tpipe对象。 |

## 返回值说明

无

## 约束说明

-   Tiling参数传入栈地址的接口：

    无

-   Tiling参数传入GM地址的接口：
    -   仅支持Matmul Tiling参数的[部分常量化](get_matmul_api_tiling.md#功能说明)场景。
    -   不支持CPU域调试。

## 调用示例

-   Tiling参数传入栈地址

    ```
    GET_TILING_DATA(tilingData, tiling);
    // ...
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm);
    mm.Init(&(tiling.cubeTilingData));
    ```

-   Tiling参数传入GM地址
    -   纯Cube模式

        ```
        #define ASCENDC_CUBE_ONLY

        GET_TILING_DATA_PTR_WITH_STRUCT(MatmulCustomTilingData, tilingDataPtr, tiling);
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
        // ...
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm);
        mm.Init(&(tilingDataPtr->cubeTilingData));
        ```

    -   MIX模式

        ```
        GET_TILING_DATA_PTR_WITH_STRUCT(MatmulCustomTilingData, tilingDataPtr, tiling);
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
        // ...
        // MIX模式下，只调用REGIST_MATMUL_OBJ接口，传入Tiling参数的GM地址，不需调用Init接口
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &(tilingDataPtr->cubeTilingData));
        ```

- [FusedLinearCrossEntropyLossGrad A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/nn/matmul/fused_linear_cross_entropy_loss_grad.md)
- [Sparse4to2QuantMatmulWeightNz A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/nn/matmul/sparse4to2_quant_matmul_weight_nz.md)
- [MhcPreBackward 950 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mhc/mhc_pre_backward_950.md)
- [TransposeQuantBatchMatMul 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/matmul/transpose_quant_batch_mat_mul_950.md)
