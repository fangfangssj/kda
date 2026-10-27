---
type: API
title: SetTensorScaleA
description: MxMatmul场景，设置矩阵乘中左矩阵的量化系数矩阵scaleA。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.programming_model.execution_unit.cube
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/cube_compute/Matmul_Kernel/SetTensorScaleA.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/adv_api/matmul/matmul_client.h
---

# SetTensorScaleA

## 功能说明

MxMatmul场景，设置矩阵乘中左矩阵的量化系数矩阵scaleA。

## 函数原型

```
__aicore__ inline void SetTensorScaleA(const GlobalTensor<ScaleT>& gm, bool isTransposeScaleA = false);
```

```
__aicore__ inline void SetTensorScaleA(const LocalTensor<ScaleT>& leftMatrix, bool isTransposeScaleA = false);
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| gm | 输入 | 量化系数scaleA矩阵。类型为[GlobalTensor](../../../basic_api/data_structures/global_tensor/index.md)。支持的数据类型为：fp8_e8m0_t。 |
| leftMatrix | 输入 | 量化系数scaleA矩阵。类型为[LocalTensor](../../../basic_api/data_structures/local_tensor/index.md)，支持的TPosition为TSCM/VECOUT。支持的数据类型为：fp8_e8m0_t。 |
| isTransposeScaleA | 输入 | scaleA矩阵是否需要转置。<br><br>参数支持的取值如下：<br>false：默认值，scaleA矩阵不转置。true：scaleA矩阵转置。<br><br>注意：<br>scaleA矩阵为NZ格式时，该参数只支持取值为false。<br>若scaleA矩阵的MatmulTypeWithScale参数中的SCALE_ISTRANS设置为true，除scaleA为NZ格式场景，该参数支持取值为true、false，即运行时scaleA矩阵可以转置和非转置交替使用。<br>若scaleA矩阵的MatmulTypeWithScale参数中的SCALE_ISTRANS设置为false，该参数只支持取值为false，若强行设置为true，精度会有异常。<br><br>对于有Bias输入的场景，为了确保Tiling侧与核函数（Kernel）侧L1 Buffer空间计算大小保持一致及结果精度正确，该参数取值必须与核函数（Kernel）侧定义A矩阵MatmulTypeWithScale的SCALE_ISTRANS参数以及Tiling侧SetScaleAType()接口的[isScaleTrans](../matmul_tiling/set_scale_a_type.md#p2934103115919)参数保持一致，即有Bias输入的场景，上述三个参数必须同时设置为true或同时设置为false。 |

## 返回值说明

无

## 约束说明

-   传入的scaleA地址空间大小必须不小于[singleCoreM](../matmul_tiling/t_cube_tiling_struct.md)\*[singleCoreK](../matmul_tiling/t_cube_tiling_struct.md)/32。
-   当开启MixDualMaster（双主模式）场景时，即模板参数[enableMixDualMaster](matmul_config.md#matmulconfig-params)设置为true，不支持使用该接口。

## 调用示例

```
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
mm.SetTensorA(gm_a);
mm.SetTensorB(gm_b);
mm.SetTensorScaleA(gm_scaleA);    // 设置左矩阵的量化系数矩阵scaleA
mm.SetTensorScaleB(gm_scaleB);
if (tiling.isBias) {
    mm.SetBias(gmBias);
}
mm.IterateAll(gm_c);
mm.End();
```

- [TransposeQuantBatchMatMul 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/matmul/transpose_quant_batch_mat_mul_950.md)
- [QuantGroupedMatMulAlltoAllv 950 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/quant_grouped_mat_mul_allto_allv_950.md)
