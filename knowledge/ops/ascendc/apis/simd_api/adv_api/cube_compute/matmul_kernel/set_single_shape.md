---
type: API
title: SetSingleShape
description: 设置Matmul单核计算的形状singleCoreM、singleCoreN、singleCoreK，单位为元素。用于运行时修改shape，比如复用Matmul对象来处理尾块。与SetTail接口功能一致，建议使用本接口。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.programming_model.execution_unit.cube
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/cube_compute/Matmul_Kernel/SetSingleShape.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
- a2
- a3
- 310p
- kirin_x90
---

# SetSingleShape

## 功能说明

设置Matmul单核计算的形状singleCoreM、singleCoreN、singleCoreK，单位为元素。用于运行时修改shape，比如复用Matmul对象来处理尾块。与SetTail接口功能一致，建议使用本接口。

## 函数原型

```
__aicore__ inline void SetSingleShape(int singleM, int singleN, int singleK)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| singleM | 输入 | 设置的singleCoreM大小，单位为元素。 |
| singleN | 输入 | 设置的singleCoreN大小，单位为元素。 |
| singleK | 输入 | 设置的singleCoreK大小，单位为元素。 |

## 返回值说明

无

## 约束说明

无

## 调用示例

```
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);

// tailM：M方向上剩余元素个数，tailN：N方向上剩余元素个数，tailK：K方向上剩余元素个数
// 如果是尾核，剩余元素可能会少于单核需要计算的元素。此时，需要使用SetSingleShape重新设置本次计算的元素个数
if (tailM < tiling.singleCoreM || tailN < tiling.singleCoreN || tailK < tiling.singleCoreK) {
    matmulObj.SetSingleShape(tailM, tailN, tailK);
}

mm.SetTensorA(gm_a);
mm.SetTensorB(gm_b);
if (tiling.isBias) {
    mm.SetBias(gmBias);
}
mm.IterateAll(gm_c);
mm.End();
```

- [FusedLinearCrossEntropyLossGrad A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/nn/matmul/fused_linear_cross_entropy_loss_grad.md)
- [UpsampleBilinear2dAABackward A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/cv/image/upsample_bilinear_2d_aa_backward.md)
- [TransposeBatchMatMul A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/nn/matmul/transpose_batch_mat_mul.md)
- [MhcPostBackward 950 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mhc/mhc_post_backward_950.md)
- [MhcPreSinkhorn A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mhc/mhc_pre_sinkhorn.md)
- [RotaryPositionEmbedding A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/posembedding/rotary_position_embedding.md)
