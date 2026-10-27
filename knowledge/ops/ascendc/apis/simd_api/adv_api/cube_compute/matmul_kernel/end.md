---
type: API
title: 矩阵计算End
description: 多个Matmul对象之间切换计算时，必须调用一次End函数，用于释放Matmul计算资源，防止多个Matmul对象的计算资源冲突。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.programming_model.execution_unit.cube
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/cube_compute/Matmul_Kernel/End.md
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

# End

## 功能说明

多个Matmul对象之间切换计算时，必须调用一次End函数，用于释放Matmul计算资源，防止多个Matmul对象的计算资源冲突。

## 函数原型

```
__aicore__ inline void End()
```

## 参数说明

无

## 返回值说明

无

## 约束说明

无

## 调用示例

```
mm1.IterateAll(gm_c1);
mm1.End();
mm2.IterateAll(gm_c2);
mm2.End();
```

- [FusedLinearCrossEntropyLossGrad A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/nn/matmul/fused_linear_cross_entropy_loss_grad.md)
- [Sparse4to2QuantMatmulWeightNz A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/nn/matmul/sparse4to2_quant_matmul_weight_nz.md)
- [MhcPostBackward 950 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mhc/mhc_post_backward_950.md)
- [WeightQuantBatchMatmulNz 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/matmul/weight_quant_batch_matmul_nz_950.md)
- [MhcPreSinkhorn A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mhc/mhc_pre_sinkhorn.md)
