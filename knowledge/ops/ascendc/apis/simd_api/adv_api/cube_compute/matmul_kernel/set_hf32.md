---
type: API
title: SetHF32
description: 在纯Cube模式（只有矩阵计算）下，设置是否开启HF32（矩阵乘计算时可采用的数据类型）模式。开启后，在矩阵乘计算时，float32数据类型会转换为hf32数据类型，可提升计算性能，但同时也会带来精度损失。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.programming_model.execution_unit.cube
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/cube_compute/Matmul_Kernel/SetHF32.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
- a2
- a3
---

# SetHF32

## 功能说明

在纯Cube模式（只有矩阵计算）下，设置是否开启HF32（矩阵乘计算时可采用的数据类型）模式。开启后，在矩阵乘计算时，float32数据类型会转换为hf32数据类型，可提升计算性能，但同时也会带来精度损失。

## 函数原型

```
__aicore__ inline void SetHF32(bool enableHF32 = false, int32_t transMode = 0)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| enableHF32 | 输入 | 配置是否开启HF32模式，默认值false(不开启)。 |
| transMode | 输入 | 配置在开启HF32模式时，float转换为hf32时所采用的ROUND模式。默认值0。<br><br>0：就近舍入，距离相等时向偶数进位。<br><br>1：就近舍入，距离相等时向远离0方向进位。 |

## 返回值说明

无

## 约束说明

本接口仅支持在纯Cube模式下调用。

## 调用示例

```
//纯Cube模式
#define ASCENDC_CUBE_ONLY
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);    //  A/B/C/BIAS类型是float
mm.SetTensorA(gm_a);
mm.SetTensorB(gm_b);
if (tiling.isBias) {
    mm.SetBias(gmBias);
}
mm.SetHF32(true);
mm.IterateAll(gm_c);
mm.SetHF32(false);
mm.End();
```

- [WeightQuantBatchMatmul 950 Direct Invoke 算子知识](../../../../../operators/direct/nn/matmul/weight_quant_batch_matmul_950.md)
- [LSTM 950 Direct Invoke 算子知识](../../../../../operators/direct/nn/rnn/lstm_950.md)
- [MatmulAlltoAll 950 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/matmul_allto_all_950.md)
- [MhcPreSinkhorn A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mhc/mhc_pre_sinkhorn.md)
