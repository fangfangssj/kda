---
type: API
title: SetBias
description: 设置矩阵乘的Bias。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.programming_model.execution_unit.cube
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/cube_compute/Matmul_Kernel/SetBias.md
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

# SetBias

## 功能说明

设置矩阵乘的Bias。

## 函数原型

```
__aicore__ inline void SetBias(const GlobalTensor<BiasT>& biasGlobal)
```

```
__aicore__ inline void SetBias(const LocalTensor<BiasT>& inputBias)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| biasGlobal | 输入 | Bias矩阵。类型为[GlobalTensor](../../../basic_api/data_structures/global_tensor/index.md)。不同型号支持的数据类型请参考[支持的数据类型](#li12616155731720)。<br><br>A矩阵、B矩阵、Bias支持的数据类型组合可参考[Matmul输入输出数据类型的组合说明](matmul_usage.md#table1996113269499)；在MxMatmul场景，A矩阵、B矩阵、Bias支持的数据类型组合可参考[MatmulTypeWithScale参数说明](../../../../../guides/operator_practice/simd_operator_impl/matrix_advanced_api/feature_scenarios/mx_matmul_scenario.md#zh-cn_topic_0000002270097206_table14759942142014)。 |
| inputBias | 输入 | Bias矩阵。类型为[LocalTensor](../../../basic_api/data_structures/local_tensor/index.md)，支持的TPosition为TSCM/VECOUT。不同型号支持的数据类型请参考[支持的数据类型](#li12616155731720)。<br><br>A矩阵、B矩阵、Bias支持的数据类型组合可参考[Matmul输入输出数据类型的组合说明](matmul_usage.md#table1996113269499)；在MxMatmul场景，A矩阵、B矩阵、Bias支持的数据类型组合可参考[MatmulTypeWithScale参数说明](../../../../../guides/operator_practice/simd_operator_impl/matrix_advanced_api/feature_scenarios/mx_matmul_scenario.md#zh-cn_topic_0000002270097206_table14759942142014)。 |

## 返回值说明

无

## 约束说明

-   在Matmul Tiling计算中，必须配置TCubeTiling结构中的[isBias](../matmul_tiling/t_cube_tiling_struct.md#p2051215216314)参数为1，即启用Bias后，才能调用本接口设置Bias矩阵。
-   传入的Bias地址空间大小需要保证不小于singleN。

-   对于Ascend 950PR&950DT系列产品，Bias矩阵的内存逻辑位置为TSCM且数据类型为float或int32\_t时，Bias矩阵的LocalTensor空间必须64字节对齐。


-   支持的数据类型<a id="li12616155731720"></a>

    
    Ascend 950PR&950DT系列产品，支持的数据类型为：half、bfloat16_t、float、int32_t，其中仅在A、B的数据类型为int8_t时，Bias的数据类型可以设置为int32_t。
    

    
    Atlas A3系列产品，支持的数据类型为：half、float、int32_t，其中仅在A、B的数据类型为int8_t时，Bias的数据类型可以设置为int32_t。
    

    
    Atlas A2系列产品，支持的数据类型为：half、float、int32_t，其中仅在A、B的数据类型为int8_t时，Bias的数据类型可以设置为int32_t。
    

    
    Atlas推理系列产品AI Core，支持的数据类型为：half、float、int32_t，对于包含biasGlobal参数的原型，仅在A、B的数据类型为int8_t时，Bias的数据类型可以设置为int32_t。
    

    
    Atlas 200I/500 A2推理产品，支持的数据类型为：half、float、int32_t，其中仅在A、B的数据类型为int8_t时，Bias的数据类型可以设置为int32_t。
    

    
    Kirin X90处理器系列产品，支持的数据类型为：half、int32_t，其中仅在A、B的数据类型为int8_t时，Bias的数据类型可以设置为int32_t。
    





## 调用示例

```
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
mm.SetTensorA(gm_a);
mm.SetTensorB(gm_b);
if (tiling.isBias) {
    mm.SetBias(gmBias);  // 设置Bias
}
mm.IterateAll(gm_c);
mm.End();
```
