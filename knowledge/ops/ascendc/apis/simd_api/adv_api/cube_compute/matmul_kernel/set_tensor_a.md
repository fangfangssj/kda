---
type: API
title: SetTensorA
description: 设置矩阵乘的左矩阵A。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.programming_model.execution_unit.cube
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/cube_compute/Matmul_Kernel/SetTensorA.md
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

# SetTensorA

## 功能说明

设置矩阵乘的左矩阵A。

## 函数原型

```
__aicore__ inline void SetTensorA(const GlobalTensor<SrcAT>& gm, bool isTransposeA = false)
```

```
__aicore__ inline void SetTensorA(const LocalTensor<SrcAT>& leftMatrix, bool isTransposeA = false)
```

```
__aicore__ inline void SetTensorA(SrcAT aScalar)
```


Atlas推理系列产品AI Core不支持SetTensorA\(SrcAT aScalar\)接口原型。



Atlas 200I/500 A2推理产品，不支持SetTensorA\(SrcAT aScalar\)接口原型。


## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| gm | 输入 | A矩阵。类型为[GlobalTensor](../../../basic_api/data_structures/global_tensor/index.md)。SrcAT参数表示A矩阵的数据类型。不同型号支持的数据类型请参考[gm和leftMatrix支持的数据类型](#li12616155731720)。|
| leftMatrix | 输入 | A矩阵。类型为[LocalTensor](../../../basic_api/data_structures/local_tensor/index.md)，支持的TPosition为TSCM/VECOUT。SrcAT参数表示A矩阵的数据类型。不同型号支持的数据类型请参考[gm和leftMatrix支持的数据类型](#li12616155731720)<br><br>若设置TSCM首地址，默认矩阵可全载，已经位于TSCM，Iterate接口无需再进行GM->L1 Buffer（A1/B1）搬运。 |
| aScalar | 输入 | A矩阵中设置的值。支持传入标量数据，标量数据会被扩展为一个形状为[1, K]的tensor参与矩阵乘计算，tensor的数值均为该标量值。例如，开发者可以通过将aScalar设置为1来实现矩阵B在K方向的reduce sum操作。SrcAT参数表示A矩阵的数据类型。不同型号支持的数据类型请参考[aScalar支持的数据类型](#li12616155731723)。 |
| isTransposeA | 输入 | A矩阵是否需要转置。<br><br>注意：<br>若A矩阵MatmulType的ISTRANS参数设置为true，该参数可以为true也可以为false，即运行时可以转置和非转置交替使用；<br>若A矩阵MatmulType的ISTRANS参数设置为false，该参数只能设置为false，若强行设置为true，精度会有异常；<br>对于非half、非bfloat16_t输入类型的场景，为了确保Tiling侧与核函数（Kernel）侧L1 Buffer空间计算大小保持一致及结果精度正确，该参数取值必须与核函数（Kernel）侧定义A矩阵MatmulType的[ISTRANS](matmul_usage.md)参数以及Tiling侧SetAType()接口的[isTrans](../matmul_tiling/set_a_type.md)参数保持一致，即上述三个参数必须同时设置为true或同时设置为false。 |

## 返回值说明

无

## 约束说明

-   传入的TensorA地址空间大小需要保证不小于singleM \* singleK。

-   gm和leftMatrix支持的数据类型<a id="li12616155731720"></a>

    
    Ascend 950PR&950DT系列产品，支持的数据类型为：half、bfloat16_t、int8_t、float、fp8_e4m3fn_t、fp8_e5m2_t、hifloat8_t。
    

    
    Atlas A3系列产品，支持的数据类型为：int4b_t、half、bfloat16_t、int8_t、float。
    

    
    Atlas A2系列产品，支持的数据类型为：int4b_t、half、bfloat16_t、int8_t、float。
    

    
    Atlas推理系列产品AI Core，支持的数据类型为：half、int8_t、float。
    

    
    Atlas 200I/500 A2推理产品，支持的数据类型为：half、bfloat16_t、int8_t、float。
    

    
    Kirin X90处理器系列产品，支持的数据类型为：half、int8_t。
    

    
    Kirin 9030处理器系列产品，支持的数据类型为：half。
    

-   aScalar支持的数据类型<a id="li12616155731723"></a>

    
    Ascend 950PR&950DT系列产品，支持的数据类型为：half、float。
    

    
    Atlas A3系列产品，支持的数据类型为：half、float。
    

    
    Atlas A2系列产品，支持的数据类型为：half、float。
    

    
    Atlas推理系列产品AI Core，不支持该参数。
    

    
    Atlas 200I/500 A2推理产品，不支持该参数。
    

    
    Kirin X90处理器系列产品，支持的数据类型为：half。
    

    
    Kirin 9030处理器系列产品，支持的数据类型为：half。
    





## 调用示例

```
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
// 示例一：左矩阵在Global Memory
mm.SetTensorA(gm_a, isTransposeA);
mm.SetTensorB(gm_b);
if (tiling.isBias) {
    mm.SetBias(gmBias);
}
mm.IterateAll(gm_c);
mm.End();
// 示例二：左矩阵在Local Memory
mm.SetTensorA(local_a, isTransposeA);
// 示例三：设置标量数据
mm.SetTensorA(scalar_a, isTransposeA);
```

- [FusedLinearCrossEntropyLossGrad A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/nn/matmul/fused_linear_cross_entropy_loss_grad.md)
- [UpsampleBilinear2dAABackward A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/cv/image/upsample_bilinear_2d_aa_backward.md)
- [QuantMatmul 950 Direct Invoke 算子知识](../../../../../operators/direct/nn/matmul/quant_matmul_950.md)
- [grouped_matmul_add A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/gmm/grouped_matmul_add.md)
- [LSTM 950 Direct Invoke 算子知识](../../../../../operators/direct/nn/rnn/lstm_950.md)
- [grouped_matmul_add 950 ACLNN 算子知识](../../../../../operators/aclnn/transformer/gmm/grouped_matmul_add_950.md)
- [grouped_matmul_finalize_routing A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/gmm/grouped_matmul_finalize_routing.md)
- [MhcPreSinkhorn A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mhc/mhc_pre_sinkhorn.md)
- [RotaryPositionEmbedding A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/posembedding/rotary_position_embedding.md)

## 配套 API 与常见组合

- [SetTensorB](set_tensor_b.md)——同一 matmul 对象的左/右矩阵绑定缺一不可，卡片调用示例即 SetTensorA+SetTensorB+IterateAll 连续序列，matmul 范式必要构件
- [SetOrgShape](set_org_shape.md)——matmul kernel 对象复用/MDL 模板场景须先 SetOrgShape 再绑定操作数：卡片约束明文"本接口需要在 SetTensorA、SetTensorB、SetBias 及 SetSingleShape
- [TPosition](../../../basic_api/aux_data_structures/t_position.md)——SetTensorA 卡明示 LocalTensor 重载"支持的TPosition为TSCM/VECOUT"、GM 重载对应 MatmulType 的 TPosition::GM，位置配置决定操作数形态且须两侧一致
