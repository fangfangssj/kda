---
type: API
title: REGIST_MATMUL_OBJ
description: 初始化Matmul对象。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.programming_model.execution_unit.cube
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/cube_compute/Matmul_Kernel/REGIST_MATMUL_OBJ.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms: []
---

# REGIST\_MATMUL\_OBJ

## 功能说明

初始化Matmul对象。

## 函数原型

```
REGIST_MATMUL_OBJ(tpipe, workspace, ...)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| tpipe | 输入 | Tpipe对象。 |
| workspace | 输入 | 系统workspace指针。 |
| ... | 输入 | 可变参数，传入Matmul对象和与之对应的Tiling结构，要求Tiling结构的数据类型为[TCubeTiling结构](../matmul_tiling/t_cube_tiling_struct.md#tcubetiling-struct)。<br><br>Tiling参数可以通过Host侧[GetTiling](../matmul_tiling/get_tiling.md)接口获取，并传递到核函数（Kernel）侧使用。 |

## 返回值说明

无

## 约束说明

-   在分离模式中，本接口必须在[InitBuffer](../../../basic_api/resource_management/t_pipe/init_buffer.md)接口前调用。
-   在程序中，最多支持定义4个Matmul对象。
-   在Matmul的内部实现中使用了[CrossCoreSetFlag](../../../basic_api/sync_control/inter_core_sync/cross_core_set_flag_isasi.md)和[CrossCoreWaitFlag](../../../basic_api/sync_control/inter_core_sync/cross_core_wait_flag_isasi.md)进行核间同步控制，因此开发者同时使用CrossCoreSetFlag（或CrossCoreWaitFlag）和Matmul时，需注意避免flagId冲突，各接口flagId占用的完整说明（含各NPU架构版本差异）参见[核间同步flagId占用说明](../../../basic_api/sync_control/inter_core_sync/key_features.md#inter_core_sync_flagid_usage)。
-   当代码中只有一个Matmul对象时，本接口可以不传入tiling参数，通过[Init](init.md)接口单独传入tiling参数。
-   当代码中有多个Matmul对象时，必须满足Matmul对象与其tiling参数一一对应，依次传入，具体方式请参考调用示例。
-   在分离模式中，调用本接口后，AIC核不会主动执行接口，仅在AIV核执行到下述接口后，才会触发AIC核的执行，其他接口则不会在AIC核上执行。
    -   [Matmul核函数（Kernel）侧接口](index.md)。
    -   [DataCopy(GMToL1连续数据搬运)](../../../basic_api/cube_compute_isasi/cube_compute_load/data_copy_gm_to_l1_continuous.md)、[DataCopy(GMToL1高维切分数据搬运)](../../../basic_api/cube_compute_isasi/cube_compute_load/data_copy_gm_to_l1_highdim_split.md)、[DataCopy(GMToL1随路转换-ND2NZ搬运)](../../../basic_api/cube_compute_isasi/cube_compute_load/data_copy_gm_to_l1_nd2_nz.md)。

## 调用示例

```
Tpipe pipe;
// 推荐：初始化单个matmul对象，传入tiling参数
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
// 推荐：初始化多个matmul对象，传入对应的tiling参数
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm1, mm1tiling, mm2, mm2tiling, mm3, mm3tiling, mm4, mm4tiling);
// 初始化单个matmul对象，未传入tiling参数。注意，该场景下需要使用Init接口单独传入tiling参数。这种方式将matmul对象的初始化和tiling的设置分离，比如，Tiling可变的场景，可通过这种方式多次对Tiling进行重新设置
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm);
mm.Init(&tiling);
```

- [grouped_matmul_add A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/gmm/grouped_matmul_add.md)
- [grouped_matmul_add 950 ACLNN 算子知识](../../../../../operators/aclnn/transformer/gmm/grouped_matmul_add_950.md)
- [UpsampleBicubic2dAA A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/cv/image/upsample_bicubic_2d_aa.md)
- [DeformableConv2d A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/nn/conv/deformable_conv2d.md)
- [FlashAttentionScoreGrad A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/attention/flash_attention_score_grad.md)
- [UpsampleBicubic2dAAGrad A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/cv/image/upsample_bicubic_2d_aa_grad.md)
- [FlashAttentionUnpaddingScoreGrad A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/attention/flash_attention_unpadding_score_grad.md)
- [UpsampleBilinear2dAABackward A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/cv/image/upsample_bilinear_2d_aa_backward.md)
- [NsaSelectedAttention A2 ACLNN 算子知识](../../../../../operators/aclnn/transformer/attention/nsa_selected_attention.md)

## 配套 API 与常见组合

- [Matmul使用说明](matmul_usage.md)
- [MatmulConfig](matmul_config.md)
- [GetMatmulApiTiling](get_matmul_api_tiling.md)
- [Matmul 高阶替换与常驻回调坐标](../../../../../guides/matmul/high_level_replacement_and_resident_callbacks.md)
- [双 Matmul 共享与卷积两级缓存模式](../../../../../guides/matmul/two_matmul_share_and_convolution_buffer_modes.md)
