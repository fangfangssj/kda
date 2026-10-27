---
type: API
title: SetOrgShape
description: 设置Matmul计算原始完整的形状M、N、K，单位为元素个数。用于运行时修改shape，比如复用同一个Matmul对象，从不同的矩阵块取数据计算。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.programming_model.execution_unit.cube
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/cube_compute/Matmul_Kernel/SetOrgShape.md
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

# SetOrgShape

## 功能说明

设置Matmul计算原始完整的形状M、N、K，单位为元素个数。用于运行时修改shape，比如复用同一个Matmul对象，从不同的矩阵块取数据计算。

## 函数原型

```
__aicore__ inline void SetOrgShape(int orgM, int orgN, int orgK)
```

```
__aicore__ inline void SetOrgShape(int orgM, int orgN, int orgKa, int orgKb, int orgKc = 0)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| orgM | 输入 | 设置原始完整的形状M大小，单位为元素。 |
| orgN | 输入 | 设置原始完整的形状N大小，单位为元素。 |
| orgK | 输入 | 设置原始完整的形状K大小，单位为元素。原始完整形状Ka=Kb时可设置。 |
| orgKa | 输入 | 设置矩阵A原始完整的形状Ka大小，单位为元素。 |
| orgKb | 输入 | 设置矩阵B原始完整的形状Kb大小，单位为元素。 |
| orgKc | 输入 | 设置输出C矩阵的N，单位为元素。需要输入B矩阵的N和输出C矩阵的N不一样时可设置，默认为0（即使用B矩阵的N，不进行修改）。 |


> [!CAUTION]注意
>-   对于Ascend 950PR&950DT系列产品上使用[MDL模板](matmul_config.md)创建的Matmul对象，L1上数据的形状与Tiling侧接口[SetOrgShape](../matmul_tiling/set_org_shape.md)中的orgMIn/orgNIn/orgKIn/orgKaIn/orgKbIn一致时，不必须调用此接口。
>-   对于Ascend 950PR&950DT系列产品上使用[MDL模板](matmul_config.md)创建的Matmul对象，L1上数据的形状与Tiling侧接口[SetOrgShape](../matmul_tiling/set_org_shape.md)中的orgMIn/orgNIn/orgKIn/orgKaIn/orgKbIn不一致时，必须调用本接口指定GM/L1上的orgM/orgN/orgK/orgKa/orgKb。
>    例如，使用[MDL模板](matmul_config.md)时，输入矩阵A在L1、输入矩阵B在GM的场景，L1上A的形状大小与Tiling侧原始的orgMIn/orgKIn/orgKaIn不一致时，调用SetOrgShape\(orgM, orgN, orgK\)/SetOrgShape\(orgM, orgN, orgKa, orgKb\)接口指定L1上A矩阵相关参数orgM/orgK/orgKa。


## 返回值说明

无

## 约束说明

本接口需要在SetTensorA接口、SetTensorB接口、SetBias接口及SetSingleShape接口前调用。

## 调用示例

-   设置矩阵原始完整的形状

    ```
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
    mm.SetTensorA(gm_a);
    mm.SetTensorB(gm_b);
    mm.SetBias(gm_bias);
    mm.IterateAll(gm_c);
    //  复用mm对象
    mm.SetOrgShape(orgM, orgN, orgK);
    mm.SetTensorA(gm_a1);
    mm.SetTensorB(gm_b1);
    mm.SetBias(gm_bias1);
    mm.IterateAll(gm_c1);
    ```


-   对于Ascend 950PR&950DT系列产品上使用[MDL模板](matmul_config.md)的Matmul对象，设置GM或L1上完整的形状

    ```
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
    for (int m = 0; m < mIter_; m++) {
         for (int n = 0; n < nIter_; n++) {
              for (int k = 0; k < kIter_; k++) {
                   // 复用mm，指定A在L1和B在GM上的shape
                   mm.SetOrgShape(alignedSingleM, tiling.N, alignedSingleK, tiling.Kb, tiling.N);
                   mm.SetSingleShape(curBaseM, curBaseN, curBaseK);
                   mm.SetTensorA(tscm_a[offset_a]); // Set aMatrix tscm input
                   mm.SetTensorB(gm_b[offset_b]);
                   mm.SetBias(gm_bias[offset_bias]);
                   mm.Iterate(k != 0);
                }
                matmulObj.GetTensorC(gm_c[offset_c]);
         }
    }
    ```

- [UpsampleBilinear2dAABackward A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/cv/image/upsample_bilinear_2d_aa_backward.md)
- [MhcPreSinkhorn A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mhc/mhc_pre_sinkhorn.md)
- [RotaryPositionEmbedding A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/posembedding/rotary_position_embedding.md)

## 配套 API 与常见组合

- [SetTensorA](set_tensor_a.md)——matmul kernel 对象复用/MDL 模板场景须先 SetOrgShape 再绑定操作数：卡片约束明文"本接口需要在 SetTensorA、SetTensorB、SetBias 及 SetSingleShape
- [SetTensorB](set_tensor_b.md)——SetTensorB 同在该卡前置约束列表与源码同对象序列中，org 形状与 B 操作数形状契约一致
