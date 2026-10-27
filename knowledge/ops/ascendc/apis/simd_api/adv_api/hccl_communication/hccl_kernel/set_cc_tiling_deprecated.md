---
type: API
title: SetCcTiling（废弃）
description: 用于设置HCCL客户端通信算法的Tiling地址。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.data_movement.communication.collective_communication
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/HCCL_communication/HCCL_Kernel/SetCcTiling_deprecated.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- a2
- a3
---

# SetCcTiling（废弃）

> [!NOTE]说明
>该接口废弃，并将在后续版本移除，请不要使用该接口。请使用[SetCcTilingV2](set_cc_tiling_v2.md)接口设置通信算法的Tiling地址。

## 功能说明

用于设置HCCL客户端通信算法的Tiling地址。

## 函数原型

```
__aicore__ inline int32_t SetCcTiling(__gm__ void* ccOpTilingData)
```

## 参数说明

**表1**  接口参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| ccOpTilingData | 输入 | 通信算法的[Mc2CcTiling](../hccl_tiling/tiling_data_struct.md#table678914014562)参数的地址。Mc2CcTiling在Host侧计算得出，具体请参考[表2 Mc2CcTiling参数说明](../hccl_tiling/tiling_data_struct.md#table678914014562)，由框架传递到核函数（Kernel）中使用。 |

## 返回值说明

-   HCCL\_SUCCESS，表示成功。
-   HCCL\_FAILED，表示失败。

## 约束说明

-   参数相同的同一种通信算法在调用Prepare接口前只需要调用一次本接口，否则需要多次调用本接口。
-   同一种通信算法只支持设置一个ccOpTilingData地址；对于同一种通信算法，重复调用本接口会覆盖该通信算法的ccOpTilingData地址。
-   若调用本接口，必须与传initTiling地址的[Init](init_deprecated.md)接口配合使用，且Init接口在本接口前被调用。
-   若调用本接口，必须使用标准C++语法定义TilingData结构体的开发方式。

- [AlltoAllAllGatherBatchMatMul A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/allto_all_all_gather_batch_mat_mul.md)
- [BatchMatMulReduceScatterAlltoAll A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/batch_mat_mul_reduce_scatter_allto_all.md)
