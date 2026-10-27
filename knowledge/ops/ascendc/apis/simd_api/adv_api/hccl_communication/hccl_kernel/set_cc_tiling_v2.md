---
type: API
title: SetCcTilingV2
description: 用于设置HCCL客户端中某个通信算法配置的TilingData地址。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.data_movement.communication.collective_communication
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/HCCL_communication/HCCL_Kernel/SetCcTilingV2.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
- a2
- a3
source_headers:
- include/adv_api/hccl/hccl.h
---

# SetCcTilingV2

## 功能说明

用于设置HCCL客户端中某个通信算法配置的TilingData地址。

## 函数原型

```
__aicore__ inline int32_t SetCcTilingV2(uint64_t offset)
```

## 参数说明

**表1**  接口参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| offset | 输入 | 通信算法配置[Mc2CcTiling](../hccl_tiling/tiling_data_struct.md#table678914014562)参数地址相对于[Mc2InitTiling](../hccl_tiling/tiling_data_struct.md#table4835205712588)起始地址的偏移。[Mc2CcTiling](../hccl_tiling/tiling_data_struct.md#table678914014562)在Host侧计算得出，具体请参考[表2 Mc2CcTiling参数说明](../hccl_tiling/tiling_data_struct.md#table678914014562)，由框架传递到核函数（Kernel）中使用。 |

## 返回值说明

-   HCCL\_SUCCESS，表示成功。
-   HCCL\_FAILED，表示失败。

## 约束说明

-   若调用本接口，必须保证[InitV2](init_v2.md)在本接口前被调用。
-   Tiling参数相同的同一种通信算法在调用Prepare接口前，只需要调用一次本接口，请参考调用示例：[类型不同、Tiling参数不同的通信](#li71505119260)。
-   对于同一种通信算法，如果Tiling参数不同，重复调用本接口会覆盖之前的Tiling参数地址，因此需要在调用Prepare接口后再调用本接口设置新的Tiling参数。请参考调用示例：[类型相同、Tiling参数不同的通信](#li1163031215116)。
-   若调用本接口，必须使用标准C++语法定义TilingData结构体的开发方式。

## 调用示例

-   用户自定义TilingData结构体：

    ```
    class UserCustomTilingData {
        AscendC::tiling::Mc2InitTiling initTiling;
        AscendC::tiling::Mc2CcTiling allGatherTiling;
        AscendC::tiling::Mc2CcTiling allReduceTiling1;
        AscendC::tiling::Mc2CcTiling allReduceTiling2;
        CustomTiling param;
    };
    ```

-   <a name="li71505119260"></a>类型不同、Tiling参数不同的通信

    ```
    extern "C" __global__ __aicore__ void userKernel(GM_ADDR aGM, GM_ADDR workspaceGM, GM_ADDR tilingGM)
    {
        REGISTER_TILING_DEFAULT(UserCustomTilingData);
        GET_TILING_DATA_WITH_STRUCT(UserCustomTilingData, tilingData, tilingGM);

        Hccl hccl;
        GM_ADDR contextGM = AscendC::GetHcclContext<0>();
        hccl.InitV2(contextGM, &tilingData);

        // 在下发任务之前，通过SetCcTilingV2设置对应的tiling
        if (hccl.SetCcTilingV2(offsetof(UserCustomTilingData, allGatherTiling)) != HCCL_SUCCESS ||
            hccl.SetCcTilingV2(offsetof(UserCustomTilingData, allReduceTiling1)) != HCCL_SUCCESS) {
            return;
        }
        const auto agHandleId = hccl.AllGather<true>(sendBuf, recvBuf, dataCount, HcclDataType::HCCL_DATA_TYPE_FP16);
        hccl.Wait(agHandleId);

        const auto arHandleId = hccl.AllReduce<true>(
            sendBuf, recvBuf, dataCount, HcclDataType::HCCL_DATA_TYPE_FP16, HcclReduceOp::HCCL_REDUCE_SUM);
        hccl.Wait(arHandleId);

        hccl.Finalize();
    }
    ```

-   <a name="li1163031215116"></a>类型相同、Tiling参数不同的通信

    ```
    extern "C" __global__ __aicore__ void userKernel(GM_ADDR aGM, GM_ADDR workspaceGM, GM_ADDR tilingGM)
    {
        REGISTER_TILING_DEFAULT(UserCustomTilingData);
        GET_TILING_DATA_WITH_STRUCT(UserCustomTilingData, tilingData, tilingGM);

        Hccl hccl;
        GM_ADDR contextGM = AscendC::GetHcclContext<0>();
        hccl.InitV2(contextGM, &tilingData);

        // 在下发通信任务之前，通过SetCcTilingV2设置对应的Tiling参数地址
        if (hccl.SetCcTilingV2(offsetof(UserCustomTilingData, allReduceTiling1)) != HCCL_SUCCESS) {
            return;
        }
        const auto arHandleId1 = hccl.AllReduce<true>(
            sendBuf, recvBuf, dataCount, HcclDataType::HCCL_DATA_TYPE_FP16, HcclReduceOp::HCCL_REDUCE_SUM);
        hccl.Wait(arHandleId1);

        // 第二次AllReduce的Tiling参数与第一次不同，在第一次Prepare之后再调用SetCcTilingV2
        if (hccl.SetCcTilingV2(offsetof(UserCustomTilingData, allReduceTiling2)) != HCCL_SUCCESS) {
            return;
        }
        const auto arHandleId2 = hccl.AllReduce<true>(
            sendBuf, recvBuf, dataCount, HcclDataType::HCCL_DATA_TYPE_FP16, HcclReduceOp::HCCL_REDUCE_SUM);
        hccl.Wait(arHandleId2);

        hccl.Finalize();
    }
    ```

- [MatmulAllReduce 950 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/matmul_all_reduce_950.md)
- [MatmulAlltoAll A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/matmul_allto_all.md)
- [AlltoAllMatmul A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/allto_all_matmul.md)
- [AlltoAllQuantMatmul 950 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/allto_all_quant_matmul_950.md)
- [MatmulAlltoAll 950 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/matmul_allto_all_950.md)
- [MoeDistributeCombine A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/moe_distribute_combine.md)
- [MoeDistributeDispatch A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/moe_distribute_dispatch.md)
- [QuantMatmulAlltoAll 950 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/quant_matmul_allto_all_950.md)
- [QuantMatmulAllReduce 950 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/quant_matmul_all_reduce_950.md)
